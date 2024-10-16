from typing import Any, Dict, List

from typing_extensions import assert_never

from .enums import LabelType, MbtiDimension, PromptMethodName, SubDataset


class PromptMethod:
    def __init__(self, dataset: SubDataset, dim: MbtiDimension, user_posts: str):
        self._dataset = dataset
        self._dim = dim
        self._user_posts = user_posts

    @property
    def _system_prompt(self):
        raise NotImplementedError

    @property
    def _turns(self):
        raise NotImplementedError

    @property
    def prompts(self):
        return [{"role": "system", "content": self._system_prompt}] + self._turns


def get_prompt_method_cls(method: PromptMethodName, label_type: LabelType) -> Any:
    if label_type == LabelType.HARD:
        method_mapper = {
            PromptMethodName.ZERO_SHOT: ZeroShotMethodHard,
            PromptMethodName.STEP_BY_STEP: StepByStepMethodHard,
            PromptMethodName.FEW_SHOT: FewShotMethodHard,
            PromptMethodName.PSYCOT: PsycotMethodHard,
        }
    elif label_type == LabelType.SOFT:
        method_mapper = {
            PromptMethodName.ZERO_SHOT: ZeroShotMethodSoft,
            PromptMethodName.STEP_BY_STEP: StepByStepMethodSoft,
            PromptMethodName.FEW_SHOT: FewShotMethodSoft,
            PromptMethodName.PSYCOT: PsycotMethodSoft,
        }
    else:
        assert_never()
    return method_mapper[method]


class ZeroShotMethodSoft(PromptMethod):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension ({self._dim.full_name}) of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}, output your final score by strictly following this format: "[[score]]" and do not give reason."""

    @property
    def _turns(self):
        return [{"role": "user", "content": f"{self._user_posts}"}, {"role": "assistant", "content": "[[PLACEHOLDER]]"}]


class ZeroShotMethodHard(ZeroShotMethodSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. Predicting whether the author is {self._dim.full_hard_choices}. Provide a choice in the format: 'CHOICE: <A/B>' and do not give reason"""


class StepByStepMethodSoft(PromptMethod):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension ({self._dim.full_name}) of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}, output your final score by strictly following this format: "[[score]]". Let's think step by step"""

    @property
    def _last_turn(self):
        return f"""According to above, what is the score of {self._dim} dimension. Output your final score by strictly following this format: "[[score]]" and do not give reason."""

    @property
    def _turns(self):
        return [
            {"role": "user", "content": self._user_posts},
            {"role": "assistant", "content": "[[PLACEHOLDER]]"},
            {"role": "user", "content": self._last_turn},
            {"role": "assistant", "content": "[[PLACEHOLDER]]"},
        ]


class StepByStepMethodHard(StepByStepMethodSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. Predicting whether the author is {self._dim.full_hard_choices}. Let's think step by step. Finally provide a choice in the format: 'CHOICE: <A/B>'"""

    @property
    def _last_turn(self):
        return f"""According to above, the author is more likely to be: {self._dim.full_hard_choices}. Provide a choice in the format: "CHOICE: <A/B>" and do not give the explanation."""


class FewShotMethodSoft(PromptMethod):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension ({self._dim.full_name}) of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}, output your final score by strictly following this format: "[[score]]" and do not give reason."""

    @property
    def _turns(self):
        (first_shot, first_score), (second_shot, second_score) = self._get_shots()
        return [
            {
                "role": "user",
                "content": f"""Consider the first example: {first_shot}\nThe score is {first_score}\nConsider the second example: {second_shot}\nThe score is {second_score}\nConsider the third example: {self._user_posts}\nThe choice is """,
            },
            {"role": "assistant", "content": "[[PLACEHOLDER]]"},
        ]

    @property
    def _shots(self) -> Dict[str, List]:
        return {
            SubDataset.KAGGLE.value: [
                (
                    {
                        "shots": """Post 1: I play Sitar as well as sing Indian Classical music.  So obviously this is artistic, and useful too - mostly for me when I am blue and sometimes to others too.  It's very much comforting for me -...; Post 2: About 95% of he time I have taken the MBTI tests, I have been typed as an X, which I identify with. However, recently in two tests I took, I was typed as INTx and X, the latter of which I...; Post 3: Books -   1. Romeo and Juliet 2. The lost symbol 3. Timeline 4. Coma 5. Hamlet; Post 4: We get really mad when people are irrational and too much emotional:angry:.   These things cloud the ability to see things clearly.  Also it's maddening when people use wrong grammar and words.; Post 5: I am mostly at home - reading, or watching TV or surfing internet. These are the typical places you would find an INJ; Post 6: I am currently not playing Sitar though. just couldn't due to the seemingly endless load of enginnering submissions.:sad:  Now very much likely to resume in next month.:happy:; Post 7: My major is Engineering, specialization in Computer Science and Engineering.:happy:; Post 8: Oh by the way, I have also read many Marathi books, since Marathi is my mother tongue. 'Yayati' is one of those books.; Post 9: Mu favorite types of books - science, science fiction, fiction, novels etc.   State of Denial  State of Fear Romeo and Juliet The Lord of the Rings trilogy Nostradamus A Brief History of...; Post 10: Name - Sudeep, Male  Location - Born in Sangli, Maharashtra, India. The name Sangli has its origin in Marathi language. In marathi 'Saha' means 'Six' and 'Galli' means 'Lanes'. So in original city,...; """,
                        "softlabel": {
                            MbtiDimension.EI: 2,
                            MbtiDimension.SN: 1.5,
                            MbtiDimension.TF: 9,
                            MbtiDimension.JP: 7,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.first_letter,
                            MbtiDimension.SN: MbtiDimension.SN.first_letter,
                            MbtiDimension.TF: MbtiDimension.TF.second_letter,
                            MbtiDimension.JP: MbtiDimension.JP.second_letter,
                        },
                    }
                ),
                (
                    {
                        "shots": """Post 1: hi i start 7th grade tomorrow i'm really nervous ;-; help; Post 2: 2013 2012 2017 2011 2016 2014 2015 2010 (ugh i hated baby and too much kesha); Post 3: i'm so glad there are more 2000's kids on here. it's good to know i'm not the only one ;-;; Post 4: that's cool both of our birthdays are in february lol; Post 5: you're really late. i already had my birthday this year; Post 6: only 6 more months until i'm officially a teenager; Post 7: an ice pack* it's fine ;-;; Post 8: oww you child abuser :sad:; Post 9: i know...trust me i'm not like that anymore but i must've really hated his friend for whatever reason i can't remember ;-;; Post 10: i don't remember much of the 2000s at all but i do have some memories of me when i was 4 years old which was in 2009..one memory was me at the beach playing in the water with my dad and my brother....; Post 11: i'm so sorry you had to go through that ;-;; Post 12: finally somebody on here around my age. and i did something similar to that when i was 4 and my mom was buying me clothes but i wanted to get toys instead but she told me no. i was upset at one of my...; Post 13: Yeah that must be why this thread says people born from 2000-2008; Post 14: Your post just made me feel really nostalgic ;-; my older brother had a collection of silly bandz that he wore and he kept them away from me so i wouldn't steal any and i begged my mom so hard for...; Post 15: but 2010 was 7 years ago :sad:; Post 16: Any song that is from the black eyed peas and lmfao will give me strong nostalgia ;-;; Post 17: Wow finally someone who's around my age on here (i was born in 2005); Post 18: um...i know this is about remembering the 2000's, but i'm born in 2005 and i remember every year in the 2010's that happened so far. so i disagree...; Post 19: I just realized this thread is for X's i hate myself now .-.; Post 20: To my 7 year old self: 1.don't be so shy. Take the opportunity to talk to ppl 2. enjoy your childhood while it lasts  To myself 5 years time: 1. don't neglect high school grades 2. did you...; Post 21: I help other people if they ask for help or seem like they need help. I ask for help when i need it.; Post 22: Hi and welcome to this site. I joined last month so i'm still quite new to this stuff too; Post 23: I did the bottle flipping with a group of friends at school and my bottle actually landed without falling. All my friends were like OOHHHHHHH; Post 24: I've been in plenty of fights with my older brother when i was younger. We would always fight over toys and stupid stuff. Sadly he usually won because he was older and stronger than me at the time....; Post 25: i do remember arguing with my brother in public if that counts. We were at a restaurant.He let me use his phone for a bit so i could play games on there and i accidentally dropped it to the...; Post 26: Won't even rate myself because i don't give a fuck; Post 27: Accidentally putting a bag of chips in the fridge; Post 28: Summer FTW. I love swimming and going to the beach and not to mention SUMMER BREAK WHOOOHOOO; Post 29: Things like you're good at this; Post 30: goodnight everyone!!; Post 31: Wish we could turn back time, to the good ol' days, When our momma sang us to sleep but now we're stressed out (oh). Wish we could turn back time (oh), to the good ol' days (oh), When our momma...; Post 32: I can't post either its really annoying; Post 33: Testing; Post 34: Ughhh this is sooo annoying im so mad; Post 35: This is sooo annoying i really hope it gets fixed ASAP because i'm starting to get mad; Post 36: Ughhhh this is really annoying i hope they fix it ;-;; Post 37: ughhh i'm having this problem too and my post count is stuck R.I.P; Post 38: I'm getting this too it's so annoying i hope they fix it; Post 39: I'm 12 but i still feel 8. I look WAY younger than 12. There's even girls now who are taller than me and everyone else looks just about their age. I don't want to be a teenager next year.. I'm not...; Post 40: I'm playing a game on my phone; Post 41: Nice to meet you too; Post 42: I have to charge my phone before going to bed it's a must and i don't really have anything next to my bed except a little lamp and an alarm clock; Post 43: 5/10   want to live near a beach and have a nice looking house; Post 44: Me as a little kid: Shy, quiet, imaginative, playful Me now: Lazy, more outgoing, hungry, bored; Post 45: Music ftw!; Post 46: I'm not reading a book right now because i'm on summer break!! :D; """,
                        "softlabel": {
                            MbtiDimension.EI: 7,
                            MbtiDimension.SN: 8.5,
                            MbtiDimension.TF: 1,
                            MbtiDimension.JP: 4,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.second_letter,
                            MbtiDimension.SN: MbtiDimension.SN.second_letter,
                            MbtiDimension.TF: MbtiDimension.TF.first_letter,
                            MbtiDimension.JP: MbtiDimension.JP.first_letter,
                        },
                    }
                ),
            ],
            SubDataset.PANDORA.value: [
                (
                    {
                        "shots": """Post 1: This is how it's going to be huh; Post 2: Well when you word it that way without context of course you're not going to understand it! So yes, a chandelier is near and dear of many of us.  :); Post 3: Gunk Shot = Dead Mega Gardevoir; Post 4: This picture is the follow-up to my [Most Used Pokemon on June 2015]( which in turn was inspired by a post to /r/Pokemon that you can find [here] ( I decided to change the idea slightly by including two sorts of stats: Unweighted (0) and 1760/1825. Unweighted functions exactly like it sounds, all Pokemon chosen from any player on the ladder, no matter how high or low, will all have the same impact on the stats. Meanwhile, 1760/1825 stats favor the choices of the ""top"" players of the ladder based on their Glicko-1 rating, with lower players having less of an impact depending on their own distance from 1760/1825. You can read more about the rating system [here]( with the resource for Showdown stats [here]( Cup wasn't included because the top 6 pokemon for each stat were the almost same (the exceptions were Gastly in 0 and Timburr in 1760), just in different locations. I concluded it wouldn't be very interesting to show.**EDIT:** I messed up the 1825 OU picture, Latios is actually 5th in usage instead of Latias.; Post 5: It's gotten this far, hope remains.; Post 6: The upcoming episode will be in the XYZ anime, which is more than probably going to be using ORAS moves and Pokemon. There's no evidence against it, so Gunk Shot could very well be achieved.; Post 7: TIL Japan's April Fools is a day laterEdit: I know they're ahead of us time-wise, I'm commenting on the fact that they trolled us on a different day; Post 8: Unpopular opinion incoming. Be warned. I can't believe THREE Eeveeloutions seriously won against two pseudo-legendaries and Alakazam. Would've been even worse if Tyranitar lost. Yeah, they're cute, but the fan base goes ridiculous for them.; Post 9: This is honestly such a beautiful picture. Condolences + prayers to you and your family.; Post 10: Typholosion ain't leaving RU for a while since it was banned from NU, BL3 is the only place it can end up in right now, and it didn't deter people from using it even when it was listed there. If you want further proof of ladder differences, check out  Dugtrio's and Amoonguss' OU ladder differences, those get me every time.; Post 11: I've just gone through so many emotions in a Budweiser commercial I can't; Post 12: Song in the commercial was Bad Blood by Nao. I can't believe a song I've been listening to on repeat today ended up in the Super Bowl commercials.; Post 13: To add to this, some laptops don't allow the cursor to appear continuously on the screen without fading due to brief idleness. It makes aiming and passing very difficult and hit or miss.; Post 14: I had the same problem, but a simple email to the developers got it solved.; Post 15: Go play on origin guys. This is your chance; Post 16: The main post at the top of this subreddit is a Google form to sign up for MLTP. Signing up doesn't mean you will automatically drafted into MLTP, but it does mean that you can be drafted into two lower leagues (NLTP and ALTP). If I have my facts right, ALTP is the lowest league and the one that anyone who signs up is ensured a spot on. Definitely download mumble. If your computer/laptop can't download mumble, smartphones have the app.; Post 17: can't wait for whatever new meme pops up to inevitably be beaten to death; Post 18: you're going to be disappointed in a few weeks then; Post 19: I am free from the tyranny of Juke Dough!; Post 20: where were u during the vegasfuel push?; Post 21: The OT bomb was frustrating, lucky, skillful, brilliant, etc. More importantly it was my fault and your success. Good games!; Post 22: Please think about context. Dubstep wasn't saying that specific phrase to get into an intellectual discussion about the state of racial rights in America. You bring up that phrase to try and either 1) intimidate someone or 2) shame someone's race. It's not at all incorrect to assume implications just because he didn't outright say something directly racist. Especially because it wasn't the only attempt dubstep made to rile him up using race.; Post 23: [**CAPALIERS IN 3**]( seriously though those were some tight games and i may or may not have squealed like a girl; Post 24: Welcome back the reunion of some man and zeus!!; Post 25: Do you realize that perhaps our players struggle with lag also, albeit to a potentially lesser degree, and that we still play the same game? If you're looking for the perfect scenario and the expected result, you're never going to find them. I'm not saying what you went through was fortunate, but you're demonizing us for being happy about a win. Your post is honestly bitter and I hope that your mood about this week gets better in the future.; Post 26: Am...am I reading these exchange wrong? Isn't TheIndian saying that you're doing away with halves, yet your post is suggesting that they are staying because of their ""benefits""?; Post 27: It's 1991 all over again :(; Post 28: SuperSans thanks for the work so far you've put in!; Post 29: Posting this here for stats purposes**Jukes for Korean Jesus (21J) vs Ballshevik Revolution (SVB)**G1H1 [2-1 21J]( [5-1 21J]( Total 7-2 21JG2H1 [4-1 21J]( [5-1 21J]( Total 9-2 21J; Post 30: Sadly I don't think anyone streamed it. ACAP might have saved a replay?; Post 31: not even linking the replay smhi wasnt *that* bad :P; Post 32: I'll take a look at this in a few to see what happened Edit: Game statistics on tagproleague were given despite the actual scores not being recorded. Were those games played as consolation or something? I updated the spreadsheet for now; Post 33: I saw Jack and Jill at a theater. Needless to say, it was a pretty terrible experience; Post 34: in regards to your last point - no one is suggesting that someone give away all their food, lol. and i think that deals well in addressing your whole point in general. if someone is suffering because they want to live on edge or kill themselves, they should be seeking support in any way imaginable or realizable - because that's not normal behavior, nor should it be tolerated as normal, even when all else has failed. i cant ever support the idea of ending ones life. this is coming from someone who has experienced ongoing bouts of depression.if you want to respond go ahead, but i don't feel like continuing this discussion beyond what i've already stated - since im a firm believer in what ive said; Post 35: [Shoutouts to an unclean desk and pictures I painted in third grade](; Post 36: I always forget how old this community is compared to other ones I've been involved in (i.e. Smogon, where some admins/mods are in high school). I don't even have a license yet lol; Post 37: Thanks. I'll definitely add it when I get my own laptop.; Post 38: Haha if my summer wasn't busy I would definitely be screaming all day on V2.; Post 39: First off, definitely lag. Second, summer break brings with it a whole lot of traveling, vacations, family time etc. It sounds ridiculous, but people having an irregular schedule hurts time for TagPro. During the school/work season, people leave time slots for it open and it's fairly consistent. And some playing in leagues see little reason to spend summer time on pubs AND competitive, so they get cut back on the former. On an individual level, both of those reasons have hurt my playing time tremendously.; Post 40: I like Del because of its layout and flow, but I do wish it had some better choke points. I can see it staying in rotation for quite some time. Love Blooper's gate concept. Mid spikes can get pretty frustrating because of that troll bomb, but overall the map is alright. Haven't played enough on Mode 7 yet. Looks like a defender's dream with a boost block.; Post 41: I never use sounds because I'm always listening to music. I'm guessing you've played competitive Tagpro before, was your experience good? Do people make a lot of accommodations, or did you just get used to it the way it normally works?; Post 42: Let me try to offer explanations for this mentality. 1. Near the end of a game where a team is losing, a player might disconnect to try and focus away from the game they left and on to a new game. By doing this they remove themselves of the burden of playing through the whole game where they felt it was impossible to win. This might be more common in players suffering from psychological effects of tilt. 2. Record-wise, I'd actually suggest it looks more appealing to the player. Obviously the % doesn't change, but a person who sees their record as 4-5-4 might be more satisfied than if it was 4-9. This is because either they focus on the win/loss of 4-5 as it seems like an easier problem to fix, or because they believe others might think the dcs weren't all losses and therefore paints them in a better light.Not trying to excuse this, but you did ask so I gave it my best shot.; Post 43: I've never legitimately cared about drama in subs. I enjoy my time until I don't, and move on.; Post 44: With all due respect to Miguel, from an outsider's perspective it seems like this could have been influenced by the outcome of the s2 finals (not necessarily because he lost, but because of what happened after). Of course that's simply speculation, and the reasons could be beyond that. I just hope he has a change of heart on his own terms and can get back to the scene because he's such a fun guy to follow.; Post 45: That wasn't the reason for his response, I think. A year ago one of his brothers had posted something on twitter that caused controversy, which then led to people asking/demanding for Tyler to respond.; Post 46: This ended up becoming a sports-themed album...Album Title: Play for Philly- Jürgen- Green as a Packer- This Game- Headlines - Already Gone- Laughter is So Damn Infectious- Promise to Cleveland- Important Indeed- Comebacks- Blood PressureBONUS tracks- JJ Watt- 7-1; Post 47: For as much as I don't care for this song on first listen, I'm glad she's at least staying true to her own words.; Post 48: Not me tbhAt least half of the songs are my jam and I'll defend twenty one pilots for life. TIWYCF, on the other hand, is bland.; Post 49: Sunday Morning AND Midnight City? Literally a perfect list; Post 50: I find it refreshing to see When We Were Young near the top ten. Hopefully that song doesn't receive as much overplay as Hello. It's a bit more relaxed of a song as well; Hello drags you up and down with sound.; """,
                        "softlabel": {
                            MbtiDimension.EI: 4,
                            MbtiDimension.SN: 7,
                            MbtiDimension.TF: 4,
                            MbtiDimension.JP: 7,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.first_letter,
                            MbtiDimension.SN: MbtiDimension.SN.second_letter,
                            MbtiDimension.TF: MbtiDimension.TF.first_letter,
                            MbtiDimension.JP: MbtiDimension.JP.second_letter,
                        },
                    }
                ),
                (
                    {
                        "shots": """Post 1: Yeah, just like you would shake hands with people who might ""invade your country"" as you put it.; Post 2: Exactly. Even dry bread has some sugar in it. Your teeth will thank you for it.; Post 3: Imagine  without this one flaw. It wouldn't be fair! Amirite?; Post 4: That was with a flourishing medieval economy and a clean environment. In nuclear winter, with everything fucked up, the number of people is not the issue.; Post 5: Reading worker dad and highly educated teacher mum. Socialist and anarchist tendencies in the worldview they raised me with. Was always in contact with socially conscious books, music and art (anarchist, communist, anti-fascist, hippie, or simply open-minded stuff). Had and still have many antifa and punk friends. Had a long; Post 6: Physical. I don't like the gadgety nature of e-book-readers. I want to hold something that is real and authentic, that can get smudges, rips, bends, smells like new or old, and that I can pick up from sidewalks or leave for others to find. I want to read from something; Post 7: Good for you. I happend to cold shoulder people I was interested in, just because I was bugged by others who took slightly more of my attention. I need to work heavily on that bias to turn it around. I find it really hard to kickstart a conversation with someone; Post 8: Humans are not only rational beings; we do have survival instincts and also collective instincts. Why do I find this important to point out?  Because ultimately I think there must come a point in everybodies life where certain conditions (some disaster f.i.) will prompt them to adapt. This adaptation; Post 9: This article struck me as surprisingly well-written. It's mind-boggling how much time people invest to convince us that people as corrupt as H. are not too bad by comparison. I assume they are trying to cope with the crushing reality that another sociopath is going to lie themselves into office; Post 10: They should rename themselves to Privix.; Post 11: In what way do you think culture affects your type? More specifically, what do you mean by low-key?And I think I'm confused by the word ""European"". That's even more ridiculous than saying I'm a ""German"". I mean, I was born here, raised here, and live here, but does that have; Post 12: That sounds quite interesting. Would you mind elaborating? What does your disorder come down to, and how did this rewiring work?; Post 13: There are no words. I wish you two make it out somehow.; Post 14: Oh, but that study was not just about eye-gazing. It's preceded by 36 increasingly personal questions. Here is another [article](  that has the questions in it.Anyway, what you say about the eye-gazing is very interesting. In my experiences in those communities I mentioned, there was also lots of eye-contact,; Post 15: That is one very interesting thought. I seem to have started to figure this out myself, this week actually, but I wasn't yet conscious of it. So, thanks!; Post 16: Yes. I meant to say you don't need these things per say, but the confidence they bring. And I totally agree that you can't fake it. I wanted to imply that once you look back, you may realize those symbols and all aren't much more than a vehicle for your; Post 17: Speak for yourself. You're not worth arguing with.; Post 18: I would never argue against that ;) For me, it was understanding that feelings are mostly the language of your body, that made me seriously question my all-too-confident thinking apparatus, that was just not going anywhere, because it never made *the connection* (to body and action).Life is about balance all; Post 19: None other than the ones that are earned. F.i. when an apprentice finds his master, or when a group chooses a leader/coordinator/moderator, bc they're the best for the job.; Post 20: Although I'm totally aware of that, my friends don't care to switch again :/; Post 21: Only when I start it, they use it. I think the desktop version is still missing such a feature, and some only use Telegram there. It's little more than a messenger with funny pictures for them. But ""at least it's not facebook!""  Services like the f-word, Dropbox, Windows, etc.; Post 22: Well, I'm poor too, but I really think bringing a child to this world is sort of the only thing a human is actually made to do. I also strongly believe I would love being a parent, passing on my knowledge and love. I'm aware of my privileges of not; Post 23: I must say I don't believe any religion, or any ideology even, can ever be the real root cause for actual human doing or wrongdoing. I also don't believe in singular causes for that matter. I think rape and violence, even colonialism, can be explained much more plausible by looking; Post 24: Oh boy! As long as radical left people will complain about being too poor or too bad at languages, to *just fucking move around*, there will be a real need for this. I know too many of them and they shouldn't have any excuse, I think ;)If you need any; Post 25: Growing old realizing I didn't have enough sex when I was young(er). Not realizing my full potential. Losing my right hand (for drawing), or losing my eyesight. Dark and scary depths under water.; Post 26: Growing old realizing I didn't have enough sex when I was young(er). Not realizing my full potential. Losing my right hand (for drawing), or losing my eyesight. Dark and scary depths under water.; Post 27: So in case of our example we should avoid assigning people to groups and labeling them, and rather let them self-organize in the proposed order, right?; Post 28: Flash is a super easy way to start learning animation, even for hand-drawn stuff. For the more interesting effects you'll need to move on to 3D and/or After Effects. Photoshop is nice to create many levels of items for a start, but for the animation itself I can only recommend; Post 29: It's a civil war, not an invasion. And it's caused by 12+ different groups, some of them supported by super powers. Who do you even fight against? And what do you fight for really, if your house is already destroyed and family and friends die? Maybe you would only leave; Post 30: Dude, whatever right you think you have don't matter if they can claim you're a terrorist. But let's be real, nobody will simply cite this thread to raze your home, if they really want to, they find something else that's illegal. And even more realistic, next time someone needs to; Post 31: Whole months? XD well okay, what I had in mind were the minutes before becoming limp again :p  Funny enough, I recently found a collection of aphorisms by Jurij Brezan, a Sorbic-German author, in my parents library. Some of these seem to fit quite nicely here:and:and one that just; Post 32: I will share this with a friend who does similar work like you do but doesn't read reddit. Thank you for what you do, it's so crucially important!; Post 33: Thank you. This also helped me see things more clearly.; Post 34: Hm, thank you for laying that out to me. It shifts an few things into perspective. I know trust is important, yet I seem to have assumed it's mostly mutual. But it does make sense that it's a bigger issue for women.Trust is a really interesting concept in general. For; Post 35: Me, when trying to explain to my mom what MBTI is, that I classify as , and how that knowledge might be useful.Every conversation becomes meta really quick, as I end up seeing how our incompatibility drags it off the cliff, and I just sit there thinking: why do I; Post 36: I will add that by the time you achieve any of these things you will realize it's in your head to begin with, and you didn't actually need it. Only confidence is crucial.; Post 37: In Germany you can get a temporary ID at your local BÃ¼rgeramt in no time.; Post 38: Very good point.If I'm not creating tangible things or processes I'm merely a consumer. And if associating with a scene is more important than building that life, then we're just entertaining each other. I don't want to be a consumer in this system, and that mutual entertainment is part of; Post 39: I wrote 'covered in poor logic'. Do I have to dig them out for you? Really?; Post 40: That's a lot of loaded questions towards the end. I'm not the one asked, so I won't answer specifically, but I think you either feel as if she had attacked the *people* from the US (rather than the structural violence of western corporate imperialism), and are now putting words in; Post 41: System of a Down.I have a playlist simply with all songs in a row, and never have to skip songs. I can't even say which songs are on which album because of that. So many favorites.; Post 42: Did your mum not love you?; Post 43: - Debt: the first 5000 years by David Graeber  - The Hero with a thousand Faces by Joseph Campbell  It doesn't look much, but those two and any companion literature has kept me quite satiated and satisfied. I started plenty of others too, ofc. Also a few novels.; Post 44: Focus on producing results. Meditate daily, to handle the distracting feelings about girls. Produce a music video. Organize a small festival. Make more art. Get bureaucratic shit done, so I can go travel in November. Practice my instrument often. Continue to fast-read the bible, trying to find/decrypt its allegorical meanings.; Post 45: Yes, moderation is crucial. I remember many times where I couldn't stop reading a series of ultra thrilling books, or watching whole seasons of certain shows. It makes so much sense to assemble a balanced daily rhythm that keeps one motivated. But then again,  gotta change something every day; Post 46: The 2016 US presidential elections will be rigged.; Post 47: Telling without explaining only works with *trust*, but if children are smart they will eventually notice how their parents can be full of shit too, and hence lose that trust.; Post 48: Can you 'endorse' things that are long gone? He only brought some historical analysis into this, acknowledging the context of a law. I don't see where it made him a liberal.; Post 49: It's a [fucking genius album]( is what it is. That music man! Makes your head bursts wide open like when taking Salvia divinorum! Well, almost ...edit: added link; Post 50: I think there is a Freudian chuckle to be had here, involving Muskies, and public money being *pumped* into cheap oversized phallus symbols ... aaand premature something-something?; """,
                        "softlabel": {
                            MbtiDimension.EI: 8,
                            MbtiDimension.SN: 1,
                            MbtiDimension.TF: 8,
                            MbtiDimension.JP: 4,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.second_letter,
                            MbtiDimension.SN: MbtiDimension.SN.first_letter,
                            MbtiDimension.TF: MbtiDimension.TF.second_letter,
                            MbtiDimension.JP: MbtiDimension.JP.first_letter,
                        },
                    }
                ),
            ],
            SubDataset.TWITTER.value: [
                (
                    {
                        "shots": """Post 1: Text: omg nag-crave tuloy ako bigla; Post 2: i don't like sofia black d'elia's character in gossip girl; Post 3: why am i not good in drawing; Post 4: ..:( nail fail; Post 5: hassle !di ako magaling !aaaand i don't have the same nail polish; Post 6: sama ka na sakin !:) ) ugh , what the flying f u c k; Post 7: tya for art in action music series; Post 8: life keeps getting in the way; Post 9: whenever we try , somehow the plan is always rearranged; Post 10: "" whenever you feel like criticising anyone , just remember that all the people in this world haven't had the advantages that you've had; Post 11: "" really , what's with you and tori box ?=) ) 5 things only you are judging about yourself | thought catalog; Post 12: ..tara tori box sa monday !dude what time did you leave ?sleep of the year !oh my god; Post 13: i need water; Post 14: just--fuck you , okay; Post 15: =) ) i'm fine; Post 16: "" the old must put aside their own happiness; Post 17: "" - - , 2012 my grandpa kinda looks like an older ralph fiennes; Post 18: types a well-thought of , surprisingly very long opinion on the parthenon marbles issue; Post 19: accidentally refreshes page while proofreading; Post 20: upuan talaga ?:) ) sa tokyo 2 na kami; Post 21: sa jollibee na pala; Post 22: gulo namin; Post 23: :( tagal !D: uy , una na daw kami; Post 24: ilang minutes nalang yan ?gawa nalang ako ng dropbox ?:( =) ) edit nyo , yung pang-poster talaga; Post 25: haha jk getting hooked on modern family !blame's on; Post 26: ;D zoobi doobi zoobi doobi pum para param pum so much layout ideas; Post 27: :'D i bought chocolates for my gramps yesterday so he cooked my favorite meal for lunch today; Post 28: i like this arrangement; Post 29: junior hs feels; Post 30: yeah , we're friends , but; Post 31: ..omg haha kahapon iniisip ko na kung ano yung mga ikkwento nila back home tungkol dyan eh; Post 32: :) ); Post 33: ..you want albums for christmas , right ?i'm getting you a copy of annebisyosa; Post 34: :D drained; Post 35: is gwapo "" : i finally saw coco martin !friend ko pala siya all along !"" crme brle + paranormal activity 3 on hbo with uncle bennett !:\ glaciers melting in the dead of the night and the superstars sucked into the supermassive heard sad news; Post 36: rip , nico; Post 37: :( there s never going to be another audrey hepburn or marilyn monroe | thought catalog hahaha !yesss !i do wonders !in sta; Post 38: : always bring biogesic tablets with you; Post 39: happy birthday , !excited for our date !i need a little push; Post 40: i am so done with games , okay; Post 41: whenever you feel like laughing at sarah jessica parker because she "" looks like a horse "" , just remember that that "" horse "" dated rdj; Post 42: wait what i look like a potato there haha; """,
                        "softlabel": {
                            MbtiDimension.EI: 9,
                            MbtiDimension.SN: 3,
                            MbtiDimension.TF: 6,
                            MbtiDimension.JP: 7,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.second_letter,
                            MbtiDimension.SN: MbtiDimension.SN.second_letter,
                            MbtiDimension.TF: MbtiDimension.TF.second_letter,
                            MbtiDimension.JP: MbtiDimension.JP.second_letter,
                        },
                    }
                ),
                (
                    {
                        "shots": """Post 1: 50 essays guaranteed to make you a better person but remember haha .; Post 2: i'll be here all night .; Post 3: and an alternate theme song; Post 4: you and chocolate.; Post 5: changing directions; Post 6: both of you sweet as can be .; Post 7: and the swell season rocks .; Post 8: until we tweet again , my friend .; Post 9: xo  years ago i stumbled onto the fact that orange looks good on me .; Post 10: you're welcome , christine :)  you're welcome .; Post 11: glad you're getting something to eat .; Post 12: you need three meals a day .; Post 13: i know , i'm super-annoying today .; Post 14: haha .; Post 15: it's honestly beyond me how someone could forget to eat .; Post 16: i get hungry about every four hours .; Post 17: haha .; Post 18: so your one meal is a feast .; Post 19: good .; Post 20: i don't want you wasting away .; Post 21: appropriate for your sweet personality .; Post 22: sugar an spice and everything nice .; Post 23: and chocolate .; Post 24: i'm all snakes and snails and puppy dog tails over here .; Post 25: mmm , and books .; Post 26: and annoying .; Post 27: don't forget that .; Post 28: that's great .; Post 29: that's the way we do it around here .; Post 30: and this one for bill; Post 31: i like the later stuff from sawyer brown; Post 32: their album " this thing called wantin ' and havin ' it all " is one of my favorite ...  albums of any genre .; Post 33: you ought to check it out sometime .; Post 34: it's probably on itunes .; Post 35: that's perfect . hilarious .  good .this is it .; Post 36: you're welcome .; Post 37: always happy to spread the word about good music .; Post 38: that album is solid from start to finish .; Post 39: no skipped songs .; Post 40: likes you .; Post 41: tolerates me , i think .; Post 42: yeah , that's one of my favorites , too .; Post 43: " some girls don't like boys like me , aww but some girls do " :); Post 44: nah , to like me , he'd have to know i exist .; Post 45: heh .  yay :) .; Post 46: yeah , i remember the name t graham brown , but never listened to him much .; Post 47: he's probably like , oh yeah i remember niles .; Post 48: i had forgotten about him .; Post 49: alright , pinocchio .; """,
                        "softlabel": {
                            MbtiDimension.EI: 4,
                            MbtiDimension.SN: 7,
                            MbtiDimension.TF: 3,
                            MbtiDimension.JP: 4,
                        },
                        "hardlabel": {
                            MbtiDimension.EI: MbtiDimension.EI.first_letter,
                            MbtiDimension.SN: MbtiDimension.SN.first_letter,
                            MbtiDimension.TF: MbtiDimension.TF.first_letter,
                            MbtiDimension.JP: MbtiDimension.JP.first_letter,
                        },
                    }
                ),
            ],
        }

    def _get_shots(self) -> List[List[str]]:
        few_shot_examples = self._shots

        first_shot = few_shot_examples[self._dataset][0]["shots"]
        first_score = f"[[{few_shot_examples[self._dataset][0]['softlabel'][self._dim]}]]"
        second_shot = few_shot_examples[self._dataset][1]["shots"]
        second_score = f"[[{few_shot_examples[self._dataset][1]['softlabel'][self._dim]}]]"

        return (first_shot, first_score), (second_shot, second_score)


class FewShotMethodHard(FewShotMethodSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following text from a user's social media posts, determine the {self._dim.rank} dimension of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. Predicting whether the author is {self._dim.full_hard_choices}. Provide a choice in the format: 'CHOICE: <A/B>' and do not give reason"""

    def _get_shots(self) -> List[List[str]]:
        few_shot_examples = self._shots

        first_shot = few_shot_examples[self._dataset][0]["shots"]
        first_score = f"CHOICE: {few_shot_examples[self._dataset][0]['hardlabel'][self._dim]}"
        second_shot = few_shot_examples[self._dataset][1]["shots"]
        second_score = f"CHOICE: {few_shot_examples[self._dataset][1]['hardlabel'][self._dim]}"

        return (first_shot, first_score), (second_shot, second_score)


class PsycotMethodSoft(PromptMethod):
    @property
    def _system_prompt(self):
        return f"""You are an AI assistant who specializes in text analysis and I am User. We will complete a text analysis task together through a multi-turn dialogue. The task is as follows: we have a set of posts written by an author, and at each turn I will give you a Question about the author. According to the author's posts, you need to choose the possible options ONLY. DO NOT give your reason, just wait for the next user input. After opting all the choices, I will ask you the {self._dim.rank} dimension ({self._dim.full_name}) score of the author. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}.\nAUTHOR'S POSTS: {self._user_posts}\n"""

    @property
    def _last_turn(self):
        return f'According to above, what is the score of {self._dim} dimension. Output your final score by strictly following this format: "[[score]]" and do not give reason.'

    @property
    def _turns(self):
        turns = []

        qa_id, qa_list = self._get_questionnaires(self._dim)
        for k, ques in enumerate(qa_list):
            if k == 0:
                statments = f'Q: {ques}. Provide a choice ID in the format: "CHOICE: <A/B/C>" only, and do not give the explanation. do not generate User input.'
            elif qa_id[k] == 7 or qa_id[k] == 25:
                statments = f'Q: {ques}. Provide a choice in the format: "CHOICE: <A/B/C/D>" only, and do not give the explanation. do not generate User input'
            else:
                statments = f'Q: {ques}. Provide a choice ID in the format: "CHOICE: <A/B/C>" only, and do not give the explanation. do not generate User input.'

            turns.extend([{"role": "user", "content": statments}, {"role": "assistant", "content": "[[PLACEHOLDER]]"}])

        turns.extend(
            [
                {"role": "user", "content": self._last_turn},
                {"role": "assistant", "content": "[[PLACEHOLDER]]"},
            ]
        )

        return turns

    def _get_questionnaires(self, dim: MbtiDimension):
        if dim == MbtiDimension.EI:
            qa_id = [3, 6, 9, 13, 16, 21, 24, 26, 29, 36, 43]
            qa_list = [
                'The author is usually: A: "A good mixer with gropus of people", B: "Quiet and reserved", or C: "Not sure whether A or B"',
                'Among the author\'s friends, the author is: A: "Full of news about everybody", B: "One of the last to hear what is going on", or C: "Not sure whether A or B"',
                'The author tends to have: A: "A broad range of friendships with many different people", B: "Deep friendship with very few people", or C: "Not sure whether A or B"',
                'When the author is with a group of people, the author is usually: A: "Join in the talk of the group", B: "Stand back and listen first", or C: "Not sure whether A or B"',
                'The author is: A: "Talk easily to almost anyone", B: "Find a lot to say only to certain people or under certain conditions", or C: "Not sure whether A or B"',
                'In a large group, the author is more often: A: "Introduce others", B: "Get introduced", or C: "Not sure whether A or B"',
                'When the author meets the new people, the author tells what they are interested in: A: "Right away", B: "Only after people to get to know the author", or C: "Not sure whether A or B"',
                'The author is usually: A: "Show their feelings freely", B: "Keep their feelings to themselves", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "QUIET", B: "HEARTY", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "RESERVED", B: "TALAKATIVE", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "CALM", B: "LIVELY", or C: "Not sure whether A or B"',
            ]
        elif dim == MbtiDimension.SN:
            qa_id = [2, 5, 10, 12, 15, 20, 23, 28, 31, 35, 38, 42, 45, 48]
            qa_list = [
                'If the author was a teacher, would they rather teach: A: "Facts-based courses", B: "Courses involving opinion or theory", or C: "Not sure whether A or B"',
                'In doing something that many other people do would the author rather: A: "Invent a way of their own", B: "Do it in the accepted way", or C: "Not sure whether A or B"',
                'Does the author admire more the people who are: A: "Normal-acting to never make themselves the center of attention", B: "Too original and individual to care whether they are the center of attention or not", or C: "Not sure whether A or B"',
                'Does the author usually get along better with: A: "Realistic people", B: "Imaginative people", or C: "Not sure whether A or B"',
                'In reading for pleasure, does the author: A: "Enjoy odd or original ways of saying things", B: "Like writers to say exactly what they mean", or C: "Not sure whether A or B"',
                'Would the author rather be considered: A: "A practical person", B: "An out-of-the-box-thinking person", or C: "Not sure whether A or B"',
                'Would the author rather has a friend: A: "Someone who is always coming up with new ideas", B: "Someone who has both feet on the ground", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "FACTS", B: "IDEAS", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "IMAGINATIVE", B: "MATTER-OF-FACT", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "STATEMENT", B: "CONCEPT", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "CREATE", B: "MAKE", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "CERTAINTY", B: "THEORY", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "FASCINATING", B: "SENSIBLE", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "LITERAL", B: "FIGURATIVE", or C: "Not sure whether A or B"',
            ]
        elif dim == MbtiDimension.TF:
            qa_id = [4, 14, 22, 30, 32, 33, 37, 39, 40, 44, 46, 47, 49, 50]
            qa_list = [
                'Does the author more often let: A: "Their heart rule their head", B: "Their head rule their heart", or C: "Not sure whether A or B"',
                'For the author, which is a higher compliment: A: "A person of real feeling", B: "A consistently reasonable person", or C: "Not sure whether A or B"',
                'Does the author usually: A: "Value emotion more than logic", B: "Value logic more than feelings", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "CONVINCING", B: "TOUCHING", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "BENEFITS", B: "BLESSINGS", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "PEACEMAKER", B: "JUDGE", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "ANALYZE", B: "SYMPATHIZE", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "DETERMINED", B: "DEVOTED", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "GENTLE", B: "FIRM", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "JUSTICE", B: "MERCY", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "FIRM-MINDED", B: "WARM HEARTED", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "FEELING", B: "THINKING", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "ANTICIPATION", B: "COMPASSION", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "HARD", B: "SOFT", or C: "Not sure whether A or B"',
            ]
        elif dim == MbtiDimension.JP:
            qa_id = [18, 1, 7, 8, 11, 17, 19, 25, 27, 34, 41]
            qa_list = [
                'When it is settled well in advance that the author will do a certain thing at a certain time, does the author find it: A: "Nice to be able to plan accordingly", B: "A little unpleasant to be tied down", or C: "Not sure whether A or B"',
                'When the author goes somewhere, would the author rather: A: "Plan what they will do and When", B: "Just go", or C: "Not sure whether A or B"',
                'Does the idea of making a list of what the author should get done over a weekend: A: "Help the author", B: "Stress the author", C: "Positively depress the author", or D: "Not sure whether A, B, or C"',
                'When the author have a special job to do, does the author like to: A: "Organize it carefully before they start", B: "Find out what is necessary as they go along", or C: "Not sure whether A or B"',
                'Does the author prefer to: A: "Arrange picnics, parties etc, well in advance", B: "Be free to do whatever to looks like fun when the time comes", or C: "Not sure whether A or B"',
                'Does following a schedule: A: "Appeal to the author", B: "Cramp the author", or C: "Not sure whether A or B"',
                'Is the author more successful: A: "At following a carefully worked out plan", B: "At dealing with the unexpected and seeing quickly what should be done", or C: "Not sure whether A or B"',
                'In author\'s daily work, does the author: A: "Usually plan their work so the author won\’t need to work under pressure", B: "Rather enjoy an emergency that makes their work against time", or C: "Hate to work under pressure", or D: "Not sure whether A, B, or C"',
                'Which word is more suitable for the author: A: "SCHEDULED", B: "UNPLANNED", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "SYSTEMATIC", B: "SPONTANEOUS", or C: "Not sure whether A or B"',
                'Which word is more suitable for the author: A: "SYSTEMATIC", B: "CASUAL", or C: "Not sure whether A or B"',
            ]
        else:
            assert_never()

        return qa_id, qa_list


class PsycotMethodHard(PsycotMethodSoft):
    @property
    def _system_prompt(self):
        return f"""You are an AI assistant who specializes in text analysis and I am User. We will complete a text analysis task together through a multi-turn dialogue. The task is as follows: we have a set of posts written by an author, and at each turn I will give you a Question about the author. According to the author's posts, you need to choose the possible options ONLY. DO NOT give your reason, just wait for the next user input. After opting all the choices, I will ask you if the author is {self._dim.full_hard_choices}, and then you need to give your choice.\nAUTHOR'S POSTS: {self._user_posts}\n"""

    @property
    def _last_turn(self):
        return f'According to above, the author is more likely to be: {self._dim.full_hard_choices}. Provide a choice in the format: "CHOICE: <A/B>" and do not give the explanation.'
