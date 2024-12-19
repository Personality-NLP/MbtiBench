from typing import Any, Optional

from typing_extensions import assert_never

from mbtibench.enums import LabelType, MbtiDimension, PromptMethodName
from mbtibench.prompt import (
    PromptMethod,
    PsycotMethodHard,
    PsycotMethodSoft,
    StepByStepMethodHard,
    StepByStepMethodSoft,
    ZeroShotMethodSoft,
)


class DreadditPromptMethod(PromptMethod):
    def __init__(self, dim: MbtiDimension, text: str):
        super().__init__(dataset=None, dim=dim, user_posts=text)


def get_prompt_method_cls(method: PromptMethodName, label_type: LabelType) -> Any:
    assert method != PromptMethodName.FEW_SHOT
    if label_type == LabelType.HARD:
        method_mapper = {
            PromptMethodName.ZERO_SHOT: DreadditZeroShotHard,
            PromptMethodName.STEP_BY_STEP: DreadditStepByStepHard,
            PromptMethodName.PSYCOT: DreadditPsycotHard,
        }
    elif label_type == LabelType.SOFT:
        method_mapper = {
            PromptMethodName.ZERO_SHOT: DreadditZeroShotSoft,
            PromptMethodName.STEP_BY_STEP: DreadditStepByStepSoft,
            PromptMethodName.PSYCOT: DreadditPsycotSoft,
        }
    else:
        assert_never()
    return method_mapper[method]


class DreadditZeroShotSoft(DreadditPromptMethod, ZeroShotMethodSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following post from a user, determine the {self._dim.rank} dimension ({self._dim.full_name}) of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}, output your final score by strictly following this format: "[[score]]" and do not give reason."""


class DreadditZeroShotHard(DreadditZeroShotSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following post from a user, determine the {self._dim.rank} dimension of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. Predicting whether the author is {self._dim.full_hard_choices}. Provide a choice in the format: 'CHOICE: <A/B>' and do not give reason"""


class DreadditStepByStepSoft(DreadditPromptMethod, StepByStepMethodSoft):
    @property
    def _system_prompt(self):
        return f"""Given the following post from a user, determine the {self._dim.rank} dimension ({self._dim.full_name}) of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}, output your final score by strictly following this format: "[[score]]". Let's think step by step"""


class DreadditStepByStepHard(DreadditStepByStepSoft, StepByStepMethodHard):
    @property
    def _system_prompt(self):
        return f"""Given the following post from a user, determine the {self._dim.rank} dimension of Myers-Briggs Type Indicator (MBTI) personality type best fits the user. Predicting whether the author is {self._dim.full_hard_choices}. Let's think step by step. Finally provide a choice in the format: 'CHOICE: <A/B>'"""


class DreadditPsycotSoft(DreadditPromptMethod, PsycotMethodSoft):
    @property
    def _system_prompt(self):
        return f"""You are an AI assistant who specializes in text analysis and I am User. We will complete a text analysis task together through a multi-turn dialogue. The task is as follows: we have a post from an author, and at each turn I will give you a Question about the author. According to the author's answer, you need to choose the possible options ONLY. DO NOT give your reason, just wait for the next user input. After opting all the choices, I will ask you the {self._dim.rank} dimension ({self._dim.full_name}) score of the author. You need to rate the statement with a score 1-9, where 1=more {self._dim.first_letter} and 9=more {self._dim.second_letter}.\n{self._user_posts}\n"""


class DreadditPsycotHard(DreadditPsycotSoft, PsycotMethodHard):
    @property
    def _system_prompt(self):
        return f"""You are an AI assistant who specializes in text analysis and I am User. We will complete a text analysis task together through a multi-turn dialogue. The task is as follows: we have a post from an author, and at each turn I will give you a Question about the author. According to the author's answer, you need to choose the possible options ONLY. DO NOT give your reason, just wait for the next user input. After opting all the choices, I will ask you if the author is {self._dim.full_hard_choices}, and then you need to give your choice.\n{self._user_posts}\n"""


class DreadditDownstream:
    def __init__(self, text: str, mbti_answer: Optional[str], labeltype: Optional[LabelType]):
        self._text = text

        assert (mbti_answer is not None and labeltype is not None) or (mbti_answer is None and labeltype is None)
        self._mbti_answer = mbti_answer
        self._labeltype = labeltype

        self._with_mbti_info = mbti_answer is not None

    @property
    def _system_prompt(self):
        return "You are a helpful assistant."

    @property
    def _turns(self):
        if self._with_mbti_info:
            return [
                {
                    "role": "user",
                    "content": f"""You need to analyze the emotional stress in a post. I will give you a post from one poster and the poster's Myers-Briggs Type Indicator (MBTI) personality type to help you make decision. You need to judge whether the poster is likely to suffer from very severe stress or not. Please analyze the post based on personality information step by step and give the answer in strict accordance with the format [[yes]] or [[no]] at the end.\n{self._text}\nPoster's MBTI type:{self._mbti_answer}""",
                },
                {
                    "role": "assistant",
                    "content": "[[PLACEHOLDER]]",
                },
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": f"""You need to analyze the emotional stress in a post. I will give you a post from one poster and you need to judge whether the poster is likely to suffer from very severe stress or not. Please analyze the post step by step and give the answer in strict accordance with the format [[yes]] or [[no]] at the end.\n{self._text}""",
                },
                {
                    "role": "assistant",
                    "content": "[[PLACEHOLDER]]",
                },
            ]

    @property
    def prompts(self):
        return [{"role": "system", "content": self._system_prompt}] + self._turns
