from enum import Enum

from typing_extensions import assert_never


class MbtiDimension(Enum):
    EI = "E/I"
    SN = "S/N"
    TF = "T/F"
    JP = "J/P"

    def __str__(self) -> str:
        return self.value

    @property
    def full_name(self) -> str:
        return f"{self.full_first_letter} or {self.full_second_letter}"

    @property
    def rank(self) -> str:
        if self == MbtiDimension.EI:
            return "first"
        elif self == MbtiDimension.SN:
            return "second"
        elif self == MbtiDimension.TF:
            return "third"
        elif self == MbtiDimension.JP:
            return "forth"
        else:
            assert_never()

    @property
    def first_letter(self) -> str:
        return self.value[0]

    @property
    def second_letter(self) -> str:
        return self.value[2]

    @property
    def full_first_letter(self) -> str:
        if self == MbtiDimension.EI:
            return "Extraversion"
        elif self == MbtiDimension.SN:
            return "Sensing"
        elif self == MbtiDimension.TF:
            return "Thinking"
        elif self == MbtiDimension.JP:
            return "Judging"
        else:
            assert_never()

    @property
    def full_second_letter(self) -> str:
        if self == MbtiDimension.EI:
            return "Introversion"
        elif self == MbtiDimension.SN:
            return "Intuition"
        elif self == MbtiDimension.TF:
            return "Feeling"
        elif self == MbtiDimension.JP:
            return "Perceiving"
        else:
            assert_never()

    @property
    def only_letter(self) -> str:
        return self.value.replace("/", "")

    @property
    def full_hard_choices(self) -> str:
        return f"A: {self.full_first_letter} or B: {self.full_second_letter}"


class SubDataset(Enum):
    KAGGLE = "kaggle"
    PANDORA = "pandora"
    TWITTER = "twitter"

    def __str__(self) -> str:
        return self.value


class PromptMethodName(Enum):
    ZERO_SHOT = "zero-shot"
    STEP_BY_STEP = "step-by-step"
    FEW_SHOT = "few-shot"
    PSYCOT = "psycot"

    def __str__(self) -> str:
        return self.value


class ModelName(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    QWEN2_7B = "qwen2-7b"
    QWEN2_72B = "qwen2-72b"
    LLAMA3_1_8B = "llama3.1-8b"
    LLAMA3_1_70B = "llama3.1-70b"

    def __str__(self) -> str:
        return self.value

    @property
    def is_gpt4(self) -> bool:
        return "gpt-4" in self.value

    @property
    def is_qwen2(self) -> bool:
        return "qwen2" in self.value

    @property
    def is_llama3_1(self) -> bool:
        return "llama3.1" in self.value


class MetricName(Enum):
    MAE = "MAE"
    RMSE = "RMSE"
    S_MAE = "S-MAE"
    S_RMSE = "S-RMSE"
    ACC = "ACC"
    F1 = "F1"

    def __str__(self) -> str:
        return self.value


class LabelType(Enum):
    SOFT = "soft"
    HARD = "hard"

    def __str__(self) -> str:
        return self.value
