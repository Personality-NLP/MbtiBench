import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from typing_extensions import assert_never

from .enums import LabelType, MbtiDimension, MetricName

logger = logging.getLogger(__name__)


class Metric:
    @classmethod
    def compute(cls, name: MetricName, y_true: np.ndarray, y_pred: np.ndarray):
        if name == MetricName.MAE:
            return cls._mae_score(y_true, y_pred)
        elif name == MetricName.RMSE:
            return cls._rmse_score(y_true, y_pred)
        elif name == MetricName.S_MAE:
            return cls._bucket_mae_score(y_true, y_pred)
        elif name == MetricName.S_RMSE:
            return cls._bucket_rmse_score(y_true, y_pred)
        elif name == MetricName.ACC:
            return cls._acc_score(y_true, y_pred)
        elif name == MetricName.F1:
            return cls._f1_score(y_true, y_pred)
        else:
            assert_never()

    @classmethod
    def _mae_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    @classmethod
    def _rmse_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return root_mean_squared_error(y_true, y_pred)

    @classmethod
    def _bucket_mae_score(cls, y_true: np.ndarray, y_pred: np.ndarray, bins: int = 9) -> float:
        assert 0 <= y_true.all() <= 1
        assert 0 <= y_pred.all() <= 1

        # 将 [0, 1] 范围分成 bins 个等宽的桶，每个桶的边界
        bins = np.linspace(0, 1, bins + 1)

        y_true_binned = np.digitize(y_true, bins) - 1
        y_pred_binned = np.digitize(y_pred, bins) - 1

        return cls._mae_score(y_true_binned, y_pred_binned)

    @classmethod
    def _bucket_rmse_score(cls, y_true: np.ndarray, y_pred: np.ndarray, bins: int = 9) -> float:
        assert 0 <= y_true.all() <= 1
        assert 0 <= y_pred.all() <= 1

        # 将 [0, 1] 范围分成 bins 个等宽的桶，每个桶的边界
        bins = np.linspace(0, 1, bins + 1)

        y_true_binned = np.digitize(y_true, bins) - 1
        y_pred_binned = np.digitize(y_pred, bins) - 1

        return cls._rmse_score(y_true_binned, y_pred_binned)

    @classmethod
    def _acc_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred) * 100

    @classmethod
    def _f1_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average="macro") * 100


class Exacter:
    @classmethod
    def get_softlabel(cls, text: str) -> Optional[float]:
        model_score = re.findall(r"\[\[\b(10(\.0{1,3})?|[0-9](\.[0-9]{1,3})?)\b\]\]", text)  # [[4.25]] / [[8]]
        if len(model_score) == 0 or len(model_score[0]) == 0:
            model_score = re.findall(r"\[\b(10(\.0{1,3})?|[0-9](\.[0-9]{1,3})?)\b\]", text)  # [4.25] / [8]
        if len(model_score) == 0 or len(model_score[0]) == 0:
            logger.info(f"Bad response from model: {text}")
            return None
        else:
            # model score is 1~9, convert to 0~1
            return (float(model_score[0][0]) - 1) / 8

    @classmethod
    def get_hardlabel(cls, dim: MbtiDimension, text: str) -> Optional[str]:
        first_letter_valid_choices_lower = ["choice: a", "choice: <a>", f"choice: {dim.first_letter.lower()}"]
        second_letter_valid_choices_lower = ["choice: b", "choice: <b>", f"choice: {dim.second_letter.lower()}"]
        if text.lower() in first_letter_valid_choices_lower:
            return dim.first_letter
        elif text.lower() in second_letter_valid_choices_lower:
            return dim.second_letter
        else:
            logger.info(f"Bad response from model: {text}")
            return None

    @classmethod
    def get_hardlabel_as_softlabel(cls, dim: MbtiDimension, text: str) -> Optional[float]:
        hardlabel = cls.get_hardlabel(dim, text)
        if hardlabel is None:
            return None
        else:
            return 1.0 if hardlabel == dim.first_letter else 0.0


class Evaluator:
    def __init__(self, database_path: Path, dim: MbtiDimension):
        self._database_path = database_path
        self._dim = dim

        self._validate()

    @property
    def _validate_database_sql(self) -> str:
        return f"SELECT id, response FROM {self._dim.only_letter}"

    def _validate(self):
        assert self._database_path.exists(), f"Database not found: {self._database_path}"

        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(self._validate_database_sql)
        db_data = c.fetchall()
        conn.close()
        assert (
            len(db_data) == 286
        ), f"Database {self._database_path} len = {len(db_data)} != 286"  # MbtiBench dataset size

        for row in db_data:
            assert "OPENAI API ERROR" not in row[1]

    def _get_human_softlables(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT softlabel FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.array([softlabel[0] for softlabel in db_data])

    def _get_human_hardlables(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT hardlabel FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.array([softlabel[0] for softlabel in db_data])

    def _get_softlabel_from_text(self, id: int, text: str) -> float:
        score = Exacter.get_softlabel(text)
        if score is not None:
            return score
        else:
            logger.info(f"Data id={id}, bad response from model: {text}, using default 0.5 score")
            return 0.5

    def _get_hardlabel_from_text(self, id: int, text: str) -> str:
        label = Exacter.get_hardlabel(self._dim, text)
        if label is not None:
            return label
        else:
            logger.info(f"Data id={id}, bad response from model: {text}, using default 1 score")
            return 1

    def _get_model_softlabels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT id, response FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.array([self._get_softlabel_from_text(id, response) for id, response in db_data])

    def _get_model_hardlabels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT id, response FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.array([self._get_hardlabel_from_text(id, response) for id, response in db_data])

    def _get_baseline_softlabels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT softlabel FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.repeat(np.mean(db_data), len(db_data))

    def eval(self, type: LabelType, metrics: List[MetricName]) -> List[float]:
        if type == LabelType.SOFT:
            # y_true, y_pred = np.array(self._get_human_softlables()), np.array(self._get_baseline_softlabels())
            y_true, y_pred = np.array(self._get_human_softlables()), np.array(self._get_model_softlabels())
        elif type == LabelType.HARD:
            y_true, y_pred = np.array(self._get_human_hardlables()), np.array(self._get_model_hardlabels())
        else:
            assert_never()
        return [Metric.compute(metric, y_true, y_pred) for metric in metrics]
