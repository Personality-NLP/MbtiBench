import logging
import re
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing_extensions import assert_never

from .enums import MbtiDimension, MetricName

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


class Evaluator:
    def __init__(self, database_path: Path, dim: MbtiDimension):
        self._database_path = database_path
        self._dim = dim

        self._validate()

    def _validate(self):
        assert self._database_path.exists()

        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT id, response FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()
        assert len(db_data) == 286  # MbtiBench dataset size

        for row in db_data:
            assert "OPENAI API ERROR" not in row[1]

    def _get_human_softlables(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT softlabel FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.array([softlabel[0] for softlabel in db_data])

    def _get_score_from_text(self, id: int, text: str) -> float:
        model_score = re.findall(r"\[\[\b(10(\.0{1,3})?|[0-9](\.[0-9]{1,3})?)\b\]\]", text)  # [[4.25]] / [[8]]
        if len(model_score) == 0 or len(model_score[0]) == 0:
            model_score = re.findall(r"\[\b(10(\.0{1,3})?|[0-9](\.[0-9]{1,3})?)\b\]", text)  # [4.25] / [8]
        if len(model_score) == 0 or len(model_score[0]) == 0:
            logger.info(f"Data id={id}, bad response from model: {text}, using default 5 score")
            return 5
        else:
            return float(model_score[0][0])

    def _get_model_softlabels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT id, response FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        # model score is 1~9, convert to 0~1
        return np.array([(self._get_score_from_text(id, response) - 1) / 8 for id, response in db_data])

    def _get_baseline_softlabels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT softlabel FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        return np.repeat(np.mean(db_data), len(db_data))

    def eval(self, metrics: List[MetricName]) -> List[float]:
        # y_true, y_pred = np.array(self._get_human_softlables()), np.array(self._get_baseline_softlabels())
        y_true, y_pred = np.array(self._get_human_softlables()), np.array(self._get_model_softlabels())
        # y_true, y_pred = np.array(self._get_human_softlables())[:96], np.array(self._get_model_softlabels())[:96]
        # y_true, y_pred = (
        #     np.array(self._get_human_softlables())[96 : 96 + 94],
        #     np.array(self._get_model_softlabels())[96 : 96 + 94],
        # )
        # y_true, y_pred = (
        #     np.array(self._get_human_softlables())[96 + 94 :],
        #     np.array(self._get_model_softlabels())[96 + 94 :],
        # )
        return [Metric.compute(metric, y_true, y_pred) for metric in metrics]
