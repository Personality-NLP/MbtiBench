import logging
import re
import sqlite3
from typing import List

import numpy as np

from mbtibench.enums import MetricName
from mbtibench.evaluator import Evaluator, Metric

logger = logging.getLogger(__name__)


class DreadditEvaluator(Evaluator):
    @property
    def _validate_database_sql(self) -> str:
        return "SELECT id, response FROM dreaddit"

    def _validate(self):
        assert self._database_path.exists(), f"Database not found: {self._database_path}"

        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(self._validate_database_sql)
        db_data = c.fetchall()
        conn.close()
        assert len(db_data) == 300  # Dreaddit dataset size

        for row in db_data:
            assert "OPENAI API ERROR" not in row[1]

    def _get_true_labels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute("SELECT label FROM dreaddit")
        db_data = c.fetchall()
        conn.close()

        return np.array([0 if label[0] == "yes" else 1 for label in db_data])

    def _get_pred_labels(self) -> np.ndarray:
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute("SELECT response FROM dreaddit")
        db_data = c.fetchall()
        conn.close()

        answers = []
        for response in db_data:
            response_clean = re.split(" |\n", response[0].strip())
            response_clean = [
                word.replace(".", "")
                .replace(",", "")
                .replace('"', "")
                .replace("*", "")
                .replace("!", "")
                .replace("[", "")
                .replace("]", "")
                .strip()
                .lower()
                for word in response_clean
            ]

            if "yes" in response_clean and "no" not in response_clean:
                answers.append(0)
            elif "no" in response_clean and "yes" not in response_clean:
                answers.append(1)
            elif "yes" in response_clean and "no" in response_clean:
                # Use the first label
                if response_clean.index("yes") < response_clean.index("no"):
                    answers.append(0)
                else:
                    answers.append(1)
            else:
                answers.append(0)  # TODO
                logger.warning(f"Invalid response: {response}")

        return np.array(answers)

    def eval(self, metrics: List[MetricName]) -> List[float]:
        y_true, y_pred = np.array(self._get_true_labels()), np.array(self._get_pred_labels())
        return [Metric.compute(metric, y_true, y_pred) for metric in metrics]
