from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import ttest_ind

from downstream.Dreaddit.evaluator import DreadditEvaluator
from mbtibench.enums import MetricName


def ttest(results: Dict[str, float]):
    none_results = [res["none"] for res in results]
    soft_results = [res["soft"] for res in results]
    hard_results = [res["hard"] for res in results]

    t_stat_none_soft, p_value_none_soft = ttest_ind(none_results, soft_results)
    t_stat_none_hard, p_value_none_hard = ttest_ind(none_results, hard_results)
    t_stat_soft_hard, p_value_soft_hard = ttest_ind(soft_results, hard_results)

    print("T-test between 'none' and 'soft':")
    print(f"t-statistic: {t_stat_none_soft}, p-value: {p_value_none_soft}")

    print("\nT-test between 'none' and 'hard':")
    print(f"t-statistic: {t_stat_none_hard}, p-value: {p_value_none_hard}")

    print("\nT-test between 'soft' and 'hard':")
    print(f"t-statistic: {t_stat_soft_hard}, p-value: {p_value_soft_hard}")


if __name__ == "__main__":
    configs = [
        {"type": None, "task_model": "llama3.1-70b", "mbti_model": None},
        {"type": "hard", "task_model": "llama3.1-70b", "mbti_model": "gpt-4o"},
        {"type": "soft", "task_model": "llama3.1-70b", "mbti_model": "gpt-4o"},
    ]

    for task in ["dreaddit"]:
        print(f"######### {task} #########")
        results_dict = {"None": [], "soft": [], "hard": []}
        for method in ["zero-shot"]:
            print(f"===== {method} =====")
            for round in range(1, 10 + 1):
                print(f"--- round {round} ---")
                for config in configs:
                    type, task_model, mbti_model = (
                        str(config["type"]),
                        str(config["task_model"]),
                        str(config["mbti_model"]),
                    )
                    try:
                        database_path = (
                            Path("downstream")
                            / "results-reproduce"
                            / task.capitalize()
                            / f"round-{round}"
                            / f"downstream--{type}--task_model_{task_model}--mbti_model_{mbti_model}.db"
                        )
                        evalutor = DreadditEvaluator(database_path, None)
                        acc, f1 = evalutor.eval([MetricName.ACC, MetricName.F1])
                        results_dict[type].append(acc)
                        # results_dict[type].append(f1)
                        print(f"{type:<8s} [MBTI]{mbti_model:<15s} [TASK]{task_model:<15s} {acc:.2f},{f1:.2f}")
                    except Exception as e:
                        print(e)

    t_stat_none_soft, p_value_none_soft = ttest_ind(results_dict["None"], results_dict["soft"], equal_var=False)
    print(f"T-test between 'None' and 'Soft': p-value = {p_value_none_soft:.4f}")
    t_stat_none_hard, p_value_none_hard = ttest_ind(results_dict["None"], results_dict["hard"], equal_var=False)
    print(f"T-test between 'None' and 'Hard': p-value = {p_value_none_hard:.4f}")
    t_stat_soft_hard, p_value_soft_hard = ttest_ind(results_dict["soft"], results_dict["hard"], equal_var=False)
    print(f"T-test between 'Soft' and 'Hard': p-value = {p_value_soft_hard:.4f}")

    print("None:", np.mean(results_dict["None"]))
    print("Hard:", np.mean(results_dict["hard"]))
    print("Soft:", np.mean(results_dict["soft"]))
