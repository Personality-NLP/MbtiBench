import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np

from mbtibench.enums import LabelType, MbtiDimension, MetricName, ModelName, PromptMethodName
from mbtibench.evaluator import Evaluator


@dataclass
class Arguments:
    model: ModelName
    method: PromptMethodName
    type: LabelType


def main(args: Arguments):
    results: Dict[MbtiDimension, List[Tuple[float, float]]] = {dim: [] for dim in MbtiDimension}

    for round in range(1, 5 + 1):
        if args.model.is_gpt4 and round > 1:
            break  # GPT-4 only has 1 round
        for dim in MbtiDimension:
            database_path = Path("results") / f"round-{round}" / f"{args.type}--{args.model}--{args.method}.db"
            evalutor = Evaluator(database_path, dim)
            s_rmse, s_mae = evalutor.eval([MetricName.S_RMSE, MetricName.S_MAE])
            results[dim].append((s_rmse, s_mae))

    # 计算每个维度的平均值和标准差
    avg_results = {
        dim: (
            np.mean([r[0] for r in res]),
            np.std([r[0] for r in res]),
            np.mean([r[1] for r in res]),
            np.std([r[1] for r in res]),
        )
        for dim, res in results.items()
    }

    # 打印结果
    for dim, (avg_rmse, std_rmse, avg_mae, std_mae) in avg_results.items():
        # print(f"{avg_rmse:.3f},{avg_mae:.3f}", end=",")
        print(f"{avg_rmse:.3f}±{std_rmse:.3f},{avg_mae:.3f}±{std_mae:.3f}", end=",")
    print()  # 换行


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MbtiBench Evaluate")
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--type", type=LabelType, help="Soft or hard label", required=True)
    args = cast(Arguments, parser.parse_args())

    main(args)
