import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mbtibench.enums import MbtiDimension, MetricName, ModelName, PromptMethodName
from mbtibench.evaluator import Evaluator


@dataclass
class Arguments:
    method: PromptMethodName
    model: ModelName


def main(args: Arguments):
    with open("output.txt", "a") as f:
        for dim in MbtiDimension:
            database_path = Path("results") / f"{args.model}--{args.method}.db"
            evalutor = Evaluator(database_path, dim)
            s_rmse, s_mae = evalutor.eval([MetricName.S_RMSE, MetricName.S_MAE])
            f.write(f"{s_rmse:.6f},{s_mae:.6f},")
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MbtiBench Evaluate")
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    args = cast(Arguments, parser.parse_args())

    main(args)
