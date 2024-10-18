import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from mbtibench.enums import LabelType, MbtiDimension, ModelName, PromptMethodName
from mbtibench.executer import Executer
from mbtibench.llm import LLM
from mbtibench.prompt import get_prompt_method_cls
from mbtibench.utils import get_base_url_and_api_key


@dataclass
class Arguments:
    method: PromptMethodName
    model: ModelName
    type: LabelType
    round: int
    host: Optional[str]
    port: Optional[str]


async def main(args: Arguments):
    base_url, api_key = get_base_url_and_api_key(args.host, args.port)
    llm = LLM(args.model, base_url, api_key)
    method_cls = get_prompt_method_cls(args.method, args.type)
    dataset_path = Path("dataset") / "mbtibench.jsonl"
    database_path = Path("results") / f"round-{args.round}" / f"{args.type}--{args.model}--{args.method}.db"
    tasks = []
    for dim in MbtiDimension:
        executer = Executer(dataset_path, database_path, dim, args.type)
        tasks.append(executer.run(llm, method_cls))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MbtiBench Inference")
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    parser.add_argument("--type", type=LabelType, help="Soft or hard label", required=True)
    parser.add_argument("--round", type=int, help="Experiment round", required=True)
    parser.add_argument("--host", type=str, help="vLLM server host address", required=False)
    parser.add_argument("--port", type=str, help="vLLM server port number", required=False)
    args = cast(Arguments, parser.parse_args())

    asyncio.run(main(args))
