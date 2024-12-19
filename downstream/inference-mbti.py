import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from Dreaddit.executer import DreadditMbtiExecuter
from Dreaddit.prompt import get_prompt_method_cls as dreaddit_get_prompt_method_cls

from mbtibench.enums import LabelType, MbtiDimension, ModelName, PromptMethodName
from mbtibench.llm import LLM
from mbtibench.utils import get_base_url_and_api_key


@dataclass
class Arguments:
    method: PromptMethodName
    model: ModelName
    type: Optional[LabelType]
    host: Optional[str]
    port: Optional[str]
    round: int


async def mbti(args: Arguments):
    base_url, api_key = get_base_url_and_api_key(args.host, args.port)
    llm = LLM(args.model, base_url, api_key)
    method_cls = dreaddit_get_prompt_method_cls(args.method, args.type)
    dataset_path = Path("downstream") / "Dreaddit" / "data" / "test.jsonl"
    database_path = (
        Path("downstream")
        / "results"
        / "Dreaddit"
        / f"round-{args.round}"
        / f"{args.type}--{args.model}--{args.method}.db"
    )
    executer_cls = DreadditMbtiExecuter()
    tasks = []
    for dim in MbtiDimension:
        executer = executer_cls(dataset_path, database_path, dim, args.type)
        tasks.append(executer.run(llm, method_cls))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreaddit Inference")
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    parser.add_argument("--type", type=LabelType, help="Soft or hard label", required=False)
    parser.add_argument("--round", type=int, help="Round number", required=False, default=1)
    parser.add_argument("--host", type=str, help="vLLM server host address", required=False)
    parser.add_argument("--port", type=str, help="vLLM server port number", required=False)
    args = cast(Arguments, parser.parse_args())

    asyncio.run(mbti(args))
