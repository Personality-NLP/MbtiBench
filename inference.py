import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv

from mbtibench.enums import MbtiDimension, ModelName, PromptMethodName
from mbtibench.executer import Executer
from mbtibench.llm import LLM


@dataclass
class Arguments:
    method: PromptMethodName
    model: ModelName
    host: Optional[str]
    port: Optional[str]


async def main(args: Arguments):
    if args.host is None and args.port is None:
        load_dotenv()
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
    elif args.host is not None and args.port is not None:
        base_url = f"http://{args.host}:{args.port}/v1"
        api_key = "EMPTY"
    else:
        raise ValueError("Either both host and port should be provided or neither")

    llm = LLM(args.model, base_url, api_key)
    dataset_path = Path("dataset") / "mbtibench.jsonl"
    tasks = []
    for dim in MbtiDimension:
        database_path = Path("results") / f"{args.model}--{args.method}.db"
        executer = Executer(dataset_path, database_path, dim)
        tasks.append(executer.run(llm, args.method))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MbtiBench Inference")
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    parser.add_argument("--host", type=str, help="vLLM server host address", required=False)
    parser.add_argument("--port", type=str, help="vLLM server port number", required=False)
    args = cast(Arguments, parser.parse_args())

    asyncio.run(main(args))
