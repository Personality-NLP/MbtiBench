import argparse
import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
from Dreaddit.executer import DreadditDownstreamExecuter
from Dreaddit.prompt import DreadditDownstream

from mbtibench.enums import LabelType, MbtiDimension, ModelName, PromptMethodName
from mbtibench.evaluator import Exacter
from mbtibench.llm import LLM
from mbtibench.utils import get_base_url_and_api_key


@dataclass
class Arguments:
    method: PromptMethodName
    model: ModelName
    mbti_model: Optional[ModelName]
    type: Optional[LabelType]
    host: Optional[str]
    port: Optional[str]
    round: int


def get_y_pred_to_y_true_map(database_path: Path, dim: MbtiDimension) -> Dict[float, float]:
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute(f"SELECT response, softlabel FROM {dim.only_letter}")
    db_data = c.fetchall()
    conn.close()

    y_pred_to_y_true = {}
    for response, softlabel in db_data:
        score = Exacter.get_softlabel(response)
        score = (0.5 if score is None else score) * 8 + 1
        if score not in y_pred_to_y_true:
            y_pred_to_y_true[score] = []
        y_pred_to_y_true[score].append(softlabel)
    y_pred_to_y_true = {k: sum(v) / len(v) for k, v in y_pred_to_y_true.items()}
    y_pred_to_y_true = {key: y_pred_to_y_true[key] for key in sorted(y_pred_to_y_true)}

    keys = np.array(list(y_pred_to_y_true.keys()))
    values = np.array(list(y_pred_to_y_true.values()))
    new_keys = np.arange(1, 10)
    new_values = np.interp(new_keys, keys, values)
    interpolated_dict = dict(zip(new_keys, new_values))
    interpolated_dict = {float(key): value for key, value in interpolated_dict.items()}

    print(y_pred_to_y_true)
    print(interpolated_dict)
    print("===========")

    return interpolated_dict


def create_normscore(args: Arguments):
    real_softscore_database_path = Path("downstream") / "results" / "round-0" / f"soft--{args.model}--{args.method}.db"
    normsoft_database_path = Path("downstream") / "results" / "round-0" / f"normsoft--{args.model}--{args.method}.db"
    normhard_database_path = Path("downstream") / "results" / "round-0" / f"normhard--{args.model}--{args.method}.db"

    mbtibench_database_path = Path("results") / "round-1" / f"soft--{args.model}--{args.method}.db"

    for dim in MbtiDimension:
        y_pred_to_y_true = get_y_pred_to_y_true_map(mbtibench_database_path, dim)

        # Real softscore
        conn = sqlite3.connect(real_softscore_database_path)
        c = conn.cursor()
        fetch_database_sql = f"SELECT * FROM {dim.only_letter}"
        c.execute(fetch_database_sql)
        db_data = c.fetchall()
        conn.commit()
        conn.close()

        if len(db_data) != 500:
            # Norm soft
            conn = sqlite3.connect(normsoft_database_path)
            c = conn.cursor()
            create_database_sql = (
                f"CREATE TABLE IF NOT EXISTS {dim.only_letter} "
                f"(id INTEGER PRIMARY KEY, messages TEXT, response TEXT, idea TEXT, opinion TEXT, label TEXT, labeltype TEXT)"
            )
            c.execute(create_database_sql)
            for id, messages, response, idea, opinion, label, labeltype in db_data:
                score = Exacter.get_softlabel(response)
                score = (0.5 if score is None else score) * 8 + 1  # 1~9
                score = y_pred_to_y_true[score]  # 0~1
                normsoft_score = score * 8 + 1  # 1~9
                new_item = {
                    "id": id,
                    "messages": messages,
                    "response": f"[[{normsoft_score:.3f}]]",
                    "idea": idea,
                    "opinion": opinion,
                    "label": label,
                    "labeltype": labeltype,
                }
                insert_database_sql = (
                    f"INSERT OR REPLACE INTO {dim.only_letter} "
                    f"(id, messages, response, idea, opinion, label, labeltype) "
                    f"VALUES "
                    f"(:id, :messages, :response, :idea, :opinion, :label, :labeltype)"
                )
                c.execute(insert_database_sql, new_item)
            conn.commit()
            conn.close()

        if len(db_data) != 500:
            # Norm hard
            conn = sqlite3.connect(normhard_database_path)
            c = conn.cursor()
            create_database_sql = (
                f"CREATE TABLE IF NOT EXISTS {dim.only_letter} "
                f"(id INTEGER PRIMARY KEY, messages TEXT, response TEXT, idea TEXT, opinion TEXT, label TEXT, labeltype TEXT)"
            )
            c.execute(create_database_sql)
            for id, messages, response, idea, opinion, label, labeltype in db_data:
                score = Exacter.get_softlabel(response)
                score = (0.5 if score is None else score) * 8 + 1  # 1~9
                score = y_pred_to_y_true[score]  # 0~1
                score = score * 8 + 1  # 1~9
                normhard_score = "A" if score < 5 else "B"
                new_item = {
                    "id": id,
                    "messages": messages,
                    "response": f"CHOICE: {normhard_score}",
                    "idea": idea,
                    "opinion": opinion,
                    "label": label,
                    "labeltype": labeltype,
                }
                insert_database_sql = (
                    f"INSERT OR REPLACE INTO {dim.only_letter} "
                    f"(id, messages, response, idea, opinion, label, labeltype) "
                    f"VALUES "
                    f"(:id, :messages, :response, :idea, :opinion, :label, :labeltype)"
                )
                c.execute(insert_database_sql, new_item)
            conn.commit()
            conn.close()


async def downstream(args: Arguments):
    base_url, api_key = get_base_url_and_api_key(args.host, args.port)
    llm = LLM(args.model, base_url, api_key)
    method_cls = DreadditDownstream()
    dataset_path = Path("downstream") / "Dreaddit" / "data" / "test.jsonl"
    database_path = (
        Path("downstream")
        / "results"
        / "Dreaddit"
        / f"round-{args.round}"
        / f"downstream--{args.type}--task_model_{args.model}--mbti_model_{args.mbti_model}.db"
    )
    mbti_result_database_path = (
        Path("downstream") / "results" / "Dreaddit" / "round-0" / f"{args.type}--{args.mbti_model}--{args.method}.db"
    )
    executer_cls = DreadditDownstreamExecuter()
    executer = executer_cls(dataset_path, database_path, mbti_result_database_path, None, args.type)

    await executer.run(llm, method_cls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreaddit Inference")
    parser.add_argument("--method", type=PromptMethodName, help="Prompt method name", required=True)
    parser.add_argument("--model", type=ModelName, help="Model name", required=True)
    parser.add_argument("--mbti_model", type=ModelName, help="MBTI Model name", required=False)
    parser.add_argument("--type", type=LabelType, help="Soft or hard label", required=False)
    parser.add_argument("--round", type=int, help="Round number", required=False, default=1)
    parser.add_argument("--host", type=str, help="vLLM server host address", required=False)
    parser.add_argument("--port", type=str, help="vLLM server port number", required=False)
    args = cast(Arguments, parser.parse_args())

    asyncio.run(downstream(args))
