import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Coroutine, Dict, Iterable, List, Sequence, Tuple

from tqdm.auto import tqdm

from .enums import MbtiDimension, PromptMethodName, SubDataset
from .llm import LLM
from .prompt import PromptMethod, get_prompt_method

logger = logging.getLogger(__name__)


def batch(iterable: Iterable, batch_size: int) -> Iterable:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _limit_concurrency(coroutines: Sequence[Coroutine], concurrency: int) -> List[Coroutine]:
    semaphore = asyncio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]


class Executer:
    def __init__(self, dataset_path: Path, database_path: Path, dim: MbtiDimension):
        self._database_path = database_path
        self._dim = dim

        self._init_database()
        self._load_data_to_resume(dataset_path)

    def _init_database(self):
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {self._dim.only_letter} (id INTEGER PRIMARY KEY, messages TEXT, response TEXT, softlabel REAL, hardlabel TEXT)"
        )
        conn.commit()
        conn.close()

    def _load_all_data(self, dataset_path) -> List[Dict]:
        with open(dataset_path) as f:
            lines = f.readlines()
        return [json.loads(line.strip()) for line in lines]

    def _load_data_to_resume(self, dataset_path):
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(f"SELECT id, messages FROM {self._dim.only_letter}")
        db_data = c.fetchall()
        conn.close()

        db_data_dict = {row[0]: row[1] for row in db_data}
        all_data = self._load_all_data(dataset_path)

        self.data_to_resume = [
            data
            for data in all_data
            if data["id"] not in db_data_dict.keys() or "OPENAI API ERROR" in db_data_dict[data["id"]]
        ]

        logger.info(f"Left {len(self.data_to_resume)} data to resume (Total {len(all_data)})")

    async def _single_run(
        self,
        llm: LLM,
        dataset: SubDataset,
        method: PromptMethodName,
        user_posts: str,
        data_id: int,
        data_softlabel: float,
        data_hardlabel: str,
    ) -> Tuple[int, str, str]:
        user_posts_str, user_posts_count = "", 1
        for i in range(len(user_posts)):
            if len(user_posts[i]) > 10:
                post = llm.tokenizer.decode(llm.tokenizer.encode(user_posts[i].replace("{", "").replace("}", ""))[:80])
                user_posts_str += f"Post {user_posts_count}: {post}; "
                user_posts_count += 1

        prompt_method: PromptMethod = get_prompt_method(dataset, self._dim, user_posts_str, method)
        prompts = prompt_method.prompts

        messages = llm.chat(prompts)

        assert messages[-1]["role"] == "assistant"
        response = messages[-1]["content"]

        return data_id, llm.show_real_prompt(messages), response, data_softlabel, data_hardlabel

    async def run(self, llm: LLM, method: PromptMethodName):
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()

        # batch_size is for database write-back
        # concurrency is for async calls to OpenAI API
        for batched_data in tqdm(batch(self.data_to_resume, batch_size=20), desc=f"{self._dim}"):
            tasks = [
                self._single_run(
                    llm,
                    data["source"],
                    method,
                    data["posts"],
                    data["id"],
                    data["softlabels"][self._dim.value],
                    data["hardlabels"][self._dim.value],
                )
                for data in batched_data
            ]
            logger.info(f"Running batched {len(tasks)} tasks")
            results = await asyncio.gather(*_limit_concurrency(tasks, concurrency=10))

            for data_id, messages_str, response, data_softlabel, data_hardlabel in results:
                c.execute(
                    f"INSERT OR REPLACE INTO {self._dim.only_letter} (id, messages, response, softlabel, hardlabel) VALUES (?, ?, ?, ?, ?)",
                    (data_id, messages_str, response, data_softlabel, data_hardlabel),
                )
            conn.commit()

        conn.close()
