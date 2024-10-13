import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Coroutine, Dict, Iterable, List, Sequence

from tqdm.auto import tqdm

from .enums import LabelType, MbtiDimension
from .llm import LLM
from .prompt import PromptMethod

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
    def __init__(self, dataset_path: Path, database_path: Path, dim: MbtiDimension, type: LabelType):
        self._database_path = database_path
        self._dim = dim
        self._type = type

        self._init_database()
        self._load_data_to_resume(dataset_path)

    @property
    def _init_database_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._dim.only_letter} "
            f"(id INTEGER PRIMARY KEY, messages TEXT, response TEXT, softlabel REAL, hardlabel TEXT, labeltype TEXT)"
        )

    @property
    def _load_database_sql(self) -> str:
        return f"SELECT id, messages FROM {self._dim.only_letter}"

    @property
    def _update_database_sql(self) -> str:
        return (
            f"INSERT OR REPLACE INTO {self._dim.only_letter} "
            f"(id, messages, response, softlabel, hardlabel, labeltype) "
            f"VALUES "
            f"(:id, :messages, :response, :softlabel, :hardlabel, :labeltype)"
        )

    def _init_database(self):
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(self._init_database_sql)
        conn.commit()
        conn.close()

    def _load_all_data(self, dataset_path: Path) -> List[Dict]:
        with open(dataset_path) as f:
            lines = f.readlines()
        return [json.loads(line.strip()) for line in lines]

    def _load_data_to_resume(self, dataset_path: Path):
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()
        c.execute(self._load_database_sql)
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

    async def _single_run(self, llm: LLM, data: Dict, method_cls: Any) -> Dict:
        user_posts, user_posts_str, user_posts_count = data["posts"], "", 1
        for i in range(len(user_posts)):
            if len(user_posts[i]) > 10:
                post = llm.tokenizer.decode(llm.tokenizer.encode(user_posts[i].replace("{", "").replace("}", ""))[:80])
                user_posts_str += f"Post {user_posts_count}: {post}; "
                user_posts_count += 1

        prompt_method: PromptMethod = method_cls(data["source"], self._dim, user_posts_str)
        prompts = prompt_method.prompts

        messages = llm.chat(prompts)

        return {
            "id": data["id"],
            "messages": llm.show_real_prompt(messages),
            "response": messages[-1]["content"],
            "softlabel": data["softlabels"][self._dim.value],
            "hardlabel": data["hardlabels"][self._dim.value],
            "labeltype": self._type.value,
        }

    async def run(self, llm: LLM, method_cls: Any):
        conn = sqlite3.connect(self._database_path)
        c = conn.cursor()

        # batch_size is for database write-back
        # concurrency is for async calls to OpenAI API
        for batched_data in tqdm(batch(self.data_to_resume, batch_size=20), desc=f"{self._dim}"):
            tasks = [self._single_run(llm, data, method_cls) for data in batched_data]
            logger.info(f"Running batched {len(tasks)} tasks")
            results = await asyncio.gather(*_limit_concurrency(tasks, concurrency=10))

            for result in results:
                c.execute(self._update_database_sql, result)
            conn.commit()

        conn.close()
