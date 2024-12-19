import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from typing_extensions import assert_never

from downstream.Dreaddit.prompt import DreadditDownstream, DreadditZeroShotSoft
from mbtibench.enums import LabelType, MbtiDimension
from mbtibench.evaluator import Exacter
from mbtibench.executer import Executer
from mbtibench.llm import LLM


class DreadditMbtiExecuter(Executer):
    @property
    def _init_database_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._dim.only_letter} "
            f"(id INTEGER PRIMARY KEY, messages TEXT, response TEXT, posts TEXT, label TEXT, labeltype TEXT)"
        )

    @property
    def _update_database_sql(self) -> str:
        return (
            f"INSERT OR REPLACE INTO {self._dim.only_letter} "
            f"(id, messages, response, posts, label, labeltype) "
            f"VALUES "
            f"(:id, :messages, :response, :posts, :label, :labeltype)"
        )

    async def _single_run(self, llm: LLM, data: Dict, method_cls: Any) -> Dict:
        prompt_method: DreadditZeroShotSoft = method_cls(self._dim, data["posts"])
        prompts = prompt_method.prompts

        messages = llm.chat(prompts)

        assert messages[-1]["role"] == "assistant"
        response = messages[-1]["content"]

        return {
            "id": data["id"],
            "messages": llm.show_real_prompt(messages),
            "response": response,
            "posts": data["posts"],
            "label": data["label"],
            "labeltype": self._type.value,
        }


class DreadditDownstreamExecuter(Executer):
    def __init__(
        self,
        dataset_path: Path,
        database_path: Path,
        mbti_result_database_path: Path,
        dim: Optional[MbtiDimension],
        type: Optional[LabelType],
    ):
        super().__init__(dataset_path, database_path, dim, type)
        assert dim is None

        self._type = type
        if type is not None:
            unformatted_mbti_answer = self._load_raw_mbti_answer(mbti_result_database_path)
            self._mbti_answer = self._format_mbti_answer(unformatted_mbti_answer)

    @property
    def _init_database_sql(self) -> str:
        return (
            "CREATE TABLE IF NOT EXISTS dreaddit "
            "(id INTEGER PRIMARY KEY, messages TEXT, response TEXT, posts TEXT, label TEXT, labeltype TEXT)"
        )

    @property
    def _load_database_sql(self) -> str:
        return "SELECT id, messages FROM dreaddit"

    @property
    def _update_database_sql(self) -> str:
        return (
            "INSERT OR REPLACE INTO dreaddit "
            "(id, messages, response, posts, label, labeltype) "
            "VALUES "
            "(:id, :messages, :response, :posts, :label, :labeltype)"
        )

    def _load_raw_mbti_answer(self, mbti_result_database_path: Path) -> Dict[int, Dict[MbtiDimension, str]]:
        conn = sqlite3.connect(mbti_result_database_path)
        c = conn.cursor()
        all_data = {}
        for dim in MbtiDimension:
            c.execute(f"SELECT id, response FROM {dim.only_letter}")
            db_data = c.fetchall()
            for data_id, response in db_data:
                if data_id not in all_data.keys():
                    all_data[data_id] = {}
                all_data[data_id][dim] = response
        conn.close()
        return all_data

    def _format_mbti_answer(self, mbti_answer: Dict[int, Dict[MbtiDimension, str]]) -> Dict[int, str]:
        formatted_mbti_answer = {}
        for data_id, answer in mbti_answer.items():
            formatted_dim = ""
            for dim in MbtiDimension:
                if self._type == LabelType.SOFT:
                    score = Exacter().get_softlabel(answer[dim])
                    score = (score if score is not None else 0.5) * 100
                elif self._type == LabelType.HARD:
                    score = Exacter().get_hardlabel_as_softlabel(dim, answer[dim])
                    score = (score if score is not None else 0.5) * 100
                else:
                    assert_never()
                # formatted_dim += f"{score}% {dim.first_letter}, {100 - score}% {dim.second_letter}\n"
                formatted_dim += f"{score:.1f}% {dim.full_first_letter}, {100 - score:.1f}% {dim.full_second_letter}\n"
            formatted_mbti_answer[data_id] = formatted_dim
        return formatted_mbti_answer

    async def _single_run(self, llm: LLM, data: Dict, method_cls: Any) -> Dict:
        prompt_method: DreadditDownstream = method_cls(
            data["posts"], self._mbti_answer[data["id"]] if self._type is not None else None, self._type
        )
        prompts = prompt_method.prompts

        messages = llm.chat(prompts, max_tokens=8192)

        assert messages[-1]["role"] == "assistant"
        response = messages[-1]["content"]

        return {
            "id": data["id"],
            "messages": llm.show_real_prompt(messages),
            "response": response,
            "posts": data["posts"],
            "label": data["label"],
            "labeltype": self._type.value if self._type is not None else "",
        }
