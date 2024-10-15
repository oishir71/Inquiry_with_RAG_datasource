import os
import sys
import csv
from typing import List, Union

from pydantic import BaseModel, field_validator, ValidationError

# Logging
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter(
    "%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s"
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../openai-gpt-wrapper"
)
from src.chat_completion import ChatCompletion
from utils.read_environments import read_environments

from rag_text_embedding import RAGTextEmbedding


class Message(BaseModel):
    response: str
    references: List[Union[int, str]]

    @field_validator("references")
    def check_references(cls, value):
        references = []
        for reference in value:
            reference = int(reference)
            references.append(reference)
        return references


class RAGInquiry:
    def __init__(self, version="gpt4") -> None:
        read_environments(version=version)
        self.llm = ChatCompletion()

    def read_rag_data(
        self, file_name: str = f"{os.path.dirname(__file__)}/../deliverable/data.csv"
    ):
        rag_data = {}
        with open(file_name, mode="r") as f:
            reader = csv.reader(f)
            for i_row, row in enumerate(reader):
                if i_row > 0:
                    text = row[0]
                    data_source = row[1]
                    embedding = eval(row[2])
                    rag_data[text] = {
                        "data_source": data_source,
                        "embedding": embedding,
                    }

        return rag_data

    def _get_cosine_similarity(self, vec1: list[float], vec2: list[float]):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def get_rag_data_for_inquiry(
        self, embedding, rag_data, number_of_rag_data_for_inquiry: int = 5
    ):
        for text in rag_data:
            similarity = self._get_cosine_similarity(
                vec1=rag_data[text]["embedding"], vec2=embedding
            )
            rag_data[text]["similarity"] = similarity

        sorted_rag_data = sorted(
            rag_data.items(), key=lambda item: item[1]["similarity"], reverse=True
        )
        rag_data_for_inquiry = {
            rag_data[0]: rag_data[1]["data_source"]
            for rag_data in sorted_rag_data[0:number_of_rag_data_for_inquiry]
        }
        return rag_data_for_inquiry

    def text_cosmetics(self, text: str) -> str:
        text = text.replace("\n", "")
        return text

    def get_refered_rag_data(
        self, indices: list[int], rag_data_for_inquiry: list[str]
    ) -> str:
        refered_rag_data = ""
        for i_rag, (text, data_source) in enumerate(rag_data_for_inquiry.items()):
            if i_rag + 1 in indices:
                refered_rag_data += f"- {text} ({data_source})\n"

        return refered_rag_data

    def run_llm_chat_base(self, text: str, rag_data_for_inquiry: list[str]) -> None:
        inquiry_text = (
            f"{self.text_cosmetics(text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にして回答してください。\n"
        )
        for rag in rag_data_for_inquiry:
            inquiry_text += f"- {rag}\n"

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )
        result = self.llm.execute()
        model_output = self.llm.get_model_output_from_result(result=result)

        return model_output

    def run_llm_chat_oishi(self, text: str, rag_data_for_inquiry: dict):
        inquiry_text = (
            f"{self.text_cosmetics(text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にして回答してください。\n"
        )
        for rag in rag_data_for_inquiry:
            inquiry_text += f"- {rag}\n"
        inquiry_text += "\nまた以下のフォーマットで回答してください。\n"
        inquiry_text += "### 質問の回答\n(質問への答えがここに入る)\n\n"

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )
        result = self.llm.execute()
        model_output = self.llm.get_model_output_from_result(result=result)

        output_text = (
            f"{model_output.replace('。', '。\n')}\n"
            "### 回答の根拠 (必ずしも正しいとは限らない)\n"
        )
        for rag in rag_data_for_inquiry:
            output_text += (
                f"- {self.text_cosmetics(rag)} ({rag_data_for_inquiry[rag]})\n"
            )

        return output_text

    def run_llm_chat_oishi_mimic_sugimoto(self, text: str, rag_data_for_inquiry: dict):
        inquiry_text = (
            f"{self.text_cosmetics(text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にして回答してください。\n"
        )
        for i_rag, rag in enumerate(rag_data_for_inquiry):
            inquiry_text += f"{i_rag + 1}: {rag}\n"
        inquiry_text += "\nまた以下のフォーマットで回答してください。\n"
        inquiry_text += "### 質問の回答\n(質問への答えがここに入る)\n\n"

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )
        result = self.llm.execute()
        model_output_response = self.llm.get_model_output_from_result(result=result)
        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="system",
            text=model_output_response,
        )
        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=(
                "上記の回答を得るために参考にした情報に対応する番号を全て以下のフォーマットで回答してください。\n"
                "[1,2,3]"
            ),
        )
        result = self.llm.execute()
        model_output_references = self.llm.get_model_output_from_result(result=result)

        output_text = model_output_response + self.get_refered_rag_data(
            indices=model_output_references, rag_data_for_inquiry=rag_data_for_inquiry
        )

        return output_text

    def run_llm_chat_oishi_with_pydantic(
        self, text: str, rag_data_for_inquiry: dict, number_of_retries: int = 5
    ):
        inquiry_text = (
            f"{self.text_cosmetics(text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にして回答してください。\n"
        )
        for i_rag, rag in enumerate(rag_data_for_inquiry):
            inquiry_text += f"{i_rag + 1}: {rag}\n"
        inquiry_text += (
            "また以下のフォーマットに従って回答してください。\n"
            '{"response": "(質問への回答がここに入る)", "references": (参考にした情報に対応する番号のリストがここにないる)}'
        )

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="system",
            text="あなたは質問とそれに関連した情報を与えられ、それらを元に回答します。",
        )
        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )

        response = ""
        references = []
        for i_trial in range(number_of_retries):
            logger.info(f"Trial No.{i_trial+1}")
            result = self.llm.execute()
            model_output = self.llm.get_model_output_from_result(result=result)
            json_parse_content, json_parse_result = self.llm.model_output_parser(
                model_output=model_output
            )
            if not json_parse_result:
                continue
            format_parse_result = Message(**json_parse_content)
            if not format_parse_result:
                continue

            response = format_parse_result.response
            references = format_parse_result.references
            break
        else:
            logger.error(f"Failed to get an appropriate response for {text}")

        output_text = response + self.get_refered_rag_data(
            ndices=references, rag_data_for_inquiry=rag_data_for_inquiry
        )
        return output_text

    def run_llm_chat_sugimoto_with_data_source(
        self, text: str, rag_data_for_inquiry: dict
    ) -> None:
        inquiry_text = (
            f"{self.text_cosmetics(text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にしてください。また参考にした情報のデータソースも一緒に回答してください。\n"
        )
        for text in rag_data_for_inquiry:
            inquiry_text += "-\n"
            inquiry_text += f"  - 内容: {text}\n"
            inquiry_text += f'  - データソース: {rag_data_for_inquiry.get(text, "https://sample.com")}\n\n'

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )
        result = self.llm.execute()
        model_output = self.llm.get_model_output_from_result(result=result)

        return model_output

    def run_llm_chat_sugimoto_with_data_source_formatted(
        self, text: str, rag_data_for_inquiry: dict
    ) -> None:
        inquiry_text = (
            f"{self.text_cosmetics(text=text)}\n"
            "以下の情報のうち回答に関連するものがあれば参考にしてください。\n"
        )
        for text in rag_data_for_inquiry:
            inquiry_text += "-\n"
            inquiry_text += f"  - 内容: {self.text_cosmetics(text=text)}\n"
            inquiry_text += f"  - データソース: {rag_data_for_inquiry.get(text, 'http://sample.com')}\n\n"
        inquiry_text += "また以下のフォーマットで回答してください。\n\n"
        inquiry_text += (
            "### 質問の回答\n"
            "(質問への答えがここに入る)\n\n"
            "### 根拠\n"
            "(参考にした情報があればデータソースと共にここに箇条書きにする)\n"
        )

        self.llm.add_message_entry_as_specified_role_with_text_content(
            role="user",
            text=inquiry_text,
        )
        result = self.llm.execute()
        model_output = self.llm.get_model_output_from_result(result=result)

        return model_output


if __name__ == "__main__":
    embedder = RAGTextEmbedding()
    inquirer = RAGInquiry()

    text = "SB Intuitions株式会社について教えて"
    embedding = embedder.embedding(text=text)
    rag_data = inquirer.read_rag_data()
    rag_data_for_inquiry = inquirer.get_rag_data_for_inquiry(
        embedding=embedding, rag_data=rag_data
    )
    inquirer.run_llm_chat_oishi_with_pydantic(
        text=text, rag_data_for_inquiry=rag_data_for_inquiry
    )
