import os
import sys
import csv

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
from src.embedding import Embedding
from utils.read_environments import read_environments


class RAGTextEmbedding:
    def __init__(self) -> None:
        read_environments("ada2")
        self.llm = Embedding()

    def read_rag_data(
        self, file_name: str = f"{os.path.dirname(__file__)}/../data/data.csv"
    ) -> list[str]:
        texts, data_sources = [], []
        with open(file_name, mode="r") as f:
            reader = csv.reader(f)
            for i_row, row in enumerate(reader):
                if i_row > 0:
                    text = row[0]
                    data_source = row[1]
                    texts.append(text)
                    data_sources.append(data_source)
        return texts, data_sources

    def embedding(self, text: str) -> list[float]:
        self.llm.set_input(text=text)
        result = self.llm.execute()
        embedding = self.llm.get_embedding_vector(result=result)
        self.llm.delete_input()

        return embedding

    def bulk_embedding(self, texts: list[str]):
        embeddings = []
        for text in texts:
            embedding = self.embedding(text=text)
            embeddings.append(embedding)

        return embeddings

    def dump_text_embedding(
        self,
        texts: list[str],
        data_sources: list[str],
        embeddings: list[list[float]],
        file_name: str = f"{os.path.dirname(__file__)}/../deliverable/data.csv",
    ):
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name), exist_ok=False)
        with open(file_name, mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "data_source", "embedding"])
            for text, data_source, embedding in zip(texts, data_sources, embeddings):
                writer.writerow([text, data_source, embedding])


if __name__ == "__main__":
    obj = RAGTextEmbedding()
    texts, data_sources = obj.read_rag_data()
    embeddings = obj.bulk_embedding(texts=texts)
    obj.dump_text_embedding(
        texts=texts, data_sources=data_sources, embeddings=embeddings
    )
