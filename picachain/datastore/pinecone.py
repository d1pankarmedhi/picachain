import time
import uuid
from typing import Optional

from pinecone import Pinecone, PodSpec, ServerlessSpec


class PineconeStore:
    def __init__(self, api_key: str, environment: str) -> None:
        self.api_key = api_key
        self.environment = environment
        self.pinecone = Pinecone(api_key=api_key)

    def _check_index(self, index_name) -> bool:
        exist = False
        existing_indexes = self.list_indexes()
        if index_name in existing_indexes:
            exist = True
        return exist

    def _create_index(
        self,
        index_name: str,
        dimension: int,
        spec: ServerlessSpec | PodSpec,
        metric: str = "cosine",
    ):
        try:
            index = self.pinecone.create_index(
                index_name,
                dimension=dimension,
                metric=metric,
                spec=spec,
            )
            return index
        except Exception as e:
            raise e

    def index(
        self,
        index_name: str,
        dimension: int,
        use_serverless: bool = False,
        cloud: str = "aws",
        region: str = "us-west-2",
    ):
        try:
            if use_serverless:
                spec = ServerlessSpec(cloud="aws", region="us-west-2")
            else:
                spec = PodSpec(environment=self.environment)

            existing_indexes = self.list_indexes()

            if index_name not in existing_indexes:
                self.pinecone.create_index(
                    index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=spec,
                )
                while not self.pinecone.describe_index(index_name).status["ready"]:
                    time.sleep(1)

            index = self.pinecone.Index(index_name)
            return index

        except Exception as e:
            raise e

    def _upsert_data(
        self,
        index_name: str,
        data: list[dict],
        namespace: Optional[str] = "ns1",
    ):
        index = self.index(
            index_name=index_name,
            dimension=len(
                data[0]["embedding"],
            ),
        )
        for doc in data:
            index.upsert(
                vectors=[
                    {
                        "id": str(uuid.uuid4()),
                        "values": doc["embedding"],
                        "metadata": {"image_key": doc["image_key"]},
                    }
                ],
                namespace=namespace,
            )

    def list_indexes(self):
        existing_indexes = [
            index_info["name"] for index_info in self.pinecone.list_indexes()
        ]
        return existing_indexes

    def build(self, index_name: str, dimension: int):
        index = self.index(index_name=index_name, dimension=dimension)
        return index
