import itertools
import time
import uuid
from typing import List, Optional

from numpy import ndarray
from PIL.Image import Image
from pinecone import Index, Pinecone, PodSpec, ServerlessSpec

from picachain.datastore import DataStore
from picachain.embedding.embedding import Embedding


class PineconeStore(DataStore):
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 512,
        namespace: Optional[str] = "ns1",
        use_serverless: bool = False,
        cloud: str = "aws",
        region: str = "us-west-2",
    ) -> None:
        """Initiate Pinecone database."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.namespace = namespace
        self.use_serverless = use_serverless
        self.cloud = cloud
        self.region = region

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
    ) -> Index:
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

    def create(self):
        """Create or connect to a pinecone index."""
        try:
            if self.use_serverless:
                spec = ServerlessSpec(cloud="aws", region="us-west-2")
            else:
                spec = PodSpec(environment=self.environment)

            existing_indexes = self.list_indexes()

            if self.index_name not in existing_indexes:
                self.pinecone.create_index(
                    self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=spec,
                )
                while not self.pinecone.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)

            index = self.pinecone.Index(self.index_name)
            return index

        except Exception as e:
            raise e

    def _chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def add(
        self,
        index,
        embeddings: List[float],
        documents: List[str],
        ids: List[str],
    ):
        """Add embeddings and images to index or index.

        Args:
        - index: index for pinecone
        - embeddings: list of image embeddings
        - documents: list of images
        - ids: list of ids for images.
        """
        for idx, doc in enumerate(documents):
            index.upsert(
                vectors=[
                    {
                        "id": ids[idx],
                        "values": embeddings[idx],
                        "metadata": {"image": doc},
                    }
                ],
                namespace=self.namespace,
            )

    def list_indexes(self):
        existing_indexes = [
            index_info["name"] for index_info in self.pinecone.list_indexes()
        ]
        return existing_indexes

    def build(self, index_name: str, dimension: int):
        index = self.index(index_name=index_name, dimension=dimension)
        return index

    def query(self, index, query_embedding: List[float], top_k: int = 3) -> list:
        """Retrieve relevant images from Pinecone vector database.

        Args:
        - index: pinecone index.
        - query_embedding: query image embedding.
        - top_k (int): top k images to retrieve.

        Returns:
        - list of images along with their score.
        """
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        return result["matches"]

    def delete(self):
        raise NotImplementedError
