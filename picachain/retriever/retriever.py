from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    @abstractmethod
    def get_relevant_images(
        self, collection, query_embedding: List[float], top_k: int
    ) -> list:
        """Retrieve relevant documents from datastore"""
