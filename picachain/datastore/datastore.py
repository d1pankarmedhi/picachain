from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL.Image import Image

from picachain.embedding import Embedding


class DataStore(ABC):
    pass

    @abstractmethod
    def generate_embeddings(
        self, images: List[Image], embedding: Embedding
    ) -> np.ndarray:
        """Generate embeddings to be stored on the datastore."""

    @abstractmethod
    def create_collection(self):
        """Create a collection for documents and embedding"""

    @abstractmethod
    def add_to_collection(self, *args, **kwargs):
        """Add embeddings and metadata to collection."""

    @abstractmethod
    def query_documents(self, *args, **kwargs) -> list:
        """Query similar images"""
