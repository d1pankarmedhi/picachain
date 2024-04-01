from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL.Image import Image

from picachain.embedding import Embedding


class DataStore(ABC):
    """Interface for datastores and vector databases."""

    def generate_embeddings(
        self, images: List[Image], embedding: Embedding
    ) -> np.ndarray:
        return embedding.encode_images(images)

    @abstractmethod
    def create(self):
        """Create a collection for documents and embedding"""

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add embeddings and metadata to collection."""

    @abstractmethod
    def query(self, *args, **kwargs) -> list:
        """Query similar images"""

    @abstractmethod
    def delete(self, *args, **kwargs):
        """Delete the collection or index"""
