from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class Embedding(ABC):
    """Interface for Embedding model"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save(self, path: str):
        """Save model"""

    @abstractmethod
    def encode_text(self, text: str):
        """Encode text"""

    @abstractmethod
    def encode_image(self, image: Image.Image):
        """Encode image"""

    @abstractmethod
    def encode_images(self, images: List[Image.Image]):
        """Encode images"""
