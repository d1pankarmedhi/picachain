from abc import ABC, abstractmethod


class Extraction(ABC):
    """An Interface for Extraction"""

    @abstractmethod
    def extract(self, *args, **kwargs):
        """Extract information from image"""
