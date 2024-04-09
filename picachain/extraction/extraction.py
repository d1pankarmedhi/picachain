from abc import ABC, abstractmethod


class Extraction(ABC):
    """An Interface for Extraction"""

    def extract_question_answer(self, *args, **kwargs):
        """Extract questions and answers from image"""

    def extract_information(self, *args, **kwargs):
        """Extract information from image"""
