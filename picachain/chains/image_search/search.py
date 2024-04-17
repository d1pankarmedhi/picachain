from typing import Any, List, Union

from PIL import Image

from picachain.embedding import Embedding
from picachain.retriever import ImageRetriever


class ImageSearchChain:
    def __init__(
        self,
        retriever: ImageRetriever,
        embedding: Embedding,
        image: Union[Image.Image, List[float]],
    ):
        self.retriever = retriever
        self.embedding = embedding
        self.image = image

    @classmethod
    def from_image(
        cls,
        retriever: ImageRetriever,
        embedding: Embedding,
        image: Image,
    ):
        image_embedding = cls._encode_image(embedding, image)

        return cls(
            retriever=retriever,
            embedding=embedding,
            image=image_embedding,
        )

    @classmethod
    def from_image_embedding(
        cls,
        retriever: ImageRetriever,
        embedding: Embedding,
        image_embedding: List[float],
    ):
        return cls(
            retriever=retriever,
            embedding=embedding,
            image=image_embedding,
        )

    def similar_images(self, top_k: int = 3):
        return self.retriever.relevant_images(
            query_embedding=self.image,
            top_k=top_k,
        )

    @staticmethod
    def _encode_image(embedding: Embedding, image: Image.Image) -> List[float]:
        return embedding.encode_images([image]).tolist()
