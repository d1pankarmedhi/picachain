from PIL import Image

from picachain.embedding import ClipEmbedding, Embedding
from picachain.retriever import ImageRetriever


class ImageSearch:
    def __init__(
        self, retriever: ImageRetriever, query_img: Image, embedding: Embedding
    ) -> None:
        self.retriever = retriever
        self.query_img = query_img
        self.embedding = embedding

    ## search method for various datastores
    def _embed_query_img(self):
        return self.embedding.encode_images([self.query_img]).tolist()

    def search_relevant_images(
        self,
        top_k: int = 3,
    ):
        return self.retriever.get_relevant_images(
            collection=self.retriever.datastore_collection,
            query_embedding=self._embed_query_img(),
            top_k=top_k,
        )
