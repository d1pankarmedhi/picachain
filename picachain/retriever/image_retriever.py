import uuid
from collections import defaultdict
from typing import List

from PIL.Image import Image

from picachain.datastore import ChromaStore, DataStore, PineconeStore
from picachain.embedding import ClipEmbedding, Embedding
from picachain.utils.image_util import base64_to_image, image_to_base64


class ImageRetriever:
    datastore_collection = None

    def __init__(
        self,
        datastore: ChromaStore,
        embedding: Embedding,
        images: List[Image],
    ) -> None:
        """Initiate Image retriever."""
        self.datastore = datastore
        self.embedding = embedding
        self.images = images

        self._push_embedding_to_datastore()

    ## generate embeddings for the images
    def _create_embeddings(self):
        if isinstance(self.embedding, ClipEmbedding):
            return self.embedding.encode_images(self.images).tolist()
        else:
            raise Exception("Error while creating embeddings")

    def _collection_metadata(self):
        raise NotImplementedError

    def _process_images_for_storage(self) -> List[str]:
        """Convert images to base64 string"""
        return [image_to_base64(img) for img in self.images]

    def _create_ids_for_images(self, images: list):
        """Generate ids for each image."""
        return [str(uuid.uuid5(uuid.NAMESPACE_URL, img)) for img in images]

    ## store the embeddings into datastore
    def _push_embedding_to_datastore(self):
        """Push embeddings and documents to datastore."""
        try:
            embeddings = self.datastore.generate_embeddings(self.images, self.embedding)
            self.datastore_collection = self.datastore.create_collection()
            documents = self._process_images_for_storage()
            ids = self._create_ids_for_images(documents)
            self.datastore.add_to_collection(
                collection=self.datastore_collection,
                embeddings=embeddings,
                documents=documents,
                ids=ids,
            )
        except Exception as e:
            raise Exception("Failed to push embeddings", e)

    def get_relevant_images(
        self, collection, query_embedding: list, top_k: int
    ) -> list:
        """Retrieve top k relevant images from datastore.

        Args:
        - collection: database collection.
        - query_embedding: query image embedding.
        - top_k (int): top k relevant images to retrieve.

        Returns:
        - list of relevant images.
        """
        relevant_images = self.datastore.query_documents(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return relevant_images
