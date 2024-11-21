import logging
from typing import List

import weaviate
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

from config import config

logger = logging.getLogger(__name__)


class WeaviateClientError(Exception):
    """Custom exception for Weaviate client-related errors"""

    pass


class WeaviateIO:
    def __init__(self):
        """Initialize Weaviate client with configuration"""
        try:
            self.headers = {
                "X-AWS-Access-Key": config.AWS_ACCESS_KEY,
                "X-AWS-Secret-Key": config.AWS_SECRET_KEY,
            }

            self.client = weaviate.connect_to_weaviate_cloud(
                config.WEAVIATE_URL,
                auth_credentials=AuthApiKey(config.WEAVIATE_AUTH_KEY),
                headers=self.headers,
            )

            self.embeddings = BedrockEmbeddings(model_id=config.EMBEDDING_MODEL_ID)
        except Exception as e:
            logger.error(f"Error initializing Weaviate client: {str(e)}")
            raise WeaviateClientError(f"Could not initialize Weaviate client: {str(e)}")

    def setup_vectorstore_for_video_and_transcript(
        self, chunks: List[Document], video_id: str
    ) -> WeaviateVectorStore:
        """
        Setup Weaviate vector store for document chunks.

        Args:
            chunks (List[Document]): Document chunks to store
            video_id (str): YouTube video ID for class naming

        Returns:
            WeaviateVectorStore: Configured vector store

        Raises:
            WeaviateClientError: If vector store setup fails
        """
        try:
            index_name = f"YoutubeTranscript_{video_id}"

            vectorstore = WeaviateVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.client,
                index_name=index_name,
                text_key="text",
            )

            return vectorstore
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise WeaviateClientError(f"Could not setup vector store: {str(e)}")

    def close(self):
        """Safely close Weaviate client connection"""
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {str(e)}")
