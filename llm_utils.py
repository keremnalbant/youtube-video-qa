import logging
from typing import Callable, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.runnables import Runnable

from config import config
from models import CitedAnswer

logger = logging.getLogger(__name__)

class LLMChainError(Exception):
    """Custom exception for LLM chain-related errors"""
    pass

class LLMUtils:
    @staticmethod
    def create_chain() -> Tuple[ChatPromptTemplate, Callable, Runnable]:
        """
        Create RAG chain with source citation.
        
        Args:
            retriever: Document retriever instance
            
        Returns:
            Tuple containing prompt template, document formatter, and structured LLM
            
        Raises:
            LLMChainError: If chain creation fails
        """
        try:
            llm = ChatBedrock(
                model=config.LLM_MODEL_ID,
                model_kwargs=dict(temperature=0, top_k=100, top_p=0.95),
                provider="anthropic",
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions about YouTube videos.
                For each answer, provide relevant timestamps and quotes from the videos.
                Make sure to extract exact timestamps where the information appears."""),
                ("human", """
                 Context: {docs}

                 Question: {question}""")
            ])

            def format_docs(docs):
                return "\n\n".join(
                    f"Content: {doc.page_content}\nTimestamp: {doc.metadata['start']} seconds"
                    for doc in docs
                )

            structured_llm = llm.with_structured_output(CitedAnswer)

            return prompt, format_docs, structured_llm
        except Exception as e:
            logger.error(f"Error creating LLM chain: {str(e)}")
            raise LLMChainError(f"Could not create LLM chain: {str(e)}") 