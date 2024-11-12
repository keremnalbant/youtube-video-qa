from typing import List

from langchain_core.documents import Document


def format_retrieved_docs(docs: List[Document]):
    return "\n\n".join(
        f"Content: {doc.page_content}\nTimestamp: {doc.metadata['start']} seconds"
        for doc in docs
    )

def _unique_docs(docs: List[Document]):
    unique_docs = []

    for doc in docs:
        if doc.page_content not in [doc.page_content for doc in unique_docs]:
            unique_docs.append(doc)
    return unique_docs