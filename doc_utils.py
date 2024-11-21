from typing import List

from langchain_core.documents import Document


def format_retrieved_docs(docs: List[Document]) -> str:
    final_text = ""

    for doc in docs:
        final_text += doc.page_content
        final_text += "\n"

    return final_text


def unique_docs(docs: List[Document]):
    unique_docs_list = []

    for doc in docs:
        if doc.page_content not in [doc.page_content for doc in unique_docs_list]:
            unique_docs_list.append(doc)

    return unique_docs_list
