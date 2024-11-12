import logging
from typing import List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

from config import config
from doc_utils import _unique_docs, format_retrieved_docs
from models import CitedAnswer
from weaviate_io import WeaviateIO
from youtube_utils import (
    TranscriptItem,
    format_time,
    format_youtube_url_with_timestamp,
    get_transcript_with_timestamps,
    get_video_id,
    prepare_chunks_for_transcript,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

weaviate_io = WeaviateIO()

def answer_question(video_id: str, url: str, question: str):
    try:
        with st.spinner("Processing video transcript..."):
            transcript = get_transcript_with_timestamps(video_id)
            if transcript:
                display_full_transcript(transcript)
                
                chunks = prepare_chunks_for_transcript(transcript)
                
                vectorstore = weaviate_io.setup_vectorstore_for_video_and_transcript(chunks, video_id)
                
                retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                search_kwargs=dict(
                    alpha=0.7,
                    score_threshold=0.7,
                    return_metadata=["explain_score", "score", "distance"],
                    k=3,
                )
                )

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
                structured_llm = llm.with_structured_output(CitedAnswer)

                retrieved_docs = retriever.invoke(question)
                unique_docs = _unique_docs(retrieved_docs)
                formatted_docs = format_retrieved_docs(unique_docs)
                
                display_context_and_prompt(formatted_docs, prompt.format(docs=formatted_docs, question=question), question)
                
                rag_chain = prompt | structured_llm
                
                with st.spinner("Generating answer..."):
                    answer: CitedAnswer = rag_chain.invoke({"docs": formatted_docs, "question": question}) # type: ignore
                    logger.debug("got answer")
                    weaviate_io.close()
                    
                    display_answer(answer)
                    display_video(answer, url)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"An error occurred while processing the video: {str(e)}")

def display_full_transcript(transcript: List[TranscriptItem]):
    with st.expander("📝 Video Transcript", expanded=False):
        st.markdown("### Complete Video Transcript")
        transcript_text = ""
        for entry in transcript:
            timestamp = format_time(entry.start)
            transcript_text += f"**[{timestamp}]** *{entry.text}*\n\n"
        st.markdown(transcript_text)

def display_context_and_prompt(formatted_context: str, prompt: str, question: str):
    with st.expander("🔍 Retrieved Context", expanded=False):
        st.markdown("### Retrieved Context")
        st.markdown(formatted_context)
        st.markdown("### Prompt")
        st.markdown(prompt)
        st.markdown("### Question")
        st.markdown(question)

def display_answer(answer: CitedAnswer):
    with st.container():
        logger.debug(f"Answer: {answer}")
        st.markdown("### Answer")
        st.markdown(answer.answer)
        st.markdown("#### Citations")
        for citation in answer.citations:
            st.markdown(f"**[{citation.timestamp}]** - {citation.quote}")

def display_video(answer: CitedAnswer, url):
    with st.container():
        timestamp_second = int(answer.citations[0].timestamp.split(':')[0])
        st.markdown("### Video")
        video_url = format_youtube_url_with_timestamp(url, timestamp_second)
        logger.debug(f"Video URL: {video_url}")
        st.video(video_url, start_time=timestamp_second)


st.title("YouTube Video Q&A")

url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Enter your question:")
video_id = None
if url and question:
    try: 
        video_id = get_video_id(url)
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        st.error(f"An error occurred while extracting the video ID: {str(e)}")

    if video_id:
        answer_question(video_id, url, question)
    else:
        st.error("Invalid YouTube URL")
