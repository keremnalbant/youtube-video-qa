import logging
from typing import List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

from config import config
from doc_utils import format_retrieved_docs, unique_docs
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

                vectorstore = weaviate_io.setup_vectorstore_for_video_and_transcript(
                    chunks, video_id
                )

                retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=dict(
                        alpha=0.7,
                        score_threshold=0.7,
                        return_metadata=["explain_score", "score", "distance"],
                        k=3,
                    ),
                )

                llm = ChatBedrock(
                    model=config.LLM_MODEL_ID,
                    model_kwargs=dict(temperature=0, top_k=100, top_p=0.95),
                    provider="anthropic",
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """You are a helpful assistant that answers questions about YouTube videos.
                You will be given transcript(captions/subtitles) of youtube videos along with the timestamps (MM:SS).
                For each answer, provide relevant timestamps and quotes from the videos.
                Make sure to extract exact timestamps where the information appears.""",
                        ),
                        (
                            "human",
                            """Transcript: {docs}

                 Question: {question}""",
                        ),
                    ]
                )

                retrieved_docs = retriever.invoke(question)
                unique_docs_list = unique_docs(retrieved_docs)
                formatted_docs = format_retrieved_docs(unique_docs_list)

                display_context_and_prompt(
                    formatted_docs,
                    prompt.format(docs=formatted_docs, question=question),
                    question,
                )

                structured_llm = llm.with_structured_output(CitedAnswer)

                rag_chain = prompt | structured_llm

                with st.spinner("Generating answer..."):
                    answer: CitedAnswer = rag_chain.invoke(
                        {"docs": formatted_docs, "question": question}
                    )  # type: ignore
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
            timestamp = format_time(entry.start_time)
            transcript_text += f"**[{timestamp}]** *{entry.text}*\n\n"
        st.markdown(transcript_text)


def display_context_and_prompt(formatted_context: str, prompt: str, question: str):
    with st.expander("🔍 Retrieved Context & Augmented Prompt", expanded=False):
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
        timestamp_seconds = int(answer.citations[0].timestamp.split(":")[1])
        st.markdown("### Video")
        video_url = format_youtube_url_with_timestamp(url, timestamp_seconds)
        logger.debug(f"Video URL: {video_url}")
        st.video(video_url, start_time=timestamp_seconds)


st.title("YouTube Video Q&A")

# Initialize button clicked state
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


def on_button_click():
    st.session_state.button_clicked = True


url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Enter your question:")

# Only enable button if both fields are filled
button_disabled = not url or not question
st.button("Ask AI", disabled=button_disabled, type="primary", on_click=on_button_click)

# Only process if button was explicitly clicked
if st.session_state.button_clicked:
    try:
        video_id = get_video_id(url)
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        st.error(f"An error occurred while extracting the video ID: {str(e)}")
    else:
        if video_id:
            answer_question(video_id, url, question)
        else:
            st.error("Invalid YouTube URL")

    # Reset button state
    st.session_state.button_clicked = False
