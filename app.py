# app.py
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import streamlit as st
import weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from pydantic import BaseModel, Field
from weaviate.auth import AuthApiKey
from youtube_transcript_api import YouTubeTranscriptApi


class Citation(BaseModel):
    """Citation from a YouTube video"""
    timestamp: str = Field(
        description="The timestamp in the video where this information appears (in format MM:SS or HH:MM:SS)"
    )
    quote: str = Field(
        description="The relevant quote from the video that answers the question"
    )
    video_id: str = Field(
        description="The YouTube video ID where this information appears"
    )

class AnswerWithCitations(BaseModel):
    """Answer with citations from YouTube videos"""
    answer: str = Field(
        description="The answer to the user's question"
    )
    citations: List[Citation] = Field(
        description="List of citations that support the answer"
    )

# Initialize environment
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"

WEAVIATE_URL = "https://xxx.gcp.weaviate.cloud"
WEAVIATE_AUTH_KEY = "xxx"

headers = {
    "X-AWS-Access-Key": "xxx",
    "X-AWS-Secret-Key": "xxx",
}

client = weaviate.connect_to_weaviate_cloud(
    WEAVIATE_URL, auth_credentials=AuthApiKey(WEAVIATE_AUTH_KEY), headers=headers
)

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be',):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

def get_transcript_with_timestamps(video_id: str) -> List[Dict]:
    """Get video transcript with timestamps"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None
    
def get_video_url_with_timestamp(video_id: str, timestamp: str) -> str:
    """Convert timestamp to seconds and generate YouTube URL"""
    # Convert timestamp (MM:SS or HH:MM:SS) to seconds
    parts = timestamp.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        total_seconds = int(minutes) * 60 + int(seconds)
    else:
        hours, minutes, seconds = parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    
    return f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}"

def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_transcript(transcript_entries: List[Dict]) -> List[Document]:
    # Create the text content while preserving timestamp metadata
    text_content = []
    metadata_map = {}
    
    for idx, sentence in enumerate(transcript_entries):
        text = sentence['text']
        metadata_map[text] = {
            'start': sentence['start'],
            'end_time': sentence['start'] + sentence['duration']
        }
        text_content.append(text)
    
    # Join all text content
    full_text = ''

    for idx, text in enumerate(text_content):
        full_text += f"[Timestamp: {metadata_map[text]['start']} - {metadata_map[text]['end_time']}] {text}"
        full_text += "\n"
    
    # Initialize the splitter with appropriate settings for transcripts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=100,  # Provides context between chunks
        length_function=len,
    )
    
    # Split the text while preserving metadata
    chunks = []
    for chunk in text_splitter.split_text(full_text):
        # Find the first and last sentence in this chunk to get time boundaries
        chunk_sentences = [s for s in text_content if s in chunk]
        if chunk_sentences:
            start_metadata = metadata_map[chunk_sentences[0]]
            end_metadata = metadata_map[chunk_sentences[-1]]
            
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    'start': start_metadata['start'],
                    'end_time': end_metadata['end_time']
                }
            ))
    
    return chunks

def setup_vectorstore(chunks: List[Document], video_id: str):
    """Setup Weaviate vector store"""
    # Create schema if not exists
    class_name = f"YoutubeTranscript_{video_id}"
    
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID)
    
    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=class_name,
        text_key="text"
    )
    
    return vectorstore

def create_chain(retriever):
    """Create RAG chain with source citation"""
    llm = ChatBedrock(
        model=LLM_MODEL_ID,
        model_kwargs=dict(temperature=0, top_k=100, top_p=0.95),
        provider="anthropic",
    )
    
    # Modify the prompt to request structured output
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

    # Use structured output with the LLM
    structured_llm = llm.with_structured_output(AnswerWithCitations)

    return prompt, format_docs, structured_llm


# Streamlit UI
st.title("YouTube Video Q&A")

url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Enter your question:")

if url and question:
    video_id = get_video_id(url)
    if video_id:
        with st.spinner("Processing video transcript..."):
            transcript = get_transcript_with_timestamps(video_id)
            if transcript:
                # Show full transcript in an expander
                with st.expander("📝 Full Video Transcript", expanded=False):
                    st.markdown("### Complete Video Transcript")
                    # Create a DataFrame for better display
                    transcript_text = ""
                    for entry in transcript:
                        timestamp = format_time(entry['start'])
                        transcript_text += f"**[{timestamp}]** {entry['text']}\n\n"
                    st.markdown(transcript_text)
                
                print(transcript)
                chunks = process_transcript(transcript)
                vectorstore = setup_vectorstore(chunks, video_id)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                prompt, format_docs, structured_llm = create_chain(retriever)
                
                # Get retrieved documents and format them
                retrieved_docs = retriever.get_relevant_documents(question)
                formatted_context = format_docs(retrieved_docs)
                
                # Create the formatted prompt
                formatted_prompt = prompt.format(
                    docs=formatted_context,
                    question=question
                )
                
                # Show context and prompt in expanders
                with st.expander("🔍 Retrieved Context", expanded=False):
                    st.markdown("### Retrieved Context from Video")
                    st.markdown(formatted_context)
                
                with st.expander("📝 Prompt Sent to LLM", expanded=False):
                    st.markdown("### Complete Prompt")
                    st.markdown(f"```\n{formatted_prompt}\n```")
                
                # Create the chain
                rag_chain = (
                    {"docs": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | structured_llm
                )
                
                # Create a container for the output
                st.markdown("### Answer")
                output_container = st.empty()
                accumulated_answer = ""
                
                with st.spinner("Generating answer..."):
                    for chunk in rag_chain.stream(question):
                        accumulated_answer += chunk # type: ignore
                        output_container.json(accumulated_answer)