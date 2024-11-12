import logging
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

@dataclass
class TranscriptItem:
    text: str
    start: float
    duration: float
    
    @property
    def end_time(self) -> float:
        return self.start + self.duration

@dataclass
class TranscriptChunkMetadata:
    start: float
    end_time: float

class InvalidYouTubeURLError(Exception):
    """Raised when the provided URL is not a valid YouTube URL"""
    pass

class YouTubeTranscriptError(Exception):
    """Raised when there is an error fetching or processing the YouTube transcript"""
    pass

def get_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Video ID
        
    Raises:
        InvalidYouTubeURLError: If the URL is not a valid YouTube URL or video ID cannot be extracted
        
    Examples:
        >>> YouTubeUtils.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeURLError("URL must be a non-empty string")
        
    try:
        parsed_url = urlparse(url)
        
        # Handle youtu.be format
        if parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path.lstrip('/')
            if not video_id:
                raise InvalidYouTubeURLError("No video ID found in youtu.be URL")
            return video_id
            
        # Handle youtube.com formats
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            # Handle /watch?v= format
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                if 'v' not in query_params:
                    raise InvalidYouTubeURLError("No video ID parameter 'v' found in URL")
                return query_params['v'][0]
                
            # Handle /embed/ format
            if parsed_url.path.startswith('/embed/'):
                video_id = parsed_url.path.split('/embed/')[-1]
                if not video_id:
                    raise InvalidYouTubeURLError("No video ID found in embed URL")
                return video_id
                
            # Handle /v/ format
            if parsed_url.path.startswith('/v/'):
                video_id = parsed_url.path.split('/v/')[-1]
                if not video_id:
                    raise InvalidYouTubeURLError("No video ID found in /v/ URL")
                return video_id
                
        raise InvalidYouTubeURLError("Invalid YouTube URL format")
        
    except Exception as e:
        if isinstance(e, InvalidYouTubeURLError):
            raise
        raise InvalidYouTubeURLError(f"Failed to extract video ID: {str(e)}")

def get_transcript_with_timestamps(video_id: str) -> List[TranscriptItem]:
    """
    Get video transcript with timestamps.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        List[TranscriptEntry]: List of transcript entries with timestamps
        
    Raises:
        YouTubeTranscriptError: If transcript cannot be fetched
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [TranscriptItem(**entry) for entry in transcript]
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        raise YouTubeTranscriptError(f"Could not fetch transcript: {str(e)}")

def format_time(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string (MM:SS)
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def prepare_chunks_for_transcript(transcript_items: List[TranscriptItem]) -> List[Document]:
    """
    Process transcript items into Document objects for vector store.
    
    Args:
        transcript_items (List[TranscriptEntry]): Raw transcript items
        
    Returns:
        List[Document]: Processed documents with metadata
    """
    try:
        metadata_map = {
            entry.text: TranscriptChunkMetadata(
                start=entry.start,
                end_time=entry.end_time
            )
            for entry in transcript_items
        }
        
        full_text = '\n'.join(
            f"[Timestamp: {entry.start} - {entry.end_time}] {entry.text}"
            for entry in transcript_items
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        chunks = []
        for chunk in text_splitter.split_text(full_text):
            chunk_sentences = [
                entry.text for entry in transcript_items 
                if entry.text in chunk
            ]
            if chunk_sentences:
                start_metadata = metadata_map[chunk_sentences[0]]
                end_metadata = metadata_map[chunk_sentences[-1]]
                
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        'start': start_metadata.start,
                        'end_time': end_metadata.end_time
                    }
                ))
        
        return chunks
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        raise YouTubeTranscriptError(f"Error processing transcript: {str(e)}")

def format_youtube_url_with_timestamp(url: str, timestamp_seconds: int) -> str:
    """
    Add timestamp to YouTube URL.
    
    Args:
        url (str): YouTube video URL
        timestamp_seconds (int): Timestamp in seconds
        
    Returns:
        str: URL with timestamp
    """
    if 'youtube.com' not in url and 'youtu.be' not in url:
        return url
        
    url = re.sub(r'[?&]t=\d+s', '', url)
    separator = '?' if '?' not in url else '&'
    return f"{url}{separator}t={timestamp_seconds}s"
