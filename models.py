from typing import List

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation from a YouTube video"""
    timestamp: str = Field(
        description="The timestamp in the video where this information appears (in format MM:SS or HH:MM:SS)"
    )
    quote: str = Field(
        description="The relevant quote from the video that answers the question"
    )

class CitedAnswer(BaseModel):
    """Answer with citations from YouTube videos"""
    answer: str = Field(
        description="The answer to the user's question"
    )
    citations: List[Citation] = Field(
        description="List of citations that support and justify the answer"
    ) 