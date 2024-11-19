# YouTube Video Q&A

An AI-powered application that allows users to ask questions about YouTube videos and get answers with relevant timestamps and citations.

## Features

- Ask questions about any YouTube video with available transcripts
- Get AI-generated answers with exact timestamps and quotes
- View full video transcripts
- Watch video segments relevant to answers
- RAG (Retrieval Augmented Generation) powered by Claude 3.5 Sonnet

## Prerequisites

- Python 3.8+
- Make (optional, for using Makefile commands)
- AWS Account with Bedrock access
- Weaviate Cloud account

## Environment Setup

1. Clone the repository
2. Create a `.env` file in the root directory with the following variables:
```plaintext
WEAVIATE_URL=your_weaviate_url
WEAVIATE_AUTH_KEY=your_weaviate_auth_key
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
EMBEDDING_MODEL_ID=embedding-model-id
LLM_MODEL_ID=llm-model-id
```

## Installation

Using Make:
```bash
make install
```

Manual:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application
Using Make:
```bash
make run
```

Manual:
```bash
streamlit run main.py
```


## Usage

1. Enter a YouTube video URL
2. Type your question about the video
3. Click "Ask AI" to get an answer with relevant timestamps
4. View the answer and watch the relevant video segment

## Project Structure

- `main.py`: Main Streamlit application
- `config.py`: Configuration and environment variables
- `models.py`: Pydantic models for structured output
- `youtube_utils.py`: YouTube transcript and URL handling
- `weaviate_io.py`: Vector database operations
- `llm_utils.py`: LLM chain setup and operations