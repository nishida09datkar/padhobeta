# Padhobeta - AI Educational Chatbot

An AI-powered educational chatbot that lets you upload academic documents and ask questions about them. Built with a multi-agent architecture using Groq's fast LLM inference.

## Features

- **Multi-format support**: PDF, DOCX, PPTX, and Images (OCR)
- **Smart PDF parsing**: Uses [pdf-inspector](https://github.com/firecrawl/pdf-inspector) for intelligent PDF classification and text extraction
- **RAG pipeline**: Document chunking, embedding, and retrieval with FAISS
- **Educational guardrail**: Only answers academic/educational questions
- **Source citations**: Answers include references to source documents

## Architecture

```
User Upload → Document Parser Agent → Chunk → Embed → FAISS Store
User Query  → Query Classifier Agent → Educational? → RAG Agent → Response Generator Agent → Answer
```

Four agents work together:
1. **Document Parser** — Detects format, routes to correct parser, extracts text
2. **Query Classifier** — Filters out non-educational queries via Groq LLM
3. **RAG Agent** — Retrieves relevant document chunks via vector similarity search
4. **Response Generator** — Generates answers using Groq LLM with retrieved context

## Setup

### 1. Install dependencies

```bash
cd padhobeta
pip install -r requirements.txt
```

### 2. Configure API key

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
```

### 3. Run

```bash
python main.py
```

The server starts at `http://127.0.0.1:8000`

## API Endpoints

### Upload a document
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@your_textbook.pdf"
```

### Ask a question
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Newtons second law?", "document_id": "abc123"}'
```

### List uploaded documents
```bash
curl http://127.0.0.1:8000/documents
```

### Delete a document
```bash
curl -X DELETE http://127.0.0.1:8000/documents/abc123
```

### Health check
```bash
curl http://127.0.0.1:8000/health
```

## Tech Stack

- **Backend**: FastAPI
- **LLM**: Groq (llama-3.3-70b-versatile)
- **PDF Parsing**: pdf-inspector (Rust-based, via Python bindings)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Other parsers**: python-docx, python-pptx, pytesseract

## Project Structure

```
padhobeta/
├── main.py                  # FastAPI app
├── config.py                # Settings
├── agents/
│   ├── document_parser.py   # Document parsing agent
│   ├── query_classifier.py  # Educational query filter
│   ├── rag_agent.py         # Context retrieval
│   └── response_generator.py # Answer generation
├── parsers/
│   ├── pdf_parser.py        # pdf-inspector wrapper
│   ├── docx_parser.py       # python-docx wrapper
│   ├── pptx_parser.py       # python-pptx wrapper
│   └── image_parser.py      # OCR wrapper
├── vector_store/
│   └── store.py             # FAISS + embeddings
├── models/
│   └── schemas.py           # Pydantic models
├── utils/
│   └── chunking.py          # Text chunking
├── requirements.txt
└── .env
```
