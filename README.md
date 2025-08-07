# Medical Chatbot with Continuous Conversation and Retrieval-Augmented Generation (RAG)

This project implements a medical chatbot that supports **natural multi-turn conversations** and provides **accurate, context-aware answers** using a Retrieval-Augmented Generation (RAG) system based on a specific medical book PDF.

---

## Features

- **Continuous multi-turn chat** with conversation memory to remember user context and maintain dialogue flow.
- **Retrieval-Augmented Generation (RAG)** to answer symptom- and treatment-specific queries by retrieving relevant information from a medical book.
- **Intent-based routing** between a conversational chat chain and a RAG-based Q&A chain.
- Supports bilingual user input (e.g., Bangla and English).
- Uses LangChain LCEL framework for modular and scalable pipeline construction.
- Memory backend can be in-memory or Redis for production use.

---

## Architecture Overview

### 1. Continuous Chat Chain
- Uses `RunnableWithMessageHistory` to maintain conversation history.
- Handles general dialogue, clarifications, greetings, empathy, and small talk.
- Stores conversation memory per user session.

### 2. Q&A (RAG) Chain
- Uses vector search to retrieve relevant passages from a medical book PDF.
- Injects retrieved context into a prompt instructing the model to answer factually and precisely.
- Stateless chain handling symptom- and treatment-specific queries.

### 3. Orchestrator / Router
- Detects user intent (via heuristic or ML classifier).
- Routes user inputs to either the chat chain or Q&A chain accordingly.
- Combines responses seamlessly for natural user experience.

---

## Usage

### Prerequisites

- Python 3.8+
- LangChain LCEL
- OpenAI API key (or compatible LLM provider)
- Redis (optional, for persistent memory)
- Vector database (e.g., Chroma, FAISS) with medical book embeddings

### Setup

1. Install dependencies:

```bash
pip install langchain-openai langchain-core redis chromadb
