# Medical Chatbot with CrewAI, RAG, and Web Search

This project implements a sophisticated medical chatbot that leverages **CrewAI** to manage a team of specialized AI agents. The chatbot provides accurate, context-aware answers by combining a Retrieval-Augmented Generation (RAG) system with web search capabilities.

---

## Project Description

This project is a conversational medical assistant that uses a combination of AI agents to provide comprehensive medical information. It is built with **CrewAI** to orchestrate a team of specialized agents, each with a specific role:

- **RAG Agent:** Answers questions based on a knowledge base of medical books.
- **Web Search Agent:** Searches the web for information not found in the local knowledge base.
- **Medicine Info Agent:** Provides information about specific medicines.
- **Tavily Answer Agent:** A specialized agent for finding prices and medicine names.

The application is built with a **FastAPI** backend and a **Streamlit** frontend, providing both a robust API and an easy-to-use interface.

---

## Features

- **Multi-Agent System:** Uses **CrewAI** to manage a team of specialized AI agents.
- **Retrieval-Augmented Generation (RAG):** Answers questions by retrieving information from a local knowledge base of medical books.
- **Web Search Fallback:** Automatically searches the web if the answer is not found in the local knowledge base.
- **Continuous Conversation:** Maintains conversation history for a natural, multi-turn chat experience.
- **Bilingual Support:** Can handle both English and Bangla user input.
- **Scalable Architecture:** Built with LangChain LCEL for a modular and scalable pipeline.
- **Persistent Memory:** Uses Redis for a persistent memory backend.

---

## Technologies and Frameworks

- **Core:** LangChain, FastAPI, Streamlit, CrewAI
- **LLM:** OpenAI's GPT-4 and `text-embedding-3-large`
- **Vector Store:** ChromaDB
- **Web Search:** Tavily
- **Memory:** Redis and in-memory `ConversationBufferWindowMemory`.
- **UI:** Streamlit
- **Backend:** FastAPI

---

## Models Used

- **Generation:** `gpt-4`
- **Embeddings:** `text-embedding-3-large`

---

## How to Run

### Prerequisites

- Python 3.8+
- OpenAI API key
- Redis (optional, for persistent memory)

### Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up your environment variables by creating a `.env` file from the `.env.example`.

### Running the Application

1.  **Run the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
2.  **Run the Streamlit UI:**
    ```bash
    streamlit run chat_ui.py
    ```

Now you can open the Streamlit UI in your browser to interact with the chatbot.
