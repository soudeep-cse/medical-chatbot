# main.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os
import re
import unicodedata
import string
import pickle
import tiktoken
from tqdm import tqdm

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Cleaning
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?i)page\s*\d+", "", text)
    text = re.sub(r"(CMDT|Merck Manual|Oxford Handbook|Current Medical Diagnosis and Treatment)[^\n]*", "", text)
    punctuations = string.punctuation + "“”‘’•…–—―•৳।॥"
    text = text.translate(str.maketrans('', '', punctuations))
    text = ''.join(
        c if (("\u0980" <= c <= "\u09FF") or ("\u0041" <= c <= "\u007A") or c in (' ', '\n', '\t', '.', ',', ':', ';', '-', '?', '!', '(', ')', '"', "'"))
        else ' '
        for c in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\x0c", "")
    return text

# ---------------------------------------------

from langchain_community.document_loaders import PyPDFLoader

def load_books(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        try:
            pages = loader.load()
            for p in pages:
                p.page_content = clean_text(p.page_content)
                p.metadata["source"] = path
            docs.extend(pages)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return docs


# Prepare Chunks
def prepare_chunks():
    pdf_files = [
        "data/CURRENT_Medical_Diagnosis_Treatment_Original.pdf",
        "data/Medical_book.pdf",
        "data/medicine_book.pdf",
    ]

    if not os.path.exists("cache/chunks.pkl"):
        pdf_docs = load_books(pdf_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(pdf_docs)

        def filter_large_chunks(chunks, model_name="text-embedding-3-large", max_tokens=8000):
            enc = tiktoken.encoding_for_model(model_name)
            return [chunk for chunk in chunks if len(enc.encode(chunk.page_content)) <= max_tokens]

        chunks = filter_large_chunks(chunks)
        os.makedirs("cache", exist_ok=True)
        with open("cache/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
    else:
        with open("cache/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    return chunks

# Build Chain
def build_chain():
    chunks = prepare_chunks()
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    persist_dir = "db"

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
        vectorstore.persist()
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=20)

    condense_question_prompt = PromptTemplate.from_template("""
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone question:
    """)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a knowledgeable, concise, and trustworthy personal medical assistant.

        Please answer the question **only** based on the information in the context below. Do not use any outside information or guess.

        <context>
        {context}
        </context>

        If you find any broken, incomplete, or misspelled words in the context, please mentally correct or infer the intended correct words before answering.

        Question: {question}

        ➡️ If the answer is found in the context:
        1. Provide clear and practical first aid advice.
        2. Name the safest medication (e.g., Paracetamol).
        3. If the condition seems serious, add at the end: "Go to the hospital as soon as possible."
        4. Always end the answer with: (Source: CMDT, Merck Manual or Medical Encyclopedia)

        ➡️ If the answer is NOT found in the context, respond exactly with:
        "I couldn't find an exact answer in the book. Please consult a doctor."

        Make your response sound like a professional doctor giving thoughtful and empathetic advice, using natural, human-like language.
        """
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
    )

    return rag_chain
