import re
import hashlib
import unicodedata

def generate_stable_id(text: str) -> str:
    """Generates a stable MD5 hash ID for a given text string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_bangla_text_simple(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)

    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\u0980-\u09FF\s.,?!ঃ]', '', text)
    text = text.replace('্ ', '্')

    return text.strip()

input_filename = 'extracted_text.txt'
output_filename = 'simple_cleaned_bangla_text.txt'

try:
    print(f"▶️  Reading raw text from '{input_filename}'...")
    with open(input_filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print("⚙️  Starting the simple cleaning process...")
    cleaned_text = clean_bangla_text_simple(raw_text)

    print(f"💾  Saving cleaned text to '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print("\n✅ Successfully cleaned the text using the simple method.")
    print(f"   Cleaned data has been saved to '{output_filename}'")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

from langchain_community.document_loaders import TextLoader
loader = TextLoader("simple_cleaned_bangla_text.txt",encoding="utf-8")
documents = loader.load()

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(documents)


#import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Chroma vector store
vector_store = FAISS.from_documents(texts, embeddings)

retreiver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful and intelligent assistant for a Retrieval-Augmented Generation (RAG) system.

You will receive user questions in either Bangla or English.
Use **only** the retrieved context provided below to answer the question accurately.
If the answer is not explicitly available, try to infer it **reasonably** from the context.
If no answer can be found or inferred, respond with:
- "দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।" (for Bangla questions), or
- "Sorry, I couldn't find the answer to your question." (for English questions).

Always respond **in the same language** as the user's question.

Context to search through:
{context}

Question: {question}

Instructions:
- If the question can be answered by **a single name, phrase, number, or entity**, return the **shortest possible answer** — ideally in **1–3 words**, no extra sentence.
- If the question **requires explanation or description**, then return a **clear full sentence**.
- Never include unnecessary repetition of question terms in the answer.
- Focus on precision and brevity.
- Always respond **in the same language** as the user's question.

Answer:
"""
)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retreiver | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


from fastapi import FastAPI
from langserve import add_routes
import uvicorn

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/chain",
)