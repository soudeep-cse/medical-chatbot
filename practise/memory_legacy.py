from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re
import unicodedata
import string
import pickle


load_dotenv()

# Set your OpenAI API key
os.environ["OPEN_AI_API"] = "OPENAI_API_KEY"


def clean_text(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove page numbers and common headers/footers
    text = re.sub(r"(?i)page\s*\d+", "", text)
    text = re.sub(r"(CMDT|Merck Manual|Oxford Handbook|Current Medical Diagnosis and Treatment)[^\n]*", "", text)

    # Remove special characters and punctuation (both English and Bangla)
    punctuations = string.punctuation + "“”‘’•…–—―•৳।॥"
    text = text.translate(str.maketrans('', '', punctuations))

    # Remove weird Unicode private use or non-printable characters
    # Unicode Private Use Area: U+E000–U+F8FF, and similar blocks
    # Replace any char outside normal printable Bangla/English with space
    text = ''.join(
        c if (
            ('\u0980' <= c <= '\u09FF')  # Bangla unicode block
            or ('\u0041' <= c <= '\u007A')  # Basic Latin letters (A-z)
            or c in (' ', '\n', '\t', '.', ',', ':', ';', '-', '?', '!', '(', ')', '"', "'")
        ) else ' '
        for c in text
    )

    # Remove multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Remove form feed and invisible characters
    text = text.replace("\x0c", "")

    return text



def load_books(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for p in pages:
            p.page_content = clean_text(p.page_content)  # ✅ Apply preprocessing
            p.metadata["source"] = path
        docs.extend(pages)
    return docs


pdf_files = [r"data\CURRENT_Medical_Diagnosis_Treatment_Original.pdf", r"data\Medical_book.pdf"]
documents = load_books(pdf_files)



import tiktoken

def filter_large_chunks(chunks, model_name="text-embedding-3-large", max_tokens=8000):
    enc = tiktoken.encoding_for_model(model_name)
    filtered = []
    for chunk in chunks:
        tokens = len(enc.encode(chunk.page_content))
        if tokens <= max_tokens:
            filtered.append(chunk)
    return filtered

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
chunks = filter_large_chunks(chunks)  # ✅ Filter oversized chunks

# Save chunks:
pickle.dump(chunks, open("cache/chunks.pkl", "wb"))  # ✅ Save
chunks = pickle.load(open("cache/chunks.pkl", "rb"))  # ✅ Just load preprocessed chunks


# Embed + Vector DB
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(embedding_function=embedding, persist_directory="db")

batch_size = 100
for i in tqdm(range(0, len(chunks), batch_size)):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)
vectorstore.persist()

# ========== Retriever ==============
multiquery_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = PromptTemplate(
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

    ➡️ If the answer is NOT found in the context, respond exactly with:
    "I couldn't find an exact answer in the book. Please consult a doctor."

    Make your response sound like a professional doctor giving thoughtful and empathetic advice, using natural, human-like language.
    """
)

# Merge chunks function
def merge_chunks(docs):
    return "\n\n".join([doc.page_content for doc in docs])

merge_chunks_lambda = RunnableLambda(merge_chunks)

from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10  # ✅ Only remember the last 10 interactions
)


# Define the LLM runnable
llm = ChatOpenAI(model="gpt-4", temperature=0.5)


from langchain.chains import ConversationalRetrievalChain

# Use legacy langchain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=multiquery_retriever,
    memory=memory,
    verbose=True,
)

