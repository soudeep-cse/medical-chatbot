from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from main import prepare_chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import os

app = FastAPI(
    title="Medical Assistant RAG API",
    version="1.0",
    description="Conversational Medical Assistant built with LangChain & FastAPI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Shared model setup
# =========================
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

# Shared prompt for RAG chain
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
    4. Always end the answer with: (Source: CMDT, Merck Manual or Medical Encyclopedia)

    ➡️ If the answer is NOT found in the context, respond exactly with:
    "I couldn't find an exact answer in the book. Please consult a doctor."

    Make your response sound like a professional doctor giving thoughtful and empathetic advice, using natural, human-like language.
    """
)

# Per-user memory storage
user_sessions = {}

def get_chain_for_user(user_id: str) -> ConversationalRetrievalChain:
    if user_id not in user_sessions:
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=prompt,
            verbose=False,
        )
        user_sessions[user_id] = chain
    return user_sessions[user_id]

# Request schema
class ChatRequest(BaseModel):
    user_id: str
    question: str

from langchain.schema import SystemMessage, HumanMessage

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id
    question = request.question

    chain = get_chain_for_user(user_id)
    results_with_scores = vectorstore.similarity_search_with_score(question, k=3)

    if not results_with_scores:
        response = llm.invoke([
            SystemMessage(content="You are a kind and empathetic personal medical assistant."),
            HumanMessage(content=question),
        ])
        answer = response.generations[0][0].text
    else:
        best_score = results_with_scores[0][1]
        threshold = 0.3

        if best_score > threshold:
            from langchain.schema import SystemMessage, HumanMessage

            response = llm.invoke([
                SystemMessage(content="You are a kind and empathetic personal medical assistant."),
                HumanMessage(content=question),
            ])

            answer = response.content

        else:
            result = chain.invoke({"question": question})
            answer = result["answer"]

    return {"response": answer}

