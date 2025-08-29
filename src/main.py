import os
from typing import Dict, List, Optional
import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from uuid import uuid4
from dotenv import load_dotenv

from src.agents.base import RAGAgent, WebSearchAgent, TavilyAnswer
from src.memory.redis_memory import RedisMemory
from src.utils.helpers import (
    load_medical_knowledge,
    validate_patient_info,
    format_medical_response
)

load_dotenv()

app = typer.Typer()
console = Console()
memory = RedisMemory()

# Initialize agents
rag_agent = RAGAgent()
web_agent = WebSearchAgent()
tavily_agent = TavilyAnswer()

# Load medical knowledge base
pdf_path = "data/Medical_book.pdf"
vectorstore = load_medical_knowledge(pdf_path)

def get_session_id() -> str:
    """Generate or retrieve session ID"""
    return str(uuid4())

@app.command()
def chat(translate: bool = typer.Option(False, "--translate", "-t", help="Enable English-Bangla translation")):
    """Start an interactive medical chat session"""
    session_id = get_session_id()
    
    # Get initial patient information
    console.print(Panel("Welcome to Medical Chatbot! Let's start with some basic information."))
    
    patient_info = {
        "name": typer.prompt("Your name"),
        "age": typer.prompt("Your age"),
        "gender": typer.prompt("Your gender (M/F/Other)"),
        "medical_history": typer.prompt("Any significant medical history (press Enter if none)") or None
    }
    
    errors = validate_patient_info(patient_info)
    if errors:
        console.print("[red]Please provide all required information.[/red]")
        return
    
    memory.save_patient_info(session_id, patient_info)
    console.print("\n[green]Thank you! You can now start asking medical questions.[/green]")
    
    while True:
        question = typer.prompt("\nYour question (or 'quit' to exit)")
        if question.lower() == "quit":
            break
            
        # Try RAG first
        context = vectorstore.similarity_search(question)
        if context:
            response = rag_agent.execute({"context": context, "question": question})
        else:
            # Fall back to web search
            response = web_agent.execute({"query": question})
            
        # Check if medicine info is requested
        if any(keyword in question.lower() for keyword in ["price of", "medicine for", "treatment for"]):
            medicine_info = tavily_agent.execute({
                "query": question
            })
            response += f"\n\nMedication Information:\n{medicine_info}"
            
        # Format and display response
        formatted_response = format_medical_response(response, translate=translate)
        console.print(Panel(formatted_response, title="Medical Assistant"))
        
        # Update conversation context
        memory.update_context(session_id, {
            "last_question": question,
            "last_response": response
        })

if __name__ == "__main__":
    app()