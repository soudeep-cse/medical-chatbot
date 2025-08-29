from typing import Dict, List, Optional
from crewai import Agent, Task
from langchain.agents import Tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

class BaseAgent(Agent):
    def __init__(self, name: str, role: str, tools: List[Tool] = None, llm=None):
        super().__init__(
            name=name,
            role=role,
            backstory=f"You are {name}, an expert medical AI assistant.",
            llm=llm or ChatOpenAI(temperature=0.7, model="gpt-4"),
            tools=tools or [],
            verbose=True,
            allow_delegation=False,
        )

class RAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Medical Knowledge Agent",
            role="Medical Expert",
        )

class WebSearchAgent(BaseAgent):
    def __init__(self):
        self.tool = TavilySearchResults()
        super().__init__(
            name="Medical Research Agent",
            role="Web Researcher",
            tools=[self.tool]
        )

class MedicineInfoAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Pharmacy Expert Agent",
            role="Pharmacist",
        )

class TavilyAnswer(Agent):
    def __init__(self):
        self.tool = TavilySearchResults()
        super().__init__(
            name="Tavily Answer Agent",
            role="Medical Researcher",
            backstory="You are a medical researcher who is an expert in finding prices and medicine names based on genre.",
            tools=[self.tool],
            llm=ChatOpenAI(temperature=0.7, model="gpt-4"),
            verbose=True,
            allow_delegation=False,
        )