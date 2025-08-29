from typing import Dict
from langchain.prompts import PromptTemplate

# RAG System Prompt
RAG_PROMPT = PromptTemplate(
    template="""You are a medical expert assistant. Use the following context from medical literature to answer the user's question. If the answer cannot be found in the context, explicitly say so.

Context: {context}

Question: {question}

Consider the following guidelines:
1. Only provide information that is supported by the medical literature context
2. If uncertain, acknowledge the limitations of the available information
3. For medical emergencies, always advise seeking immediate professional help
4. Maintain a professional and clear communication style

Answer:""",
    input_variables=["context", "question"]
)

# Medicine Info Agent Prompt
MEDICINE_PROMPT = PromptTemplate(
    template="""As a pharmacy expert, analyze the following symptoms and patient information to suggest appropriate medications. Include drug classifications, usage guidelines, and potential alternatives.

Patient Info:
{patient_info}

Symptoms:
{symptoms}

Guidelines:
1. Prioritize well-established medications with proven efficacy
2. Consider potential drug interactions and contraindications
3. Include both generic and brand name options
4. Specify if prescription is required
5. Note any relevant warnings or precautions

Response:""",
    input_variables=["patient_info", "symptoms"]
)

# Web Search Agent Prompt
WEB_SEARCH_PROMPT = PromptTemplate(
    template="""As a medical researcher, search reliable medical sources for information about:

Query: {query}

Guidelines:
1. Focus on peer-reviewed sources and official medical websites
2. Prioritize recent information (within last 5 years when possible)
3. Cross-reference findings for accuracy
4. Include source citations
5. Summarize key findings clearly

Research Results:""",
    input_variables=["query"]
)

# Bilingual Response Prompt
BILINGUAL_PROMPT = PromptTemplate(
    template="""Provide a bilingual response in both English and Bangla:

English Response:
{english_response}

বাংলা অনুবাদ:
{bangla_translation}""",
    input_variables=["english_response", "bangla_translation"]
)
