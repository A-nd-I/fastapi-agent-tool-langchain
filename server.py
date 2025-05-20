from fastapi import FastAPI, File, UploadFile, Form
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import numpy as np
from io import BytesIO
import tempfile
import base64
import random

from langserve import add_routes
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.tools import tool
import uvicorn
import os
from dotenv import load_dotenv
from typing import Annotated
from langchain.agents import ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")

app = FastAPI(
    title="Legal Document Comparison API",
    version="1.0",
    description="API para comparar documentos legales y generar frases de abogacía"
)

# Frases de abogacía
frases_abogacia = [
    "La justicia no es solo una virtud, es un deber fundamental.",
    "El derecho es el conjunto de condiciones que permiten a la libertad de cada uno acomodarse a la libertad de todos.",
    "La ley debe ser como la muerte, que no exceptúa a nadie.",
    "La justicia es la reina de las virtudes republicanas.",
    "El fin del Derecho es la paz; el medio para ello es la lucha.",
    "La justicia es el pan del pueblo.",
    "Donde hay sociedad, hay derecho.",
    "La ley es dura, pero es la ley.",
    "La justicia sin fuerza es impotente; la fuerza sin justicia es tiránica.",
    "El derecho es el arte de lo bueno y lo equitativo."
]

@app.post("/compare-pdfs")
async def compare_pdfs_endpoint(pdf1: UploadFile, pdf2: UploadFile):
    """Endpoint para comparar dos PDFs usando embeddings"""
    try:
        # Leer los PDFs
        pdf1_content = await pdf1.read()
        pdf2_content = await pdf2.read()
        
        # Convertir a base64
        pdf1_base64 = base64.b64encode(pdf1_content).decode()
        pdf2_base64 = base64.b64encode(pdf2_content).decode()
        
        def pdf_to_text(pdf_base64: str, max_chars: int = 8000) -> str:
            pdf_bytes = base64.b64decode(pdf_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.flush()
                
                loader = PyPDFLoader(temp_pdf.name)
                pages = loader.load()
                
                full_text = ' '.join(page.page_content for page in pages)
                if len(full_text) > max_chars:
                    return full_text[:max_chars] + "... (texto truncado)"
                return full_text

        # Convertir PDFs a texto
        text1 = pdf_to_text(pdf1_base64)
        text2 = pdf_to_text(pdf2_base64)
        
        # Crear embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=1000
        )
        
        # Calcular similitud
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)
        
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_percentage = similarity * 100
        
        # Generar respuesta detallada
        response = {
            "similarity_score": float(similarity),
            "similarity_percentage": f"{similarity_percentage:.2f}%",
            "analysis": {
                "percentage": f"{similarity:.2%}",
                "interpretation": ""
            }
        }
        
        if similarity > 0.95:
            response["analysis"]["interpretation"] = f"Los documentos son prácticamente idénticos (Similitud: {similarity_percentage:.2f}%)"
        elif similarity > 0.8:
            response["analysis"]["interpretation"] = f"Los documentos son muy similares (Similitud: {similarity_percentage:.2f}%)"
        elif similarity > 0.6:
            response["analysis"]["interpretation"] = f"Los documentos tienen algunas similitudes (Similitud: {similarity_percentage:.2f}%)"
        else:
            response["analysis"]["interpretation"] = f"Los documentos son significativamente diferentes (Similitud: {similarity_percentage:.2f}%)"
            
        return response
        
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "type": "processing_error"
        }

@app.get("/frase-legal")
async def get_random_legal_phrase():
    """Endpoint para obtener una frase legal aleatoria"""
    return {
        "frase": random.choice(frases_abogacia)
    }

# Configurar el LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=1000
)

# Definir herramientas
tools = [
    Tool(
        name="ComparePDFs",
        func=compare_pdfs_endpoint,
        description="Compara dos PDFs y determina su similitud usando embeddings"
    )
]

# Configurar el prompt del agente
prefix = """Eres un asistente legal especializado en comparación de documentos."""

suffix = """Comienza!

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# Inicializar el agente
memory = ConversationBufferMemory(memory_key="chat_history")

agent = ZeroShotAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt),
    tools=tools,
    verbose=True
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    max_execution_time=30
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
