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

from langchain.vectorstores import FAISS


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
        
        def split_text_into_sections(text: str, section_length: int = 1000) -> list[str]:
            words = text.split()
            sections = []
            current_section = []
            current_length = 0
            
            for word in words:
                current_length += len(word) + 1  # +1 for space
                if current_length > section_length:
                    sections.append(' '.join(current_section))
                    current_section = [word]
                    current_length = len(word)
                else:
                    current_section.append(word)
            
            if current_section:
                sections.append(' '.join(current_section))
            return sections

        # Dividir textos en secciones
        sections1 = split_text_into_sections(text1)
        sections2 = split_text_into_sections(text2)
        
        # Crear embeddings para cada sección
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=1000
        )
        
        # Calcular similitud global
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)
        
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_percentage = similarity * 100

        # Analizar similitudes por sección
        section_similarities = []
        for i, sec1 in enumerate(sections1):
            sec1_embedding = embeddings.embed_query(sec1)
            best_match = {"similarity": 0, "section": "", "index": -1}
            
            for j, sec2 in enumerate(sections2):
                sec2_embedding = embeddings.embed_query(sec2)
                sec_similarity = np.dot(sec1_embedding, sec2_embedding) / (
                    np.linalg.norm(sec1_embedding) * np.linalg.norm(sec2_embedding)
                )
                
                if sec_similarity > best_match["similarity"]:
                    best_match = {
                        "similarity": sec_similarity,
                        "section": sec2[:200] + "...",  # Primeros 200 caracteres
                        "index": j
                    }
            
            section_similarities.append({
                "section1": sec1[:200] + "...",  # Primeros 200 caracteres
                "best_match": best_match,
                "similarity_score": best_match["similarity"]
            })

        # Identificar secciones más y menos similares
        sorted_similarities = sorted(section_similarities, key=lambda x: x["similarity_score"])
        different_sections = [s for s in sorted_similarities if s["similarity_score"] < 0.95]
        most_different = different_sections[:2] if different_sections else sorted_similarities[:2]
        
        response = {
            "similarity_score": float(similarity),
            "similarity_percentage": f"{similarity_percentage:.2f}%",
            "analysis": {
                "percentage": f"{similarity:.2%}",
                "interpretation": "",
                "similar_sections": [
                    {
                        "text1": section["section1"],
                        "text2": section["best_match"]["section"],
                        "similarity": f"{section['similarity_score']:.2%}"
                    } for section in most_different
                ],
                "different_sections": [
                    {
                        "text1": section["section1"],
                        "text2": section["best_match"]["section"],
                        "similarity": f"{section['similarity_score']:.2%}",
                        "position": f"Sección {i+1}"
                    } for i, section in enumerate(most_different)
                ]
            }
        }
        
        if similarity > 0.98:
            response["analysis"]["interpretation"] = f"Los documentos son prácticamente idénticos (Similitud: {similarity_percentage:.2f}%)"
        elif similarity > 0.90:
            response["analysis"]["interpretation"] = f"Los documentos son muy similares, con pequeñas diferencias (Similitud: {similarity_percentage:.2f}%)"
        elif similarity > 0.75:
            response["analysis"]["interpretation"] = f"Los documentos tienen diferencias notables (Similitud: {similarity_percentage:.2f}%)"
        else:
            response["analysis"]["interpretation"] = f"Los documentos son significativamente diferentes (Similitud: {similarity_percentage:.2f}%)"
            
        return response
        
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "type": "processing_error"
        }









@app.post("/compare-pdfs-faiss")
async def compare_pdfs_faiss(pdf1: UploadFile, pdf2: UploadFile):
    """Comparación por búsqueda semántica con FAISS"""
    try:
        # Leer archivos PDF
        pdf1_bytes = await pdf1.read()
        pdf2_bytes = await pdf2.read()

        def pdf_to_text(pdf_bytes: bytes, max_chars: int = 8000) -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.flush()
                loader = PyPDFLoader(temp_pdf.name)
                pages = loader.load()
                full_text = ' '.join(page.page_content for page in pages)
                return full_text[:max_chars] + "... (texto truncado)" if len(full_text) > max_chars else full_text

        def split_text(text: str, section_length: int = 500) -> list[str]:  # Reducido para mejor granularidad
            words = text.split()
            sections, current, length = [], [], 0
            for word in words:
                length += len(word) + 1
                if length > section_length:
                    sections.append(' '.join(current))
                    current = [word]
                    length = len(word)
                else:
                    current.append(word)
            if current:
                sections.append(' '.join(current))
            return sections

        # Convertir ambos PDFs a texto y secciones
        text1 = pdf_to_text(pdf1_bytes)
        text2 = pdf_to_text(pdf2_bytes)
        sections1 = split_text(text1)
        sections2 = split_text(text2)

        # Construir índice FAISS del PDF1
        vectorstore = FAISS.from_texts(sections1, OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=1000))

        # Mejorar la búsqueda y presentación de coincidencias
        matches = []
        for i, section in enumerate(sections2):
            result = vectorstore.similarity_search_with_score(section, k=1)
            if result:
                matched_doc, score = result[0]
                similarity_score = 1 - score  # Convertir distancia a similitud
                
                matches.append({
                    "query_section": section,  # Texto completo de la sección
                    "matched_section": matched_doc.page_content,  # Texto completo de la coincidencia
                    "similarity_score": f"{similarity_score:.2%}",
                    "section_number": i + 1
                })

        # Ordenar coincidencias por score de similitud
        matches.sort(key=lambda x: float(x["similarity_score"].rstrip('%')), reverse=True)

        return {
            "match_count": len(matches),
            "sections_analyzed": {
                "total": len(sections2),
                "similar": len([m for m in matches if float(m["similarity_score"].rstrip('%')) >= 90]),
                "different": len([m for m in matches if float(m["similarity_score"].rstrip('%')) < 90])
            },
            "similar_sections": [
                {
                    "section_number": match["section_number"],
                    "text_original": match["matched_section"],
                    "text_modified": match["query_section"],
                    "similarity": match["similarity_score"]
                }
                for match in matches
                if float(match["similarity_score"].rstrip('%')) >= 90  # Mostrar secciones similares
            ],
            "differences_found": [
                {
                    "section_number": match["section_number"],
                    "text_original": match["matched_section"],
                    "text_modified": match["query_section"],
                    "similarity": match["similarity_score"]
                }
                for match in matches
                if float(match["similarity_score"].rstrip('%')) < 90  # Mostrar diferencias significativas
            ],
            "summary": {
                "total_differences": len([m for m in matches if float(m["similarity_score"].rstrip('%')) < 90]),
                "total_similarities": len([m for m in matches if float(m["similarity_score"].rstrip('%')) >= 90]),
                "average_similarity": f"{sum(float(m['similarity_score'].rstrip('%')) for m in matches) / len(matches):.2f}%"
            }
        }

    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "type": "faiss_comparison_error"
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
