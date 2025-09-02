from fastapi import FastAPI, UploadFile, Query, File, Form
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import OpenAIEmbeddings
import numpy as np
import tempfile

import random
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import difflib

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

def pdf_to_text(pdf_bytes: bytes, max_chars: int = 8000) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush()
        loader = PyPDFLoader(temp_pdf.name)
        pages = loader.load()
        full_text = ' '.join(page.page_content for page in pages)
        return full_text[:max_chars] + "... (texto truncado)" if len(full_text) > max_chars else full_text

def split_text_into_sections(text: str, section_length: int = 1000) -> list:
    words = text.split()
    sections = []
    current_section = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > section_length:
            sections.append(' '.join(current_section))
            current_section = [word]
            current_length = len(word)
        else:
            current_section.append(word)
    if current_section:
        sections.append(' '.join(current_section))
    return sections

def analyze_difference_with_llm(llm, texto_a: str, texto_b: str) -> str:
    prompt = f"""
Eres un abogado experto en análisis de contratos. Compara las siguientes cláusulas de dos contratos distintos:

Contrato A:
\"\"\"{texto_a}\"\"\"

Contrato B:
\"\"\"{texto_b}\"\"\"

Indica si estas cláusulas tienen el mismo propósito e implicaciones legales. Si hay diferencias, explica qué cambia en términos de responsabilidades, obligaciones, derechos u otros efectos jurídicos. Sé claro y conciso.
"""
    return llm.invoke(prompt).content

@app.post("/compare-pdfs")
async def compare_pdfs_endpoint(pdf1: UploadFile, pdf2: UploadFile):
    """Endpoint para comparar dos PDFs usando embeddings"""
    try:
        pdf1_content = await pdf1.read()
        pdf2_content = await pdf2.read()
        text1 = pdf_to_text(pdf1_content)
        text2 = pdf_to_text(pdf2_content)
        sections1 = split_text_into_sections(text1)
        sections2 = split_text_into_sections(text2)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=1000)
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_percentage = similarity * 100

        section_similarities = []
        for i, sec1 in enumerate(sections1):
            sec1_embedding = embeddings.embed_query(sec1)
            best_match = {"similarity": 0, "section": "", "index": -1}
            for j, sec2 in enumerate(sections2):
                sec2_embedding = embeddings.embed_query(sec2)
                sec_similarity = np.dot(sec1_embedding, sec2_embedding) / (np.linalg.norm(sec1_embedding) * np.linalg.norm(sec2_embedding))
                if sec_similarity > best_match["similarity"]:
                    best_match = {
                        "similarity": sec_similarity,
                        "section": sec2[:200] + "...",
                        "index": j
                    }
            section_similarities.append({
                "section1": sec1[:200] + "...",
                "best_match": best_match,
                "similarity_score": best_match["similarity"]
            })
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
        return {"error": True, "message": str(e), "type": "processing_error"}


@app.post("/compare-pdfs-diff")
async def compare_pdfs_diff(
    pdf1: UploadFile, 
    pdf2: UploadFile, 
    explain_with_llm: bool = True
):
    """
    Comparación palabra a palabra y por línea usando difflib para encontrar diferencias exactas.
    """
    try:
        print("Starting PDF comparison...")
        pdf1_bytes = await pdf1.read()
        pdf2_bytes = await pdf2.read()
        print(f"PDF files read. PDF1 size: {len(pdf1_bytes)} bytes, PDF2 size: {len(pdf2_bytes)} bytes")
        
        text1 = pdf_to_text(pdf1_bytes, max_chars=30000)
        text2 = pdf_to_text(pdf2_bytes, max_chars=30000)
        print(f"Converted PDFs to text. Text1 length: {len(text1)}, Text2 length: {len(text2)}")

        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        print(f"Split texts into lines. Lines in doc1: {len(lines1)}, Lines in doc2: {len(lines2)}")
        
        diff = list(difflib.unified_diff(lines1, lines2, fromfile='Documento A', tofile='Documento B', lineterm=''))
        print(f"Generated diff with {len(diff)} differences")

        sm = difflib.SequenceMatcher(None, text1, text2)
        similarity = sm.ratio()
        similarity_percentage = similarity * 100
        print(f"Calculated similarity: {similarity_percentage:.2f}%")

        changes = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                changes.append({
                    "change_type": tag,
                    "original_text": text1[i1:i2],
                    "modified_text": text2[j1:j2],
                    "original_index": i1,
                    "modified_index": j1
                })
        print(f"Found {len(changes)} changes between documents")

        summary = {
            "total_changes": len(changes),
            "interpretation": "Los documentos son prácticamente idénticos" if similarity > 0.98 else (
                "Los documentos son muy similares, con pequeños cambios" if similarity > 0.90 else (
                    "Los documentos tienen diferencias notables" if similarity > 0.75 else
                    "Los documentos son significativamente diferentes"
                )
            )
        }
        print(f"Generated summary: {summary}")

        if explain_with_llm:
            print("Generating LLM explanation...")
            # Genera una explicación amigable con LLM
            prompt = f"""Eres un abogado experto y tu tarea es analizar y explicar las diferencias entre dos documentos legales. No seas técnico, explica de manera comprensible para un cliente, de forma clara y estructurada.

            Documento A:
            \"\"\"{text1[:6000]}\"\"\"  # Puedes truncar para evitar mensajes muy largos

            Documento B:
            \"\"\"{text2[:6000]}\"\"\"

            Lista de diferencias detectadas:
            {[c for c in changes]}

            Por favor, explica los cambios más relevantes en lenguaje simple, indicando cómo afectan el significado legal. Hazlo punto por punto.
            """
            print("Generated prompt for LLM, length:", len(prompt))
            try:
                explanation = llm.invoke(prompt).content
                print("LLM explanation received, length:", len(explanation))
                if not explanation:
                    explanation = "⚠️ El modelo no devolvió contenido."
                    print("Warning: LLM returned empty content")
            except Exception as e:
                print(f"Error al invocar LLM: {str(e)}")
                explanation = "⚠️ No se pudo generar una explicación por un error del modelo."
            return {
                "explanation": explanation,
                "summary": summary
            }
        else:
            print("Returning technical diff results...")
            # Retorna el diff técnico como antes
            return {
                "diff_lines": diff,
                "similarity_ratio": f"{similarity_percentage:.2f}%",
                "changes": changes,
                "summary": summary
            }
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return {"error": True, "message": str(e), "type": "diff_comparison_error"}

from typing import List


@app.post("/ask-pdfs")
async def ask_pdfs(
    pdfs: List[UploadFile] = File(...),
    question: str = Form(...)
):
    combined_text = ""
    for pdf in pdfs:
        pdf_bytes = await pdf.read()
        combined_text += pdf_to_text(pdf_bytes) + "\n\n"

    prompt = f"""
        Eres un asistente legal que responde preguntas sobre documentos. Aquí tienes el contenido combinado de los documentos:

        \"\"\"{combined_text}\"\"\"

        Pregunta: \"{question}\"

        Por favor, proporciona una respuesta clara y concisa basada en los documentos.
        """
    answer = llm.invoke(prompt).content
    return {"answer": answer}

# Modelo para la investigación legal
class LegalResearchRequest(BaseModel):
    question: str
    research_type: str = "comprehensive_legal_research"
    format: str = "article"  # "article" o "references"

@app.post("/legal-research")
async def legal_research_endpoint(request: LegalResearchRequest):
    """
    Endpoint para investigación jurídica integral colombiana con análisis 
    histórico, filosófico y jurisprudencial
    """
    try:
        print(f"Starting legal research for question: {request.question[:100]}...")
        
        # Detectar si la consulta es específica de Colombia o de otro país
        country_detection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)
        
        detection_prompt = f"""
        Analiza la siguiente consulta jurídica y determina si es específica de Colombia o de otro país:
        
        Consulta: {request.question}
        
        Responde únicamente con:
        - "COLOMBIA" si la consulta es sobre derecho colombiano
        - "INTERNACIONAL: [nombre del país]" si es sobre otro país específico
        - "GENERAL" si es sobre derecho en general sin país específico
        
        Solo responde con una de estas opciones, nada más.
        """
        
        country_context = country_detection_llm.invoke(detection_prompt).content.strip()
        print(f"Country detection result: {country_context}")
        
        # Seleccionar prompt según formato solicitado
        if request.format == "references":
            # PROMPT PARA FORMATO DE REFERENCIAS TÉCNICAS
            comprehensive_prompt = f"""
Eres un jurista experto especializado en investigación técnica y referencias jurídicas. Responde con un esquema técnico formal y estructurado.

CONSULTA: {request.question}
JURISDICCIÓN: {country_context}

INSTRUCCIONES: Proporciona un esquema técnico formal con referencias precisas, organizadas de la siguiente manera:

I. MARCO NORMATIVO VIGENTE
• Constitución Política: [Artículos específicos aplicables]
• Leyes principales: [Nombre completo, número, fecha, artículos relevantes]
• Decretos reglamentarios: [Números, fechas, artículos específicos]
• Resoluciones aplicables: [Entidad emisora, número, fecha]

II. JURISPRUDENCIA RELEVANTE
• Corte Constitucional: [Número de sentencia, fecha, magistrado ponente, ratio decidendi]
• Corte Suprema de Justicia: [Sala, sentencia, fecha, doctrina probable]
• Consejo de Estado: [Sala, sentencia, fecha, precedente administrativo]
• Tribunales internacionales: [Corte, caso, fecha, decisión relevante]

III. TRATADOS Y NORMATIVIDAD INTERNACIONAL
• Tratados ratificados: [Nombre, fecha ratificación, artículos aplicables]
• Convenios internacionales: [Organización, convenio, vigencia]
• Soft law: [Declaraciones, principios, recomendaciones]

IV. ANTECEDENTES HISTÓRICOS
• Fecha de origen: [Año específico, contexto histórico]
• Reformas principales: [Años, cambios significativos]
• Casos paradigmáticos: [Nombres, fechas, impacto jurídico]

V. ORGANISMOS COMPETENTES
• Entidades de control: [Nombres oficiales, funciones específicas]
• Autoridades jurisdiccionales: [Competencia, procedimientos]
• Organismos internacionales: [Mandato, supervisión]

VI. PROCEDIMIENTOS TÉCNICOS
• Requisitos formales: [Lista específica, normatividad aplicable]
• Plazos legales: [Términos exactos, consecuencias]
• Recursos disponibles: [Tipos, términos, instancias]

VII. ESTADÍSTICAS Y DATOS
• Cifras oficiales: [Fuente, año, datos relevantes]
• Tendencias: [Períodos, variaciones, análisis cuantitativo]

FORMATO REQUERIDO:
- Usa viñetas y numeración clara
- Cita exacta de normatividad con números de artículos
- Referencias jurisprudenciales precisas con radicados cuando los conozcas
- Fechas específicas y datos verificables
- Lenguaje técnico-jurídico formal
- Estructura esquemática clara y organizada

IMPORTANTE: Si desconoces información específica, indica "Por verificar" en lugar de especular."""
        else:
            # PROMPT PARA FORMATO ARTÍCULO (como estaba antes)
            comprehensive_prompt = f"""
Eres un jurista experto con décadas de experiencia en derecho internacional, derecho comparado y legislación colombiana. Responde de manera exhaustiva y profesional.

CONSULTA: {request.question}
JURISDICCIÓN DETECTADA: {country_context}

INSTRUCCIONES PARA TU RESPUESTA:

Escribe una investigación jurídica completa en párrafos fluidos y cohesivos (SIN títulos, SIN secciones, SIN numeraciones). Tu respuesta debe incluir naturalmente:

1. Comienza respondiendo directamente la consulta con una explicación clara del concepto o institución jurídica consultada, su relevancia actual y aplicabilidad práctica.

2. Desarrolla la normatividad específica aplicable citando con precisión: artículos constitucionales exactos, leyes vigentes con números y fechas, decretos reglamentarios, jurisprudencia vinculante con números de sentencia y fechas exactas, tratados internacionales ratificados.

3. Presenta el contexto histórico completo explicando los antecedentes que originaron la institución/norma, factores sociales, políticos o económicos que motivaron su creación, evolución cronológica con fechas clave, casos paradigmáticos que marcaron precedentes.

4. Proporciona ejemplos concretos y casos reales: casos judiciales relevantes con nombres y fechas, ejemplos de aplicación práctica, procedimientos específicos, consecuencias jurídicas para las personas.

5. Analiza el tema desde las tres corrientes filosóficas del derecho: desde la perspectiva iusnaturalista (derechos inherentes, principios universales, dignidad humana), desde el enfoque iuspositivista (validez formal, jerarquía normativa, legalidad), desde el análisis iusrealista (aplicación práctica real, factores sociales, efectividad real).

6. Evalúa críticamente la efectividad real de la normatividad, problemas de implementación, beneficios tangibles, aspectos que requieren reforma.

7. Identifica el panorama actual: reformas en curso, debates jurídicos actuales, tendencias jurisprudenciales recientes, proyectos de ley, posicionamiento en estándares internacionales.

8. Destaca la normatividad favorable: leyes que protegen o favorecen el tema, incentivos y beneficios disponibles, programas gubernamentales de apoyo, mecanismos de defensa constitucional.

FORMATO REQUERIDO:
- SOLO párrafos completos y fluidos
- NO uses títulos, subtítulos, numeraciones o secciones
- Transiciones naturales entre ideas
- Lenguaje técnico-jurídico pero comprensible
- Citas exactas de normatividad y jurisprudencia
- Fechas precisas y datos verificables

IMPORTANTE: Si desconoces información específica (fechas, números de sentencias), indícalo claramente en lugar de especular. La precisión es fundamental.

Procede con la investigación de manera exhaustiva integrando todos estos elementos en un texto cohesivo y profesional."""
        
        print("Generated comprehensive legal research prompt")
        
        # Crear un LLM específico para investigación con más tokens
        research_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=4000)
        
        # Generar ambos formatos para evitar regeneraciones
        print("Generating both article and references formats...")
        
        # Generar formato artículo
        article_prompt = comprehensive_prompt if request.format == "article" else f"""
Eres un jurista experto con décadas de experiencia en derecho internacional, derecho comparado y legislación colombiana. Responde de manera exhaustiva y profesional.

CONSULTA: {request.question}
JURISDICCIÓN DETECTADA: {country_context}

INSTRUCCIONES PARA TU RESPUESTA:

Escribe una investigación jurídica completa en párrafos fluidos y cohesivos (SIN títulos, SIN secciones, SIN numeraciones). Tu respuesta debe incluir naturalmente:

1. Comienza respondiendo directamente la consulta con una explicación clara del concepto o institución jurídica consultada, su relevancia actual y aplicabilidad práctica.

2. Desarrolla la normatividad específica aplicable citando con precisión: artículos constitucionales exactos, leyes vigentes con números y fechas, decretos reglamentarios, jurisprudencia vinculante con números de sentencia y fechas exactas, tratados internacionales ratificados.

3. Presenta el contexto histórico completo explicando los antecedentes que originaron la institución/norma, factores sociales, políticos o económicos que motivaron su creación, evolución cronológica con fechas clave, casos paradigmáticos que marcaron precedentes.

4. Proporciona ejemplos concretos y casos reales: casos judiciales relevantes con nombres y fechas, ejemplos de aplicación práctica, procedimientos específicos, consecuencias jurídicas para las personas.

5. Analiza el tema desde las tres corrientes filosóficas del derecho: desde la perspectiva iusnaturalista (derechos inherentes, principios universales, dignidad humana), desde el enfoque iuspositivista (validez formal, jerarquía normativa, legalidad), desde el análisis iusrealista (aplicación práctica real, factores sociales, efectividad real).

6. Evalúa críticamente la efectividad real de la normatividad, problemas de implementación, beneficios tangibles, aspectos que requieren reforma.

7. Identifica el panorama actual: reformas en curso, debates jurídicos actuales, tendencias jurisprudenciales recientes, proyectos de ley, posicionamiento en estándares internacionales.

8. Destaca la normatividad favorable: leyes que protegen o favorecen el tema, incentivos y beneficios disponibles, programas gubernamentales de apoyo, mecanismos de defensa constitucional.

FORMATO REQUERIDO:
- SOLO párrafos completos y fluidos
- NO uses títulos, subtítulos, numeraciones o secciones
- Transiciones naturales entre ideas
- Lenguaje técnico-jurídico pero comprensible
- Citas exactas de normatividad y jurisprudencia
- Fechas precisas y datos verificables

IMPORTANTE: Si desconoces información específica (fechas, números de sentencias), indícalo claramente en lugar de especular. La precisión es fundamental.

Procede con la investigación de manera exhaustiva integrando todos estos elementos en un texto cohesivo y profesional."""

        # Generar formato referencias  
        references_prompt = comprehensive_prompt if request.format == "references" else f"""
Eres un jurista experto especializado en investigación técnica y referencias jurídicas. Responde con un esquema técnico formal y estructurado.

CONSULTA: {request.question}
JURISDICCIÓN: {country_context}

INSTRUCCIONES: Proporciona un esquema técnico formal con referencias precisas, organizadas de la siguiente manera:

I. MARCO NORMATIVO VIGENTE
• Constitución Política: [Artículos específicos aplicables]
• Leyes principales: [Nombre completo, número, fecha, artículos relevantes]
• Decretos reglamentarios: [Números, fechas, artículos específicos]
• Resoluciones aplicables: [Entidad emisora, número, fecha]

II. JURISPRUDENCIA RELEVANTE
• Corte Constitucional: [Número de sentencia, fecha, magistrado ponente, ratio decidendi]
• Corte Suprema de Justicia: [Sala, sentencia, fecha, doctrina probable]
• Consejo de Estado: [Sala, sentencia, fecha, precedente administrativo]
• Tribunales internacionales: [Corte, caso, fecha, decisión relevante]

III. TRATADOS Y NORMATIVIDAD INTERNACIONAL
• Tratados ratificados: [Nombre, fecha ratificación, artículos aplicables]
• Convenios internacionales: [Organización, convenio, vigencia]
• Soft law: [Declaraciones, principios, recomendaciones]

IV. ANTECEDENTES HISTÓRICOS
• Fecha de origen: [Año específico, contexto histórico]
• Reformas principales: [Años, cambios significativos]
• Casos paradigmáticos: [Nombres, fechas, impacto jurídico]

V. ORGANISMOS COMPETENTES
• Entidades de control: [Nombres oficiales, funciones específicas]
• Autoridades jurisdiccionales: [Competencia, procedimientos]
• Organismos internacionales: [Mandato, supervisión]

VI. PROCEDIMIENTOS TÉCNICOS
• Requisitos formales: [Lista específica, normatividad aplicable]
• Plazos legales: [Términos exactos, consecuencias]
• Recursos disponibles: [Tipos, términos, instancias]

VII. ESTADÍSTICAS Y DATOS
• Cifras oficiales: [Fuente, año, datos relevantes]
• Tendencias: [Períodos, variaciones, análisis cuantitativo]

FORMATO REQUERIDO:
- Usa viñetas y numeración clara
- Cita exacta de normatividad con números de artículos
- Referencias jurisprudenciales precisas con radicados cuando los conozcas
- Fechas específicas y datos verificables
- Lenguaje técnico-jurídico formal
- Estructura esquemática clara y organizada

IMPORTANTE: Si desconoces información específica, indica "Por verificar" en lugar de especular."""

        # Generar ambos formatos
        article_result = research_llm.invoke(article_prompt).content
        references_result = research_llm.invoke(references_prompt).content
        
        print(f"Both formats generated - Article: {len(article_result)} chars, References: {len(references_result)} chars")
        
        if not article_result or not references_result:
            print("Warning: One of the formats returned empty content")
            return {
                "error": True, 
                "message": "No se pudo generar la investigación jurídica en ambos formatos"
            }
        
        # Generar fuentes jurídicas simuladas (esto debería conectarse con bases de datos reales)
        legal_sources = [
            "Constitución Política de Colombia 1991",
            "Corte Constitucional de Colombia - Jurisprudencia",
            "Corte Suprema de Justicia - Sala Civil y Penal",
            "Consejo de Estado - Sala de lo Contencioso Administrativo",
            "Código Civil Colombiano",
            "Código de Procedimiento Civil",
            "Tratados Internacionales ratificados por Colombia"
        ]
        
        methodology = """
        Metodología de Investigación Aplicada:
        1. Análisis hermenéutico-jurídico de fuentes primarias
        2. Revisión jurisprudencial sistemática
        3. Estudio comparativo de corrientes filosóficas del derecho
        4. Contextualización histórico-normativa
        5. Síntesis doctrinal contemporánea
        """
        
        return {
            "article_format": article_result,
            "references_format": references_result,
            "legal_sources": legal_sources,
            "methodology": methodology,
            "research_type": request.research_type,
            "status": "completed",
            "requested_format": request.format
        }
        
    except Exception as e:
        print(f"Error during legal research: {str(e)}")
        return {
            "error": True, 
            "message": f"Error en la investigación jurídica: {str(e)}"
        }

# LLM config (igual)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens=1000)
tools = [Tool(name="ComparePDFs", func=compare_pdfs_endpoint, description="Compara dos PDFs y determina su similitud usando embeddings")]
prefix = """Eres un asistente legal especializado en comparación de documentos."""
suffix = """Comienza!

Question: {input}
{agent_scratchpad}"""
from langchain.agents import ZeroShotAgent
prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=prompt), tools=tools, verbose=True)
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

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
