from fastapi import FastAPI, File, UploadFile, Form
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import numpy as np
from io import BytesIO
import tempfile
import base64

from langserve import add_routes
from langchain_community.chat_models import ChatOpenAI  # <-- actualizado
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

# Load environment variables (like your OpenAI API key)
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Verificar la API key al inicio
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")

# Define the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API for calendar availability using LangChain agent"
)

# Define the tool
@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder: in a real app, this would connect to an actual calendar system
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

@tool
def classify_legal_topic(topic: str) -> str:
    """Classify a legal topic and determine which type of lawyer is needed."""
    topic = topic.lower()
    
    # Criminal law keywords
    criminal_keywords = ["robo", "hurto", "asalto", "homicidio", "delito", "penal", "crimen", "violencia", "drogas", "estafa"]
    
    # Civil law keywords
    civil_keywords = ["divorcio", "herencia", "alquiler", "propiedad", "familia", "daños", "civil", "contratos personales", "negligencia"]
    
    # Commercial law keywords
    commercial_keywords = ["empresa", "comercial", "sociedad", "marcas", "patentes", "mercantil", "corporativo", "contratos comerciales", "competencia"]
    
    # Political law keywords
    political_keywords = ["constitucional", "electoral", "partido político", "campaña", "elecciones", "reforma política", "legislación", "derecho parlamentario", "político", "gobierno"]
    
    # Check which category has more matching keywords
    criminal_matches = sum(1 for keyword in criminal_keywords if keyword in topic)
    civil_matches = sum(1 for keyword in civil_keywords if keyword in topic)
    commercial_matches = sum(1 for keyword in commercial_keywords if keyword in topic)
    political_matches = sum(1 for keyword in political_keywords if keyword in topic)
    
    # Find the maximum matches
    max_matches = max(criminal_matches, civil_matches, commercial_matches, political_matches)
    
    if max_matches == 0:
        return "Se necesita más información para clasificar el caso correctamente"
    
    if criminal_matches == max_matches:
        return "Este caso requiere un abogado penal"
    elif civil_matches == max_matches:
        return "Este caso requiere un abogado civil"
    elif commercial_matches == max_matches:
        return "Este caso requiere un abogado comercial"
    elif political_matches == max_matches:
        return "Este caso requiere un abogado político"

@tool
def compare_articles(articles: str) -> str:
    """Compare two articles and determine if they are identical or have differences.
    Returns response in Spanish."""
    
    def extract_articles(text):
        """Helper function to extract articles from input text."""
        article1_start = text.find("Article1:") + len("Article1:")
        article2_start = text.find("Article2:") + len("Article2:")
        
        if article1_start == -1 or article2_start == -1:
            return None, None
            
        if article2_start > article1_start:
            article1 = text[article1_start:text.find("Article2:")].strip()
            article2 = text[article2_start:].strip()
        else:
            article2 = text[article2_start:text.find("Article1:")].strip()
            article1 = text[article1_start:].strip()
            
        return article1, article2

    def find_char_differences(text1, text2):
        """Helper function to find character-level differences."""
        char_diff = []
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                char_diff.append(f"Posición {i+1}: '{c1}' vs '{c2}'")
        return char_diff

    def find_line_differences(text1, text2):
        """Helper function to find line-level differences."""
        from difflib import Differ
        differ = Differ()
        diff = list(differ.compare(text1.splitlines(), text2.splitlines()))
        
        differences = []
        for line in diff:
            if line.startswith('- '):
                differences.append(f"Eliminado: {line[2:]}")
            elif line.startswith('+ '):
                differences.append(f"Agregado: {line[2:]}")
        return differences

    # Extract articles
    article1, article2 = extract_articles(articles)
    if article1 is None:
        return "Error: Por favor proporcione dos artículos marcados como 'Articulo1:' y 'Articulo2:'"
    
    # Check if articles are identical
    if article1 == article2:
        return "Los artículos son idénticos."
    
    # Find line-level differences
    line_differences = find_line_differences(article1, article2)
    if line_differences:
        return "Diferencias encontradas:\n" + "\n".join(line_differences)
    
    # If no line differences, check character-level differences
    char_differences = find_char_differences(article1, article2)
    if char_differences:
        return "Diferencias encontradas:\n" + "\n".join(char_differences)
    
    return "Los artículos tienen diferencias en espacios o puntuación que no son visibles"


@tool
def compare_pdfs(input: str, pdf1_base64: str = None, pdf2_base64: str = None) -> str:
    """Compara dos PDFs y determina su similitud."""
    if not pdf1_base64 or not pdf2_base64:
        return "Para comparar los PDFs, necesito que proporciones ambos documentos en formato base64."
    
    try:
        def pdf_to_text(pdf_base64: str, max_chars: int = 8000) -> str:
            """Convert base64 PDF to text with length limit."""
            pdf_bytes = base64.b64decode(pdf_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.flush()
                
                loader = PyPDFLoader(temp_pdf.name)
                pages = loader.load()
                
                # Concatenar el texto de todas las páginas y limitar la longitud
                full_text = ' '.join(page.page_content for page in pages)
                if len(full_text) > max_chars:
                    return full_text[:max_chars] + "... (texto truncado)"
                return full_text

        # Convertir PDFs a texto con límite de caracteres
        text1 = pdf_to_text(pdf1_base64)
        text2 = pdf_to_text(pdf2_base64)
        
        # Obtener embeddings de manera más eficiente
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Modelo más ligero
            chunk_size=1000  # Procesar en chunks más pequeños
        )
        
        # Calcular similitud
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)
        
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Retornar resultado resumido
        if similarity > 0.95:
            return f"Los documentos son prácticamente idénticos (Similitud: {similarity:.2%})"
        elif similarity > 0.8:
            return f"Los documentos son muy similares (Similitud: {similarity:.2%})"
        elif similarity > 0.6:
            return f"Los documentos tienen algunas similitudes (Similitud: {similarity:.2%})"
        else:
            return f"Los documentos son diferentes (Similitud: {similarity:.2%})"
            
    except Exception as e:
        return f"Error al procesar los PDFs: {str(e)}"


# 2. Configurar el LLM correctamente
llm = ChatOpenAI(
    model="gpt-4.1",  # Modelo con mayor capacidad de contexto
    temperature=0,
    max_tokens=1000  # Limitar la longitud de las respuestas
)


# 3. Definir las herramientas con un formato más simple
tools = [
    Tool(
        name="CheckCalendarAvailability",
        func=check_calendar_availability,
        description="Verifica la disponibilidad del calendario para un día específico"
    ),
    Tool(
        name="ClassifyLegalTopic",
        func=classify_legal_topic,
        description="Clasifica un tema legal y determina qué tipo de abogado se necesita"
    ),
    Tool(
        name="CompareArticles",
        func=compare_articles,
        description="Compara dos artículos y determina si son idénticos o tienen diferencias"
    ),
    Tool(
        name="ComparePDFs",
        func=compare_pdfs,
        description="Compara dos PDFs y determina su similitud. Requiere pdf1_base64 y pdf2_base64."
    )
]

# 4. Configurar el prompt del agente
prefix = """Eres un asistente legal que ayuda a comparar documentos y clasificar temas legales.
Tienes acceso a las siguientes herramientas:"""

suffix = """Comienza!

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# 5. Inicializar el agente con la nueva configuración
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
    max_execution_time=30  # Limitar el tiempo de ejecución
)

# 6. Configurar la ruta con el nuevo agente
@app.post("/general-agent/invoke")
async def invoke_agent(request: dict):
    try:
        response = await agent_chain.ainvoke(
            {"input": request.get("input", "")},
        )
        return response
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "type": "agent_error"
        }

# Expose the agent as an API route using LangServe
add_routes(
    app,
    agent_chain,
    path="/general-agent"
)

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
