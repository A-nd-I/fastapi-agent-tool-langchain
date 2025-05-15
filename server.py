from fastapi import FastAPI
from langserve import add_routes
from langchain_community.chat_models import ChatOpenAI  # <-- actualizado
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.tools import tool
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables (like your OpenAI API key)
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

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

# Initialize the language model (updated import)
llm = ChatOpenAI(model="gpt-4o")

# Wrap the tool inside a LangChain agent
tools = [
    Tool(
        name="CheckCalendarAvailability",
        func=check_calendar_availability,
        description="Check calendar availability for a given day"
    ),
    Tool(
        name="ClassifyLegalTopic",
        func=classify_legal_topic,
        description="Classify a legal topic and determine which type of lawyer is needed"
    ),
    Tool(
        name="CompareArticles",
        func=compare_articles,
        description="Compare two articles and determine if they are identical or have differences"
    )
]

general_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Expose the agent as an API route using LangServe
add_routes(
    app,
    general_agent,
    path="/general-agent"
)

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
