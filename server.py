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

# Initialize the language model (updated import)
llm = ChatOpenAI(model="gpt-4o")

# Wrap the tool inside a LangChain agent
tools = [
    Tool(
        name="CheckCalendarAvailability",
        func=check_calendar_availability,
        description="Check calendar availability for a given day"
    )
]

calendar_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Expose the agent as an API route using LangServe
add_routes(
    app,
    calendar_agent,
    path="/calendar-agent"
)

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
