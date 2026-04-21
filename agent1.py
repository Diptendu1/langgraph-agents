from langchain.tools import tool
from langchain.agents import create_agent
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyChOUEIJfJ3CdoK4xxrRW_ecNi6ZcZlctI"
class Agent1:

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 72°F"

    def send_agent(self):
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=1.0,  # Gemini 3.0+ defaults to 1.0
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
        agent = create_agent(model, tools=[self.search, self.get_weather])
        return agent
