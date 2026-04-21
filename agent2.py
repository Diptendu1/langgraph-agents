import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent
from langchain_core.tools import Tool
from langchain_classic.agents import AgentType
from langchain.chat_models import init_chat_model

class Agent2:

    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = ""

    def chat_without_memory(self):
        llm_agent = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.7)
        # response = llm.invoke("What are the best practices for handling authentication?")
        # print(response.content)
        return llm_agent

    def chat_with_memory(self):
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash")
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        # Chatting examples
        # conversation.predict(input="Hello, I'm building a LangChain app.")
        # conversation.predict(input="What is the best way to use it with Gemini?")
        return conversation

    def tool_usage(self):
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash")
        tools = [
            Tool(
                name="Search",
                func=lambda x: "Result from search",
                description="Useful for answering questions about current events"
            )
        ]
        agent = initialize_agent(
            tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        #Agent run example
        #agent.run("What is the current weather in Tokyo?")
        return agent

    def modern_impl(self):
        model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        #Invoking model
        #response = model.invoke("Why do parrots talk?")
        return model
