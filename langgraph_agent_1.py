from langgraph.graph import StateGraph, MessagesState, START, END

class LanggraphAgent1:

    def __init__(self):
        pass

    def mock_llm(state: MessagesState):
        return {"messages": [{"role": "ai", "content": "hello world"}]}

    def create_graph(self):
        graph = StateGraph(MessagesState)
        graph.add_node(self.mock_llm)
        graph.add_edge(START, "mock_llm")
        graph.add_edge("mock_llm", END)
        graph = graph.compile()
        return graph

