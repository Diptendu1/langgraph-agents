from flask import Flask
from agent1 import Agent1
from langgrah_agent_2 import LanggraphAgent2
app = Flask(__name__)

@app.route("/v1/agent1/result", methods=["GET"])
def index():
    result = Agent1().send_agent().invoke(
            {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
        )
    return result['messages'][3].content

@app.route("/v1/langgraph/agent/2/result", methods=["GET"])
def langgraph_ag_2():
    messages = LanggraphAgent2().invoke()
    return messages['messages'][3].content


app.run(host="0.0.0.0", port=5000)