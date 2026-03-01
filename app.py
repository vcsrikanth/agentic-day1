import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

if __name__ == "__main__":
    messages = [
        SystemMessage(content="You are a senior AI architect reviewing production systems."),
        HumanMessage(content="We are building an AI system for processing medical insurance claims."),
        HumanMessage(content="What are the main risks in this system?"),
    ]
    try:
        response = llm.invoke(messages)
        print("Response:", response.content)
    except Exception as e:
        print(f"Error invoking LLM: {e}", file=sys.stderr)
        sys.exit(1)

"""
Reflection:

1. Why did string-based invocation fail?
   Passing a plain string flattens all context into one blob. The model cannot distinguish
   system instructions from user messages or conversation turns, so role and turn order are lost.

2. Why does message-based invocation work?
   SystemMessage, HumanMessage, and AssistantMessage preserve role and turn order. The API
   expects this structure and can correctly apply system instructions and conversation context.

3. What would break in a production AI system if we ignore message history?
   Multi-turn coherence breaks: follow-ups like "What about X?" become meaningless. Users lose
   continuity, corrections, and refinement. Each request is treated as isolated, causing
   inconsistent and incoherent conversations.
"""
