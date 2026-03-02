from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def test_context_break_demo():
    """Demonstrates context break: string-based vs message-based invocation.

    When system instructions are flattened into a single string with user messages,
    the model cannot distinguish roles. Critical instructions (e.g., output format)
    get lost. Message-based invocation preserves structure so the API applies
    system instructions correctly.
    """
    system = (
        "You are a senior AI architect. CRITICAL: You must respond with ONLY "
        "the word 'RISKS:' followed by a comma-separated list of exactly 3 risks, nothing else."
    )
    user1 = "We are building an AI system for processing medical insurance claims."
    user2 = "What are the main risks in this system?"

    # BAD: Flattened string - context breaks, system instruction often ignored
    flattened = f"{system}\n\n{user1}\n\n{user2}"
    bad_response = llm.invoke(flattened)
    print("=== STRING-BASED (context break) ===")
    print("Input: single flattened string, no role/turn structure")
    print("Response:", bad_response.content[:400])
    print()

    # GOOD: Message-based - preserves role and turn order, system instruction applied
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user1),
        HumanMessage(content=user2),
    ]
    good_response = llm.invoke(messages)
    print("=== MESSAGE-BASED (correct context) ===")
    print("Input: SystemMessage + HumanMessage(s) with proper structure")
    print("Response:", good_response.content[:400])


if __name__ == "__main__":
    test_context_break_demo()

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
