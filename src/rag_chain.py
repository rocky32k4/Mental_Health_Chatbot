"""
rag_chain.py
------------
Phase 4 : RAG Pipeline

LLM : llama-3.3-70b-versatile via Groq (free tier)

Flow every chat message:
  user message
      -> safety.check_input()     crisis / out-of-scope check
      -> retrieve()               find relevant chunks from books
      -> build_prompt()           system prompt + context + message
      -> Groq LLM                 generate response
      -> safety.process()         validate + disclaimer injection
      -> return final response

Setup:
    1. pip install groq python-dotenv
    2. Get free key: https://console.groq.com
    3. Add to .env: GROQ_API_KEY=gsk_your_key_here

Run:
    python main.py
"""

import os
from groq import Groq
from dotenv import load_dotenv

try:
    from src.vectorstore import retrieve
    from src.safety      import check_input, process as safety_process
except ImportError:
    from vectorstore import retrieve
    from safety      import check_input, process as safety_process

load_dotenv()


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

LLM_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS  = 250
TEMPERATURE = 0.7
N_RESULTS   = 3

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a compassionate and supportive mental health assistant.
You help people by providing emotional support, psychoeducation, and evidence-based coping strategies.

Important rules:
- You are NOT a therapist or doctor
- Never diagnose conditions or prescribe medication
- Always encourage professional help when appropriate
- Be warm, empathetic, and non-judgmental
- Use simple, clear language — avoid clinical jargon
- Keep responses concise and focused
- Only use information from the provided context
- If the context is not relevant, say so honestly"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_groq_client() -> Groq:
    """Create Groq client using GROQ_API_KEY from .env"""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "\nGROQ_API_KEY not found in .env file."
            "\nGet your free key at: https://console.groq.com"
            "\nAdd to .env: GROQ_API_KEY=gsk_your_key_here"
        )

    return Groq(api_key=api_key)


def build_prompt(user_message: str, context_chunks: list) -> list:
    """Build message list for the LLM with retrieved context."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["source"].replace("_", " ").title()
        context_text += f"\n[Source {i} — {source}]\n{chunk['text']}\n"

    user_content = f"""Use the following context from psychology books to answer the message.
If the context is not directly relevant, use your general knowledge
but stay within safe, supportive, non-diagnostic boundaries.

CONTEXT FROM BOOKS:
{context_text}

USER MESSAGE:
{user_message}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────
# MAIN CHAT FUNCTION
# ─────────────────────────────────────────────

def chat(user_message: str, chat_history: list = None) -> dict:
    """
    Takes user message, returns safe bot response.

    Args:
        user_message : what the user typed
        chat_history : previous messages for memory
                       [{"role": "user", "content": "..."}, ...]

    Returns:
        {
            "response"        : final response text,
            "sources"         : list of book names used,
            "is_crisis"       : True if crisis response triggered,
            "is_out_of_scope" : True if out of scope,
            "disclaimer_added": True if disclaimer was appended,
        }
    """
    if chat_history is None:
        chat_history = []

    # ── Step 1: Safety check on input ──
    # Handles crisis AND out-of-scope before touching the LLM
    safety_input = check_input(user_message)

    if safety_input["override_response"]:
        return {
            "response"        : safety_input["override_response"],
            "sources"         : [],
            "is_crisis"       : safety_input["is_crisis"],
            "is_out_of_scope" : safety_input["is_out_of_scope"],
            "disclaimer_added": False,
        }

    # ── Step 2: Retrieve relevant chunks ──
    try:
        context_chunks = retrieve(user_message, n_results=N_RESULTS)
    except Exception:
        return {
            "response"        : "I'm having trouble accessing my knowledge base. Please try again.",
            "sources"         : [],
            "is_crisis"       : False,
            "is_out_of_scope" : False,
            "disclaimer_added": False,
        }

    # ── Step 3: Build prompt ──
    messages = build_prompt(user_message, context_chunks)

    # ── Step 4: Add chat history ──
    if chat_history:
        messages = [messages[0]] + chat_history[-12:] + [messages[-1]]

    # ── Step 5: Call Groq LLM ──
    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model       = LLM_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        raw_reply = response.choices[0].message.content.strip()

    except Exception as e:
        error = str(e)

        if "401" in error or "invalid_api_key" in error.lower():
            raw_reply = (
                "ERROR: Invalid Groq API key.\n"
                "Check GROQ_API_KEY in your .env file.\n"
                "Get a new key at: https://console.groq.com"
            )
        elif "429" in error or "rate_limit" in error.lower():
            raw_reply = (
                "ERROR: Rate limit reached.\n"
                "Wait a minute and try again.\n"
                "Free tier: 6,000 tokens/min, 14,400 requests/day."
            )
        elif "model_not_found" in error.lower():
            raw_reply = (
                f"ERROR: Model '{LLM_MODEL}' not found.\n"
                "Try: llama-3.1-8b-instant"
            )
        else:
            raw_reply = f"I encountered an issue: {error}"

    # ── Step 6: Safety check on response ──
    # Validates LLM output + injects disclaimers if needed
    safety_output = safety_process(user_message, raw_reply)

    # ── Step 7: Collect source names ──
    sources = list(set(
        chunk["source"].replace("_", " ").title()
        for chunk in context_chunks
    ))

    return {
        "response"        : safety_output["final_response"],
        "sources"         : sources,
        "is_crisis"       : False,
        "is_out_of_scope" : False,
        "disclaimer_added": safety_output["disclaimer_added"],
    }


# ─────────────────────────────────────────────
# TERMINAL CHAT
# ─────────────────────────────────────────────

def run_terminal_chat():
    """Terminal chat loop for testing."""
    print("=" * 55)
    print("  MENTAL HEALTH CHATBOT")
    print(f"  Model : {LLM_MODEL}")
    print("=" * 55)
    print("Type your message and press Enter.")
    print("Type 'quit' to exit.\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nTake care. Goodbye!")
            break

        print("\nBot: thinking...\n")

        result = chat(user_input, chat_history)

        print(f"Bot: {result['response']}")

        if result["sources"]:
            print(f"\n     [Sources: {', '.join(result['sources'])}]")

        if result["is_crisis"]:
            print("\n     *** CRISIS RESPONSE TRIGGERED ***")

        if result["is_out_of_scope"]:
            print("\n     *** OUT OF SCOPE ***")

        if result["disclaimer_added"]:
            print("\n     *** DISCLAIMER ADDED ***")

        print()

        # Update history
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant",  "content": result["response"]})


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_terminal_chat()
