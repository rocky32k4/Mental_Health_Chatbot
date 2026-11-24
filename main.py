"""
main.py
-------
Entry point for the Mental Health Chatbot.
Run this from the project root folder.

Run:
    python main.py
"""

import sys
import os

# Add src/ to path so all imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.vectorstore import build_vectorstore, get_status
from src.ingestion   import run_ingestion
from src.rag_chain   import run_terminal_chat


def check_setup() -> bool:
    """
    Check if the vector store is ready.
    If not, build it automatically.
    """
    print("Checking setup...")
    status = get_status()

    if status["status"] == "not_built":
        print("Vector store not found. Running ingestion + build...")
        run_ingestion()
        build_vectorstore()

    elif status["status"] == "empty":
        print("Vector store is empty. Rebuilding...")
        build_vectorstore(force_rebuild=True)

    elif status["status"] == "ready":
        print(f"Vector store ready — {status['count']} embeddings loaded.\n")

    else:
        print(f"Vector store error: {status.get('error')}")
        return False

    return True


if __name__ == "__main__":
    if check_setup():
        run_terminal_chat()
