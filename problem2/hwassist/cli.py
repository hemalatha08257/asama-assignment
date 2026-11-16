"""python -m hwassist.cli [--llm qwen2.5:7b]"""
from __future__ import annotations
import argparse
from .assistant import Assistant
from .llm import OllamaClient


def main() -> None:
    ap = argparse.ArgumentParser(description="Local Linux hardware Q&A assistant")
    ap.add_argument("--llm", help="Ollama model name (omit for offline rule-based mode)")
    ap.add_argument("-q", "--question", help="Ask one question and exit")
    a = ap.parse_args()
    llm = OllamaClient(a.llm) if a.llm else None
    if llm and not llm.alive():
        print("Ollama not reachable; falling back to offline mode."); llm = None
    bot = Assistant(llm=llm)
    if a.question:
        print(bot.ask(a.question)); return
    print("Hardware assistant ready. Ctrl-D to exit.")
    try:
        while True:
            q = input("you> ").strip()
            if q:
                print(bot.ask(q))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
