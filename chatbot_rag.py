#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vortex AI RAG Chatbot CLI
An interactive command-line interface for querying PDF documents using RAG.
Reuses the core application architecture for consistent processing and logic.
"""

import argparse
import logging
import os
import sys

# Set logging level to warning for clean CLI output unless requested otherwise
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.config import settings
from app.services.rag_service import rag_pipeline

# ANSI Color codes for clean and beautiful terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


def print_banner():
    banner = rf"""{BOLD}{CYAN}
============================================================
      _    __           _             _    ___  
     | |  / /__  ____ _|_|____ __  __| |  / / \ 
     | | / / _ \/ __/ __/ / __ ` / / _  | / /__\ 
     | |/ / (_) / / / /_/ / /_/ / / /_/ |/ /____\ 
     |___/\___//_/  \__/_/\____/  \____//_/     
                Vortex AI RAG Chatbot CLI
============================================================{RESET}
  * Embedding Model: {BLUE}{settings.embedding_model_id}{RESET}
  * LLM Provider:    {BLUE}{settings.llm_provider}{RESET}
  * Advanced RAG:    {BLUE}{"Enabled (Parent Retriever)" if settings.use_parent_retriever else "Disabled"}{RESET}
============================================================
"""
    print(banner)


def print_help():
    print(f"""
{BOLD}Available Commands:{RESET}
  {YELLOW}/help{RESET}   - Show this help message
  {YELLOW}/reset{RESET}  - Clear document context and chat history
  {YELLOW}/exit{RESET}   - Exit the CLI chatbot
  Or type any question to query the loaded document.
""")


def main():
    parser = argparse.ArgumentParser(description="Vortex AI RAG Chatbot CLI")
    parser.add_argument("-f", "--file", type=str, help="Path to the PDF document to process on start")
    args = parser.parse_args()

    print_banner()

    session_id = "cli_session"
    pdf_path = args.file

    # Prompt user for PDF file if not provided as argument
    if not pdf_path:
        print(f"{YELLOW}No PDF file specified.{RESET}")
        while True:
            path_input = input(f"{BOLD}Enter path to PDF document to analyze: {RESET}").strip()
            if not path_input:
                continue
            if path_input.lower() in ("exit", "/exit", "quit", "/quit"):
                print(f"\n{BLUE}Goodbye!{RESET}")
                return
            # Remove enclosing quotes if user drag-and-dropped file into terminal
            path_input = path_input.strip("\"'")
            if not os.path.exists(path_input):
                print(f"{RED}Error: File not found at '{path_input}'. Please try again.{RESET}")
                continue
            if not path_input.lower().endswith(".pdf"):
                print(f"{RED}Error: Only PDF documents are supported.{RESET}")
                continue
            pdf_path = path_input
            break

    # Ingest document
    print(f"\n{BLUE}Ingesting and indexing '{os.path.basename(pdf_path)}'...{RESET}")
    print(f"{BLUE}This might take a moment depending on document size and device availability...{RESET}")
    try:
        rag_pipeline.process_document(pdf_path, session_id=session_id)
        print(f"{GREEN}✓ Successfully indexed document!{RESET}")
    except Exception as e:
        print(f"{RED}Error processing document: {e}{RESET}")
        return

    print_help()

    # Chat loop
    while True:
        try:
            user_input = input(f"\n{BOLD}{GREEN}You > {RESET}").strip()
            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print(f"\n{BLUE}Goodbye!{RESET}")
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() == "/reset":
                print(f"{BLUE}Clearing session context...{RESET}")
                rag_pipeline.reset(session_id=session_id)
                print(f"{GREEN}Session reset. Please enter a new PDF path to continue.{RESET}")
                # Get new PDF
                while True:
                    path_input = input(f"{BOLD}Enter path to PDF document to analyze: {RESET}").strip()
                    if not path_input:
                        continue
                    if path_input.lower() in ("exit", "/exit"):
                        return
                    # Remove enclosing quotes
                    path_input = path_input.strip("\"'")
                    if not os.path.exists(path_input):
                        print(f"{RED}Error: File not found. Try again.{RESET}")
                        continue
                    pdf_path = path_input
                    break
                print(f"\n{BLUE}Ingesting '{os.path.basename(pdf_path)}'...{RESET}")
                rag_pipeline.process_document(pdf_path, session_id=session_id)
                print(f"{GREEN}✓ Successfully indexed document!{RESET}")
                print_help()
                continue

            # Standard QA query
            print(f"{BLUE}Vortex AI is thinking...{RESET}", end="\r")
            answer = rag_pipeline.ask_question(user_input, session_id=session_id)
            # Clear the "thinking..." line and print response
            print(" " * 30, end="\r")
            print(f"{BOLD}{CYAN}Vortex AI >{RESET}\n{answer}")

        except KeyboardInterrupt:
            print(f"\n\n{BLUE}Goodbye!{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")


if __name__ == "__main__":
    main()
