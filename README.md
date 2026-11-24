# CS305 Project

Name: Aadharsh Venkat
Roll Number: 241CS101
Section: S1

This project is a small chat application that uses retrieval augmented generation (RAG) to help in the software development lifecycle through requirements analysis (SRS document) and identifying the apprpriate SDLC. It runs a Streamlit interface and a language model with the RAG documents on a Kaggle GPU session. The local app sends requests to the Kaggle server through an ngrok URL.

The PDFs in `docs/` are indexed by the Kaggle server and used as reference material for RAG. The model is *Qwen/Qwen3-4B-Instruct*.

## Agents

- **Requirements Agent** creates or updates a Software Requirements Specification (SRS) for the scenario. It uses retrieved document passages and marks assumptions where the documents do not support a detail.
- **SDLC Agent** recommends a software development lifecycle based on the latest SRS, and explains the recommendation and alternatives.

- **Requirements Verifier** reviews the latest SRS for gaps, ambiguity, conflicts, unsupported claims, and requirements that are difficult to test. It reports findings and questions for stakeholders.
- **SDLC Verifier** reviews the latest SDLC recommendation for fit, missing steps, risks, and unsupported reasoning.

The SDLC Agent and Requirements Verifier become available after an SRS has been generated. The SDLC Verifier becomes available after an SDLC recommendation has been generated. Each chat is saved as a separate JSON file in `chats/`. The sidebar can also export the latest Requirements Agent response as an SRS PDF.

## Requirements

- Python 3.10+
- A Kaggle account with GPU notebook access and Internet enabled
- An ngrok account and authtoken
- The RAG documents

## Start the model server on Kaggle

1. Create a **private Kaggle Dataset** containing `kaggle_server.py` and the `docs/` directory with the project PDFs. Keep the `docs/` directory next to `kaggle_server.py`.
2. Create a Kaggle notebook, enable a T4 GPU and Internet, and add that dataset as notebook input.
3. In the notebook's **Add-ons → Secrets**, add two secrets:
   - `NGROK_AUTHTOKEN`: your ngrok authtoken
   - `RAG_API_TOKEN`: Set the value as 1
4. Run this in a notebook cell, replacing `<dataset-slug>` with the directory name shown under `/kaggle/input/`:

```python
!pip install -q fastapi uvicorn pyngrok sentence-transformers transformers bitsandbytes pymupdf langchain-text-splitters

import os
import sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

project_dir = Path("/kaggle/input/<dataset-slug>")
sys.path.insert(0, str(project_dir))

secrets = UserSecretsClient()
os.environ["NGROK_AUTHTOKEN"] = secrets.get_secret("NGROK_AUTHTOKEN")
os.environ["RAG_API_TOKEN"] = secrets.get_secret("RAG_API_TOKEN")

from kaggle_server import start_server
start_server()
```

Keep the notebook session running while using the app. When the server starts, it prints a public ngrok URL. Copy that URL for the local configuration.

## Run the app locally

From the project directory, create and activate a virtual environment, then install the local dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create the local environment file:

```sh
cp .env.example .env
```

Edit `.env` and set the values below. Use the ngrok URL printed by Kaggle and the exact same `RAG_API_TOKEN` value that you saved in Kaggle Secrets.

```dotenv
RAG_SERVER_URL=https://your-ngrok-url.ngrok-free.app
RAG_API_TOKEN=your-long-random-secret
RAG_REQUEST_TIMEOUT_SECONDS=900
```

Start Streamlit:

```sh
streamlit run chat_app.py
```

Open the local URL printed by Streamlit in your browser. Start a chat by describing a software scenario. The Kaggle notebook must remain active, and its ngrok URL must remain valid, for generation to work.

## Project files

- `chat_app.py` — Streamlit chat interface and PDF export
- `cs305-llm-rag.py` — local client that sends agent requests to Kaggle
- `kaggle_server.py` — model server, document indexing, retrieval, and agent prompts
- `config.py` — reads local connection settings from `.env`
- `docs/` — PDF reference documents indexed by the Kaggle server
- `chats/` — saved conversations, created when the app runs
