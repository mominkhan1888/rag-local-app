# Local RAG App (Streamlit + Ollama)

## Overview
This project runs a fully local Retrieval-Augmented Generation (RAG) app on your laptop. Upload PDFs, index them into ChromaDB, and chat with streaming responses from Ollama Llama 3.2 3B. Mobile browsers on the same WiFi can connect to the Streamlit server.

## Prerequisites
- Python 3.10+
- Ollama installed and running
- Git (optional)

## Installation
1. Open a terminal in `rag-local-app/`.
2. Run the setup script:

```bash
bash setup.sh
```

If you prefer manual setup:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
cp .env.example .env
```

## Run The App
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Deploy On Streamlit Cloud
`python-dotenv` is optional in cloud. Streamlit Cloud should use app secrets/environment variables.

Set these in **Streamlit Cloud > App settings > Secrets**:
```toml
LLM_PROVIDER="openrouter"
OPENAI_BASE_URL="https://openrouter.ai/api/v1"
OPENAI_API_KEY="YOUR_OPENROUTER_KEY"
OPENAI_MODEL="openrouter/free"
```

Notes:
- `.env` is optional on Streamlit Cloud.
- After changing dependencies or secrets, use **Reboot app** in Streamlit Cloud.

## Remote LLM Providers (For Free Hosting)
This app defaults to local Ollama. For free hosting platforms, use an internet LLM API instead.

Set these in `.env`:
```
LLM_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_MODEL=openrouter/free
```

Alternative provider example:
```
LLM_PROVIDER=groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_MODEL=llama-3.1-8b-instant
```

## Find Your Laptop's Local IP
- Windows:
  - Run `ipconfig` and look for `IPv4 Address` on your WiFi adapter.
- macOS:
  - Run `ipconfig getifaddr en0` (or `en1` if needed).
- Linux:
  - Run `hostname -I` or `ip addr` and look for your WiFi interface address.

## Mobile Access
From a phone on the same WiFi, open:

```
http://<YOUR_LAPTOP_IP>:8501
```

Example:
```
http://192.168.1.25:8501
```

## Expected Behavior
- Upload a PDF in the sidebar and click **Index PDF**.
- Progress displays while pages are processed and embedded.
- Ask questions in the chat input and receive streaming answers.
- If another user is already receiving a response, you will see your queue position.

## Troubleshooting
- **Ollama not running**: Start Ollama and re-try. Verify `http://localhost:11434` is reachable.
- **Model missing**: Run `ollama pull llama3.2:3b`.
- **Slow indexing**: Large PDFs can take 30-60 seconds; this is expected for 500 pages.
- **Mobile cannot connect**: Ensure the laptop firewall allows port 8501 and both devices are on the same WiFi.
- **ChromaDB errors**: Delete `data/chroma_db/` to reset, then re-index.
