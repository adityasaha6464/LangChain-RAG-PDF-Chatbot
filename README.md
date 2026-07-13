# 📄 LangChain RAG PDF Chatbot

> **Ask questions about any PDF — get instant, AI-powered answers grounded in the document's content.**

Built with **LangChain**, **ChromaDB**, and **HuggingFace** embeddings, this Retrieval-Augmented Generation (RAG) chatbot ingests a PDF, stores its content as vector embeddings, and uses an LLM to answer your questions with full context from the document.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📑 **PDF Ingestion** | Loads and parses any PDF using `PyPDFLoader` |
| 🔪 **Smart Chunking** | Splits documents into overlapping chunks for optimal retrieval |
| 🧠 **Semantic Embeddings** | Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| 💾 **Persistent Vector Store** | Stores embeddings in ChromaDB — no re-indexing needed |
| ⚡ **Groq LLM (Primary)** | Blazing-fast inference via Groq's `llama-3.3-70b-versatile` |
| 🔁 **Gemini Fallback** | Automatically falls back to Google Gemini if Groq is unavailable |
| 🛡️ **Retry & Error Handling** | Graceful retries with exponential backoff and clear error messages |
| 🖥️ **Interactive CLI** | Simple terminal-based Q&A loop |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  PDF File   │────▶│  loader.py   │────▶│  vector_db.py  │
│ (data/*.pdf)│     │  Parse & Split│     │  Embed & Store │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  User Query │────▶│   chat.py    │◀───▶│   ChromaDB     │
│  (terminal) │     │  RAG + LLM   │     │  (chroma_db/)  │
└─────────────┘     └──────┬───────┘     └────────────────┘
                           │
                    ┌──────▼───────┐
                    │   LLM API    │
                    │ Groq / Gemini│
                    └──────────────┘
```

---

## 📁 Project Structure

```
LangChain-RAG-PDF-Chatbot/
├── data/
│   └── sample.pdf          # Your source PDF document
├── chat.py                 # Main chatbot — interactive Q&A loop
├── loader.py               # PDF loading & text chunking
├── vector_db.py            # Embedding generation & ChromaDB storage
├── embedding.py            # Standalone embedding test utility
├── requirements.txt        # Python dependencies
├── .env.example            # Template for API keys
├── .gitignore              # Git exclusions
└── README.md               # You are here
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- An API key from **[Groq](https://console.groq.com/)** (free tier available) and/or **[Google AI Studio](https://aistudio.google.com/)**

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LangChain-RAG-PDF-Chatbot.git
cd LangChain-RAG-PDF-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```env
GROQ_API_KEY=gsk_your_groq_key_here
GOOGLE_API_KEY=your_google_key_here   # optional fallback
```

### 5. Add Your PDF

Place your PDF file at `data/sample.pdf` (or update the path in `loader.py` and `vector_db.py`).

### 6. Build the Vector Database

```bash
python vector_db.py
```

You should see:

```
Chunks Created: N
Vector DB created successfully
```

### 7. Start Chatting!

```bash
python chat.py
```

```
PDF Chatbot Ready!
Type 'exit' to quit

Ask Question: What is this document about?

AI Answer:
This document covers...

--------------------------------------------------
```

---

## ⚙️ How It Works

### Step 1 — Load & Chunk (`loader.py`)
The PDF is loaded with `PyPDFLoader` and split into 500-character chunks (with 50-char overlap) using `RecursiveCharacterTextSplitter` to preserve context across chunk boundaries.

### Step 2 — Embed & Store (`vector_db.py`)
Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer. These vectors are persisted in a local **ChromaDB** database at `chroma_db/`.

### Step 3 — Retrieve & Answer (`chat.py`)
When you ask a question:
1. Your query is embedded using the same model.
2. The **top 3** most semantically similar chunks are retrieved from ChromaDB.
3. The retrieved context + your question are sent to the LLM.
4. The LLM generates an answer grounded in the document content.

---

## 🔑 LLM Configuration

| Priority | Provider | Model | How to Enable |
|---|---|---|---|
| 1st | **Groq** | `llama-3.3-70b-versatile` | Set `GROQ_API_KEY` in `.env` |
| 2nd | **Google Gemini** | `gemini-2.0-flash` | Set `GOOGLE_API_KEY` in `.env` |

- If **both** keys are set, Groq is used as the primary LLM with Gemini as an automatic fallback.
- If **only** one key is set, that provider is used exclusively.
- If **neither** key is set, a warning is displayed.

---

## 🧪 Testing Individual Components

```bash
# Test PDF loading & chunking
python loader.py

# Test embedding generation
python embedding.py

# Rebuild the vector database
python vector_db.py
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure your virtual environment is activated and run `pip install -r requirements.txt` |
| `GOOGLE_API_KEY` not working | Verify the key at [Google AI Studio](https://aistudio.google.com/) and ensure the Generative Language API is enabled |
| `ConnectionError` on Groq | Check your internet connection; the chatbot will auto-fallback to Gemini if configured |
| `UnicodeEncodeError` on Windows | Run `chcp 65001` in your terminal before launching, or set `PYTHONUTF8=1` |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using <a href="https://python.langchain.com/">LangChain</a> · <a href="https://www.trychroma.com/">ChromaDB</a> · <a href="https://huggingface.co/">HuggingFace</a>
</p>
