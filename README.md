# 🤖 RAG PDF Chatbot

> Ask questions over **any PDF document** using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-pink)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation** is an AI architecture that:
1. **Indexes** your document into a vector database (semantic search)
2. **Retrieves** the most relevant chunks when you ask a question
3. **Generates** a grounded answer using an LLM — based on your document, not hallucination

This project demonstrates RAG end-to-end with **fully open-source, free tools** — no OpenAI API key needed.

---

## 🏗️ Architecture

```
PDF File
   │
   ▼
PyPDFLoader → Text Chunks (RecursiveCharacterTextSplitter)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Store ←── User Question ──→ Similarity Search
                                              │
                                              ▼
                                    Top 3 Relevant Chunks
                                              │
                                              ▼
                                   Flan-T5 LLM (local)
                                              │
                                              ▼
                                         Answer + Sources
```

---

## ⚙️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| Framework | LangChain 0.2 | Chains retriever + LLM together |
| Embeddings | `all-MiniLM-L6-v2` | Fast, free, 384-dim semantic vectors |
| Vector DB | FAISS (CPU) | Local, no cloud needed, blazing fast |
| LLM | `google/flan-t5-base` | Small open-source model, no API key |
| PDF Loader | PyPDFLoader | Extracts text per page with metadata |
| UI | Gradio | One-file web app, deploys to HF Spaces |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Aaricacoding/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads ~330MB of models (Flan-T5 + MiniLM). Cached after that.

### 3. Run the app
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 📖 How to Use

1. **Upload** any PDF using the file picker
2. Click **"Index PDF"** — wait for the ✅ success message
3. **Type a question** in the text box
4. Click **"Get Answer"** or press Enter
5. Answer appears with source page references

---

## 🌐 Deploy to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set **SDK: Gradio**, **Python 3.10**
3. Push this repo to the Space
4. In `app.py`, change `share=False` → `share=True`
5. HF Spaces will auto-install `requirements.txt` and launch

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py            # Gradio UI — upload, Q&A, event wiring
├── rag_engine.py     # RAG logic — PDF loading, FAISS indexing, QA chain
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🔧 Customization

| What to change | Where | How |
|---|---|---|
| Chunk size | `rag_engine.py` | Adjust `chunk_size` in `RecursiveCharacterTextSplitter` |
| Number of retrieved chunks | `rag_engine.py` | Change `k=3` in `as_retriever()` |
| LLM model | `rag_engine.py` | Replace `google/flan-t5-base` with any HF model |
| Embedding model | `rag_engine.py` | Replace `all-MiniLM-L6-v2` with a larger model |

---

## 🏷️ Skills Demonstrated

- **RAG architecture** end-to-end implementation
- **Vector databases** (FAISS) — indexing and similarity search
- **LangChain** chains and retrievers
- **HuggingFace** — embeddings + local LLMs
- **Gradio** — building and deploying AI web apps
- **Python** — clean modular code with error handling

---

## 👩‍💻 Author

**Aarica Raj** — AI/ML Developer | BCA Student  
🔗 [GitHub](https://github.com/Aaricacoding) · [LinkedIn](https://linkedin.com/in/aarica-raj)

---

## 📄 License

MIT License — free to use, modify, and distribute.
