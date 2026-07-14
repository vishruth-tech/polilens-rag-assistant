# 🔍 PoliLens — AI Insurance Knowledge Assistant

An AI-powered knowledge worker that answers any question about your insurance company — products, contracts, employees, and operations — using Retrieval Augmented Generation (RAG). Built with LangChain, ChromaDB, HuggingFace embeddings, and Groq inference.

---

## 🚀 Overview

Insurance companies deal with massive amounts of internal documentation — product specs, contract terms, employee handbooks, claims procedures. Finding the right information is slow and frustrating.

PoliLens solves this by turning your company's documents into a conversational AI assistant. Ask any question in plain English and get accurate, grounded answers with source attribution — powered by RAG, not hallucination.

---

## 🧠 How It Works

```
User asks a question
        ↓
Question embedded into a vector using HuggingFace all-MiniLM-L6-v2
        ↓
ChromaDB performs semantic similarity search
→ retrieves top 10 most relevant document chunks
        ↓
Retrieved chunks injected into system prompt as context
        ↓
Groq LLM generates a grounded answer using that context
        ↓
Answer + source documents displayed in two-panel Gradio UI
```

---

## 📚 Knowledge Base Structure

```
knowledge-base/
├── company/          → Company overview, mission, history
├── contracts/        → Contract templates and terms
├── employees/        → Employee profiles and departments
└── products/
    ├── Bizllm.md     → Business insurance product
    ├── Carllm.md     → Car insurance product
    ├── Claimllm.md   → Claims management product
    ├── Healthllm.md  → Health insurance product
    ├── Homellm.md    → Home insurance product
    ├── Lifellm.md    → Life insurance product
    ├── Markellm.md   → Marketing platform product
    └── Rellm.md      → Reinsurance product
```

Knowledge base uses synthetic data generated for demonstration purposes.

---

## ⚙️ Key Engineering Decisions

**HuggingFace embeddings (local)** — Uses `all-MiniLM-L6-v2` to convert text into vectors locally without any API cost. Fast, free, and accurate for semantic search on structured company documents.

**RecursiveCharacterTextSplitter** — Chunks documents at 500 characters with 200 character overlap. The overlap ensures context spanning chunk boundaries is never lost during retrieval.

**ChromaDB as vector store** — Persists embeddings to disk so ingestion only runs once. On subsequent runs the vector store loads instantly without re-embedding all documents.

**Top-K retrieval (K=10)** — Retrieves the 10 most semantically similar chunks per query for maximum context coverage.

**Combined question strategy** — Before retrieval, all previous user messages are combined with the current question. This gives the retriever full conversation context, not just the latest message — dramatically improves accuracy for follow-up questions.

**Source attribution** — Every answer shows exactly which documents were retrieved, giving users full transparency into where the information came from.

**Two-panel Gradio UI** — Left panel for conversation, right panel for retrieved context. Users see both the answer and the evidence simultaneously.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector store (local, persistent) |
| HuggingFace all-MiniLM-L6-v2 | Embedding model (local, free) |
| Groq API + openai/gpt-oss-120b | LLM inference |
| langchain-groq | Groq integration for LangChain |
| Gradio | Two-panel chat UI |
| python-dotenv | API key management |
| uv | Package management |

---

## ▶️ Setup

**1. Get a free Groq API key** at [console.groq.com](https://console.groq.com)

**2. Create a `.env` file** in the project root:

```
GROQ_API_KEY=your_api_key_here
```

**3. Install dependencies with uv:**

```bash
uv venv
.venv\Scripts\activate
uv sync
```

**4. Ingest the knowledge base (run once):**

```bash
uv run python implementation/ingest.py
```

This embeds all documents and saves them to the `vector_db/` folder.

---

## ▶️ Run

```bash
uv run python app.py
```

Opens automatically at `http://localhost:7860`

---

## 📌 Example Questions

```
→ What insurance products does PoliLens offer?
→ Tell me about the Health insurance product
→ What are the contract terms for business insurance?
→ Who are the key employees at PoliLens?
→ How does the claims process work?
→ What is the reinsurance product about?
```

---

## 💡 What This Project Demonstrates

- Complete RAG pipeline from document ingestion to answer generation
- Semantic chunking with overlap for context preservation
- Persistent vector store — ingest once, query forever
- Multi-turn conversation with combined question strategy
- Source attribution — every answer shows retrieved documents
- Clean separation of concerns — ingest.py, answer.py, app.py each have one job
- Local embeddings (zero cost) + cloud LLM (fast inference)
- Two-panel Gradio UI for answer and context visibility simultaneously

---

## 🔮 Next Steps — Advanced RAG (Coming Soon)

This is the foundation RAG implementation. The next version will add:

- Query expansion and rewriting
- Re-ranking retrieved documents by relevance score
- Semantic chunking (split by meaning, not character count)
- RAG evaluation with MRR and nDCG metrics
- Hybrid search (semantic + keyword combined)

---

## 📌 Notes

- Run `ingest.py` once before `app.py` — vector_db must exist first
- Never commit your `.env` file — it is listed in `.gitignore`
- `vector_db/` is excluded from git — regenerate locally by running ingest.py
- Knowledge base is synthetic data created for learning purposes

---

## 👨‍💻 Author

Vishruth — built as part of learning practical LLM engineering.
Seventh project in a series exploring real-world LLM application patterns.
First RAG implementation — foundation before Advanced RAG version.