from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_groq import ChatGroq


from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "openai/gpt-oss-120b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company PoliLens.
You are chatting with a user about PoliLens.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
# k needs to be set here at retriever-creation time, not passed into invoke()
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatGroq(temperature=0, model_name=MODEL)


def _as_text(content) -> str:
    """
    Gradio's Chatbot (type='messages') can hand back `content` as a plain
    string OR as a list of content-part dicts (e.g. [{"type": "text", "text": "..."}])
    depending on version/config. Normalize either shape down to a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # common shapes: {"text": "..."} or {"type": "text", "text": "..."}
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _sanitize_history(history: list[dict]) -> list[dict]:
    """
    Keep only role/content, coerce content to str, and drop any turns
    LangChain's convert_to_messages can't handle (e.g. stray tool/metadata keys).
    """
    clean = []
    for m in history or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        clean.append({"role": role, "content": _as_text(m.get("content"))})
    return clean


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    history = _sanitize_history(history)
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    question = _as_text(question)
    return (prior + "\n" + question) if prior else question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    question = _as_text(question)
    history = _sanitize_history(history)

    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs