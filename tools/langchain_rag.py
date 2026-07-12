"""
Tool 3: Answer questions using an uploaded company filing (10-K, earnings PDF, etc.)
via RAG (Retrieval-Augmented Generation) with Pinecone as the vector store.

Uses Pinecone's own hosted inference API for embeddings (no OpenAI needed) --
same embed_texts() approach as your original FastAPI project, wrapped to match
LangChain's Embeddings interface so it plugs into PineconeVectorStore.
"""
import os
from typing import List

from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pinecone import Pinecone

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = "multilingual-e5-large"  # Pinecone's hosted embedding model

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)


class PineconeEmbeddings(Embeddings):
    """
    Wraps Pinecone's hosted inference API (pc.inference.embed) so it matches
    LangChain's Embeddings interface (embed_documents / embed_query).
    This replaces needing OpenAIEmbeddings -- no OpenAI key required.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [r["values"] for r in result]

    def embed_query(self, text: str) -> List[float]:
        result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        return result[0]["values"]


embedding_function = PineconeEmbeddings()


def _get_loader(file_path: str):
    """Pick the right LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_filing(file_path: str, namespace: str) -> int:
    """
    One-time ingestion step: load a document (PDF or TXT), split it into
    chunks, embed with Pinecone's hosted model, and upsert into Pinecone
    under the given namespace.
    Call this once per document before the agent can query it.
    Returns the number of chunks created.
    """
    loader = _get_loader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding_function,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )
    return len(chunks)


@tool
def query_company_filing(question: str, namespace: str = "default") -> str:
    """Answer a question using the company's uploaded financial filing (10-K, earnings report, etc.)
    via document retrieval. Use this when the question needs specific numbers or statements
    from the official filing.
    Example input: question='What was Tesla's total revenue in Q3?', namespace='tesla-10k'
    """
    try:
        stats = index.describe_index_stats()
        if namespace not in stats.get("namespaces", {}):
            return f"No filing found for namespace '{namespace}'. Ingest a document first."

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_function,
            namespace=namespace,
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        results = retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        if not context.strip():
            return "The filing does not contain information relevant to this question."

        return context
    except Exception as e:
        return f"Error querying filing: {str(e)}"