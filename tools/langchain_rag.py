"""
Tool 3: Answer questions using an uploaded company filing (10-K, earnings PDF, etc.)
via RAG (Retrieval-Augmented Generation) with Pinecone as the vector store.

Uses Pinecone's own hosted inference API for embeddings (no OpenAI needed) --
same embed_texts() approach as your original FastAPI project, wrapped to match
LangChain's Embeddings interface so it plugs into PineconeVectorStore.
"""
import os
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt
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
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [r["values"] for r in result]
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
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

from langchain_experimental.text_splitter import SemanticChunker
from tenacity import retry, wait_random_exponential, stop_after_attempt


def add_overlap(chunks, overlap_chars=100):
    if not chunks:
        return chunks

    overlapped_chunks = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1].page_content
        curr_text = chunks[i].page_content

        raw_overlap = prev_text[-overlap_chars:]
        # snap to the next full word instead of cutting mid-word
        first_space = raw_overlap.find(" ")
        overlap_text = raw_overlap[first_space + 1:] if first_space != -1 else raw_overlap

        if not curr_text.startswith(overlap_text):
            new_content = overlap_text.strip() + " " + curr_text
        else:
            new_content = curr_text

        new_chunk = chunks[i].model_copy(deep=True) if hasattr(chunks[i], "model_copy") else chunks[i]
        new_chunk.page_content = new_content
        overlapped_chunks.append(new_chunk)

    return overlapped_chunks


#semantic chunking the best way to get the best results

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def ingest_filing(file_path: str, namespace: str, company_metadata: dict) -> int:
    loader = _get_loader(file_path)
    documents = loader.load()

    splitter = SemanticChunker(
        embeddings=embedding_function,
        breakpoint_threshold_type="gradient",
        breakpoint_threshold_amount=0.9,
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
    chunks = add_overlap(chunks, overlap_chars=100)   # <-- new line

    for chunk in chunks:
        chunk.metadata.update(company_metadata)

    PineconeVectorStore.from_documents(
        chunks, embedding=embedding_function,
        index_name=PINECONE_INDEX_NAME, namespace=namespace,
    )
    return len(chunks)

@tool
def query_company_filing(question: str, namespace: str = "default") -> str:
    """Answer a question using the company's uploaded financial filing (10-K, earnings report, etc.)
    via document retrieval. Use this when the question needs specific numbers or statements
    from the official filing.

    A single call retrieves multiple relevant sections at once, so do NOT call this multiple
    times for related facts about the same company (e.g. revenue, expenses, net income, margin).
    Instead, ask one broader question like 'financial performance this quarter' and it will
    return all relevant sections together.

    If multiple similar figures exist (e.g. total revenue vs. segment revenue, current quarter
    vs. prior year), prefer the total/current-period figure unless the question explicitly asks
    for a specific segment or period.

    Example input: question='What was Tesla's total revenue in Q3?', namespace='tesla-10k'
    """
    print(f"QUERYING: question='{question}' namespace='{namespace}'")
    try:
        stats = index.describe_index_stats()
        if namespace not in stats.get("namespaces", {}):
            return f"No filing found for namespace '{namespace}'. Ingest a document first."

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_function,
            namespace=namespace,
        )

        retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12, "lambda_mult": 0.5, "fetch_k": 30}
)

        results = retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        if not context.strip():
            return "The filing does not contain information relevant to this question."

        # surface the verified metadata so the agent has ground truth to check against
        meta = results[0].metadata
        header = f"[Verified company info: {meta.get('company_name')}, sector: {meta.get('sector')}, country: {meta.get('country')}]\n\n"

        print(f"RESULT (first 150 chars): {(header + context)[:150]}")
        return header + context

            

  
    except Exception as e:
        return f"Error querying filing: {str(e)}"