"""
Tool 3: Answer questions using an uploaded company filing (10-K, earnings PDF, etc.)
via RAG (Retrieval-Augmented Generation) with Pinecone as the vector store.

This is the same RAG pipeline you built manually with FastAPI + Pinecone,
now wrapped with LangChain's PineconeVectorStore for consistent chunking,
embedding, and retrieval.
"""
import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader



PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)












def ingest_filing(pdf_path: str, namespace: str) -> int:
    """
    One-time ingestion step: load a PDF, split it into chunks, embed, and
    upsert into Pinecone under the given namespace.
    Call this once per document before the agent can query it.
    Returns the number of chunks created.
    """
    loader = TextLoader("document.txt")
    documents = loader.load()
    print(documents[0].page_content)

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