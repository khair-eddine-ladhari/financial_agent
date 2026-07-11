from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import run_financial_analysis
from tools.filing_rag import ingest_filing

load_dotenv()

app = FastAPI(title="Financial Research Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    raw_findings: str
    report: str


class IngestRequest(BaseModel):
    pdf_path: str
    namespace: str


class IngestResponse(BaseModel):
    status: str
    chunks: int


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """
    Main endpoint. Send a natural-language question, e.g.:
    "Analyze Tesla's Q3 performance and give me an investment recommendation"
    """
    try:
        result = run_financial_analysis(req.question)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-filing", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    One-time step: ingest a company's PDF filing so the RAG tool can query it.
    """
    try:
        num_chunks = ingest_filing(req.pdf_path, req.namespace)
        return IngestResponse(status="done", chunks=num_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))