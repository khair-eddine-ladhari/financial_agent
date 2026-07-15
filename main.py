from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing agent/tools, since they read env vars at import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from agent import run_financial_analysis
from tools.langchain_rag import ingest_filing
from tenacity import retry, wait_random_exponential, stop_after_attempt

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
    namespace: str = "default"

class AnalysisResponse(BaseModel):
    raw_findings: str
    report: str


class IngestRequest(BaseModel):
    pdf_path: str
    namespace: str
    company_metadata: dict


class IngestResponse(BaseModel):
    status: str
    chunks: int


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """
    Main endpoint. Send a natural-language question, e.g.:
    "Analyze Tesla's Q3 performance and give me an investment recommendation"
    The agent decides which tools it needs and returns a formatted report.
    """
    try:
        result = run_financial_analysis(req.question, req.namespace)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-filing", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        num_chunks = ingest_filing(req.pdf_path, req.namespace, req.company_metadata)
        return IngestResponse(status="done", chunks=num_chunks)
    except Exception as e:
        traceback.print_exc()  # ← prints the full traceback to your terminal
        raise HTTPException(status_code=500, detail=str(e))
