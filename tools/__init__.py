from .langchain_stock import get_stock_price
from .langchain_news import search_recent_news
from .langchain_rag import query_company_filing, ingest_filing
from .langchain_cal import calculate_financial_ratio

all_tools = [
    get_stock_price,
    search_recent_news,
    query_company_filing,
    calculate_financial_ratio,
]