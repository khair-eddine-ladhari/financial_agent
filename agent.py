"""
Sets up the ReAct agent (reasoning + tool use) and a sequential chain
that formats the agent's raw findings into a polished report.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from tools import all_tools

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

FINANCIAL_AGENT_SYSTEM_PROMPT = """You are a financial research analyst assistant with access to the following tools:

- get_stock_price: real-time stock price and daily change for a ticker
- search_recent_news: recent news/events about a company
- query_company_filing: retrieves specific facts and figures from an uploaded official filing (10-K, earnings report)
- calculate_financial_ratio: precise calculation of net income and profit margin from revenue/expenses

Guidelines for using these tools:
- Only call the tools that are actually needed to answer the question.
- Prefer query_company_filing over your own knowledge for any specific financial figures -- never estimate or guess numbers.
- Use calculate_financial_ratio whenever a calculation is required, instead of computing it yourself.
- Use search_recent_news for qualitative context, not for hard numbers.
- If a tool returns no useful information, say so plainly instead of filling the gap with assumptions.
- Do not fabricate data.
- Keep your final answer factual and neutral. Frame recommendations as analysis, not instructions to buy/sell.
- If the user's question is narrow, answer directly without unnecessary tool calls.
-Before incorporating any information from search_recent_news, verify it 
matches the company's verified details (sector, country, business description) 
as found in query_company_filing. If a search result describes a company in a 
different sector, country, or of clearly different scale, treat it as 
referring to a DIFFERENT company and discard it -- do not include it in your analysis.
  When asked for qualitative judgment or context (e.g. "is this healthy", "how does this compare"), 
  only use general, industry-level reasoning (e.g. typical margin ranges for the sector) if you 
  clearly label it as general knowledge, not a fact about this specific company.
- Never state specific claims about a named company's competitors, strategy, plans, or products 
  unless that information came directly from a tool result (query_company_filing or search_recent_news). 
  If you don't have that information, say so explicitly rather than inferring or guessing it."""

# Step 1: the ReAct agent -- decides which tools to call, in what order,
# based on reasoning about the user's question.
agent = create_react_agent(llm, all_tools, prompt=FINANCIAL_AGENT_SYSTEM_PROMPT)

# Step 2: a sequential chain (LCEL) that takes the agent's raw findings
# and reformats them into a clean, structured report.
report_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst assistant. Turn the raw research notes below "
     "into a clear answer for the user.\n\n"
     "Rules:\n"
     "1. Match your response's length and structure to the complexity of the "
     "findings and the original question. If the findings are a single fact "
     "(e.g. one number), answer in one sentence -- do NOT create a full report "
     "with Summary/Risks/Recommendation sections.\n"
     "2. Only use the Summary/Key Financials/Risks/Recommendation structure "
     "when the findings include multiple data points that warrant real analysis "
     "(e.g. financials plus stock price plus news).\n"
     "3. Never rename, relabel, or reinterpret what a figure represents. "
     "Use the exact same label the raw findings use (e.g. if raw findings say "
     "'expenses', do not call it 'investment value', 'cost basis', or anything else).\n"
     "4. Do not add any information, risk, or recommendation not present in "
     "the raw findings."),
    ("human", "Original question: {question}\n\nRaw findings:\n{findings}")
])
report_chain = report_prompt | llm | StrOutputParser()





def run_financial_analysis(question: str, namespace: str = "default") -> dict:
    """
    Full pipeline: agent gathers findings using tools it decides to call,
    then the sequential chain formats those findings into a final report.
    """
    contextual_question = f"{question}\n\n(Use namespace='{namespace}' when calling query_company_filing.)"

    try:
        agent_result = agent.invoke(
            {"messages": [("human", contextual_question)]},
            config={"recursion_limit": 8}
        )
        raw_findings = agent_result["messages"][-1].content
    except GraphRecursionError:
        raw_findings = (
            "The agent could not converge on an answer within the allowed number "
            "of steps. It may have been stuck repeating the same tool call — "
            "check the tool's output for errors or empty results."
        )
        print("HIT RECURSION LIMIT — investigate tool result handling")

    final_report = report_chain.invoke({
    "findings": raw_findings,
    "question": question,
})

    return {
        "raw_findings": raw_findings,
        "report": final_report,
    }