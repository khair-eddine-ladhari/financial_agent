from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing tools.langchain_rag

from tools.langchain_rag import index, query_company_filing

stats = index.describe_index_stats()
print("=== INDEX STATS ===")
print(stats)

result = query_company_filing.invoke({"question": "What was TechStack's revenue?", "namespace": "tesla"})
print("\n=== TOOL RESULT ===")
print(result)