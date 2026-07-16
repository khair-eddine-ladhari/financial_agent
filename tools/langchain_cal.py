from langchain_core.tools import tool


@tool
def calculate_financial_ratio(revenue: float, expenses: float) -> str:
    """Calculate net income and profit margin given revenue and expenses in raw dollars
    (not millions). Example input: revenue=25000000, expenses=22000000 (i.e. $25,000,000 and $22,000,000)
    """
    if revenue == 0:
        return "Cannot calculate margin: revenue is zero."

    net_income = revenue - expenses
    margin = (net_income / revenue) * 100

    return (
        f"Net income: ${net_income:,.2f}M | "
        f"Profit margin: {margin:.1f}%"
    )