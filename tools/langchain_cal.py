from langchain_core.tools import tool


@tool
def calculate_financial_ratio(revenue: float, expenses: float) -> str:
    """Calculate net income and profit margin given revenue and expenses (in the same currency unit).
    Example input: revenue=25000, expenses=22000  (millions of dollars)
    """
    if revenue == 0:
        return "Cannot calculate margin: revenue is zero."

    net_income = revenue - expenses
    margin = (net_income / revenue) * 100

    return (
        f"Net income: ${net_income:,.2f}M | "
        f"Profit margin: {margin:.1f}%"
    )