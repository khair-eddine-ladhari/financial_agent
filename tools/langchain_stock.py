

import yfinance as yf
from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price and daily change for a given ticker symbol.
    Example input: 'TSLA', 'AAPL', 'GOOGL'
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.history(period="2d")

        if info.empty:
            return f"Could not find data for ticker '{ticker}'. Check the symbol is correct."

        latest_close = info["Close"].iloc[-1]
        prev_close = info["Close"].iloc[-2] if len(info) > 1 else latest_close
        change_pct = ((latest_close - prev_close) / prev_close) * 100

        return (
            f"{ticker.upper()} is currently trading at ${latest_close:.2f}, "
            f"{'up' if change_pct >= 0 else 'down'} {abs(change_pct):.2f}% from the previous close."
        )
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"