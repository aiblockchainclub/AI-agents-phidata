from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools


load_dotenv()
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True)],
    instructions=["Use Tables to diaplay the data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)
web_agent.print_response("summarise and compare analyst recommnedations and fundamentals for TSLA and NVDA ")
