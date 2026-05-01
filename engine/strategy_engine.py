from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

strategy_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    timeout=90
)

@tool("sales_data_tool")
def sales_data_tool(question: str, tenant_id: str) -> str:
    """Query the sales database for revenue, clients, orders, year-over-year comparisons. Returns formatted results with ₹ amounts."""
    from engine.sql_engine import ask_cfo
    res = ask_cfo(question, tenant_id)
    if isinstance(res, dict) and 'answer' in res:
        return res['answer']
    return str(res)

def ask_strategy(question: str, tenant_id: str):
    """
    The main entry point for the Strategy Engine.
    Called by api/index.py when a strategic query is detected.
    """
    analyst = Agent(
        role='Data Reporter',
        goal='Get numbers from the database and report ONLY the raw data.',
        backstory="You are a data reporter. Your ONLY job is to fetch and relay numbers. Never analyze or strategize.",
        tools=[sales_data_tool],
        llm=strategy_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    strategist = Agent(
        role='Business Consultant',
        goal='Give concise, direct strategy advice based on the data.',
        backstory="You are a business consultant. Answer directly using the analyst's data.",
        llm=strategy_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    research_task = Task(
        description=(
            f"Query the database for: '{question}'\n"
            f"Use tenant_id: '{tenant_id}'\n"
            "Make ONE tool call. Then output ONLY the data as a numbered list of company names and ₹ amounts. "
            "Do NOT give strategies or advice — that is someone else's job."
        ),
        expected_output="A short list of companies and ₹ amounts. No analysis.",
        agent=analyst
    )

    strategy_task = Task(
        description=(
            f"Based on the analyst's data above, answer: {question}\n"
            "Reference specific numbers. Keep it to 3-4 sentences max."
        ),        
        expected_output="A short answer with data references.",
        agent=strategist,
        context=[research_task]
    )

    crew = Crew(
        agents=[analyst, strategist],
        tasks=[research_task, strategy_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Strategy Engine Error: {str(e)}"
