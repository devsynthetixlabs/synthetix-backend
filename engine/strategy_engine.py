import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from engine.sql_engine import ask_cfo  # Import your perfected engine

load_dotenv()

# 1. THE BRAIN
# Using Llama 3.1 8b for speed and to minimize rate limits
my_llm = LLM(model="groq/llama-3.1-8b-instant", temperature=0.1)

# 2. THE TOOL
# We wrap your sql_engine in a CrewAI tool decorator
@tool("sales_data_tool")
def sales_data_tool(question: str, tenant_id: str) -> str:
    """Useful for querying the sales database for revenue, clients, and orders."""
    res = ask_cfo(question, tenant_id)
    # Ensure we return only the clean string answer
    if isinstance(res, dict) and 'answer' in res:
        return res['answer']
    return str(res)

# 3. THE AGENTS
analyst = Agent(
    role='Lead Financial Analyst',
    goal='Extract precise sales figures...',
    backstory="""You are an expert at using the sales_data_tool. 
    You must always include the tenant_id provided in your instructions 
    when calling the tool to ensure data privacy.""",
    tools=[sales_data_tool],
    llm=my_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,              # 🚨 Stops infinite loops
    max_execution_time=120,   # 🚨 Gives it 2 minutes to finish
    cache=True               # 🚨 Faster subsequent calls
)

strategist = Agent(
    role='Senior Business Consultant',
    goal='Develop marketing plans...',
    backstory="...",
    llm=my_llm,
    verbose=True,
    max_execution_time=120    # 🚨 Increase timeout
)

# 4. THE EXECUTION FUNCTION
def ask_strategy(question: str, tenant_id: str):
    """
    The main entry point for the Strategy Engine.
    Called by main.py when a strategic query is detected.
    """
    
    # Define tasks dynamically based on the user's specific question
    research_task = Task(
        description=(
            f"1. Use sales_data_tool to find data for: '{question}'.\n"
            f"2. You MUST use tenant_id: '{tenant_id}' for all tool calls.\n"
            "3. Summarize the findings with company names and ₹ amounts."
        ),
        expected_output="A summary report with the full company name and the exact amount in ₹.",
        agent=analyst
    )

    strategy_task = Task(
        description=f"Based on the analyst's report for tenant {tenant_id}, answer: {question}",        
        expected_output="A professional 3-step business strategy or response.",
        agent=strategist,
        context=[research_task]
    )

    # Initialize the Crew
    crew = Crew(
        agents=[analyst, strategist],
        tasks=[research_task, strategy_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        # Execute the workflow
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Strategy Engine Error: {str(e)}"