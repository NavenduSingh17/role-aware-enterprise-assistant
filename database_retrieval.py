from dotenv import load_dotenv
import sqlite3
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.tools import tool

load_dotenv()

COMPANY_DB = "./company.sqlite3"

#SQL_MODEL = "qwen/qwen3.8-27b"
SQL_MODEL = "openai/gpt-oss-120b"

SQL_LLM = ChatGroq(model=SQL_MODEL, temperature=0)

SCHEMA = """
Table: employees
Columns:
employee_id
full_name
role
department
email
location
date_of_birth
date_of_joining
manager_id - (this is employee_id of the manager)
salary
leave_balance - no. of leaves left
leaves_taken - no. of leaves already availed
attendance_pct - attendance in percentage
performance_rating
last_review_date
phone_number
"""

# Role, Task, Context, Constraints, Format, Examples
SQL_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert SQL Generator.
Your sole task is to translate natural language questions into valid, optimized, and executable SELECT statements for sqlite3 database.

SCHEMA:
{schema}

STRICT INSTRUCTIONS:
1. ONLY return the raw SQL code. Do NOT wrap the query in markdown code blocks (e.g., ```sql). No explanations, no intro text, no trailing text.
2. Use ONLY the table and column names explicitly defined in the SCHEMA. Do not invent columns.
3. Only generate READ-ONLY queries (SELECT). Any request involving INSERT, UPDATE, DELETE, or schema modifications must be ignored.
4. For hierarchical/manager queries, use a SELF-JOIN on the 'employees' table matching 'manager_id' to 'employee_id'.
5. Handle string/text matching case-insensitively using LOWER(column) LIKE LOWER('%value%') since standard SQLite functions apply. Do NOT use ILIKE.
6. When calculating averages or percentages, cast to numeric types to avoid integer division issues.
7. Always append `LIMIT 50` to non-aggregated queries to prevent oversized outputs.

Now, process the following input:

Question:
{question}
""").partial(schema=SCHEMA)

SQL_CHAIN = SQL_GENERATION_PROMPT | SQL_LLM | StrOutputParser()   

def execute_sql(sql: str):
    sql = sql.strip().rstrip(";")
    
    # Clean up accidental markdown code blocks if the LLM slips up
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.replace("```sql", "").replace("```", "").strip()

    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    # Open a read-only connection for data safety
    with sqlite3.connect(
        f"file:{COMPANY_DB}?mode=ro",
        uri=True
    ) as read_conn:
        read_conn.row_factory = sqlite3.Row
        cursor = read_conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are a data assistant.
Answer the question based ONLY on the provided SQL results.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the clear facts directly mentioned in the SQL results.
2. Do NOT assume, extrapolate, or invent any information. 
3. If the SQL results are empty, missing, or do not contain the answer, reply EXACTLY with: "Data not available."
4. Do not include any introductory text, pleasantries, or explanations. Keep the answer direct and short.

Question: {question}

SQL results: {results}
Answer:
""")

ANSWER_CHAIN = ANSWER_GENERATION_PROMPT | SQL_LLM | StrOutputParser()

# Define a pipeline wrapper that executes the query mid-transit
FULL_CHAIN = (
    {
        'question': RunnablePassthrough(),
        'results': SQL_CHAIN | execute_sql
    }
    | ANSWER_CHAIN
)

@tool
def search_database(query:str, config: RunnableConfig) -> str:
    """
    Search the internal company employees database to answer natural language questions 
    about staff, roles, salaries, departments, and management hierarchies.
    """

    # Extract user_role safely (without llm seeing it) from the runtime configuration. If not provided, fallback to default
    user_role = config.get("configurable", {}).get("user_role", "general")

    if user_role != 'c-level':
        return "User doesn't have authorization to access database"

    try:
        return FULL_CHAIN.invoke(query)
    
    except sqlite3.Error as e:
        return f"Database query failed: {str(e)}"
    except ValueError as e:
        return f"Invalid query generated: {str(e)}"