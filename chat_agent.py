from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware, ToolRetryMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from knowledge_retrieval import search_company_docs
from database_retrieval import search_database

load_dotenv()

class PhoneRedactionMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        # Execute tool
        tool_message = handler(request)
        
        if isinstance(tool_message, ToolMessage) and isinstance(tool_message.content, str):
            # Strict pattern matching (+91 1234-5678)
            specific_pattern = r'\+91\s\d{4}-\d{4}'
            tool_message.content = re.sub(specific_pattern, "[REDACTED PHONE]", tool_message.content)
            
        return tool_message

# Can do this too instead (wrapper function approach instead of creating AgentMiddleware class), and middleware=[redact_phone_numbers] instead of middleware=[PhoneRedactionMiddleware()]
# def redact_phone_numbers(request, handler):
#     # Execute the tool
#     tool_message = handler(request)
#     # Redact phone numbers from string outputs
#     if isinstance(tool_message, ToolMessage) and isinstance(tool_message.content, str):
#         specific_pattern = r'\+91\s\d{4}-\d{4}'
#         tool_message.content = re.sub(specific_pattern, "[REDACTED PHONE]", tool_message.content)       
#     return tool_message

#AGENT_MODEL = "qwen/qwen3.8-27b"
AGENT_MODEL = "openai/gpt-oss-120b"
AGENT_LLM = ChatGroq(model=AGENT_MODEL, temperature=0.2)

# Role, Task, Context, Constraints, Format, Examples
SYSTEM_PROMPT = """
You are an internal assistant for a FinTech company. Your job is to answer employee questions accurately using ONLY the provided tools.

RULES
1. Scope:
    - Respond naturally to basic greetings (e.g., "Hi", "Hello") and questions about your own capabilities.
    - Only answer questions relevant to the company or internal operations.
    - If a question is unrelated or out-of-scope for the company, do NOT call any tools and reply strictly with: 
        "I can't help you with that. Please ask questions relevant to your role."

2. Tool Selection: You have access to two tools - search_company_docs and search_database, decide which to call on basis of question asked by user.
    - search_company_docs for documentation, reports etc.
    - search_database for employee records and details

3. Complex Queries: If the user has asked a complex multi-part question, break it into simpler sub-questions and make multiple tool calls if required.
    When calling `search_company_docs`, keep keywords close to the user's original phrasing, even if you break it down into simpler sub-questions.

4. Citations: Always cite your sources using the returned metadata.
    - For documents, cite [Document Name, Page Number]. If page number is unavailable, cite [Document Name].
    - For database results, cite [Table Name].

5. Authorization: 
    - If search_database tool indicates that user doesn't have authorization to access database, do not attempt workarounds and reply strictly for any part requiring database access with:
        "You don't have authorization to access database"
    - If search_company_docs tool answers "This information is not available in the documents accessible to your role." or "I can't help you with that. Please ask questions relevant to your role.", give the exact same reply to user 

6. Accuracy: 
    - Never guess, extrapolate or invent facts. Rely strictly on tool outputs. Don't hallucinate. 
    - If you can't find the answer after checking the appropriate tool(s), stop and reply strictly with:
        "I couldn't find the answer."
    - For multi-part questions, provide the answers you did find with citations, and explicitly state which parts could not be found.
    - If the response or part of response is redacted, mention it

7. Style:
    - Use a friendly and clear conversational style, like you're explaining it to a colleague.
    - Avoid giving unnecessarily long answers.
"""

CHECKPOINTER = InMemorySaver()

AGENT = create_agent(
    model = AGENT_LLM,
    tools = [search_company_docs, search_database],
    system_prompt = SYSTEM_PROMPT,
    middleware=[PhoneRedactionMiddleware(), ModelRetryMiddleware(max_retries=2), ToolRetryMiddleware(max_retries=2)],
    checkpointer=CHECKPOINTER
)

def chat(user_question:str, user_role:str, session_thread_id:str)->str:

    config = {
        'configurable' : {
            'thread_id': session_thread_id,
            'user_role': user_role
        },
        'recursion_limit' : 20
    }

    try:
        result = AGENT.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question
                    }
                ]
            },
            config=config
        )
        return result['messages'][-1].content

    except GraphRecursionError:
        return "Recursion limit reached. I couldn't find the answer. Try breaking down your question into sub-questions."



if __name__=="__main__":
    from langchain_core.utils.uuid import uuid7

    current_user = 'USER_5'

    USER_DATABASE = {
        "USER_1": {'user_role': 'c-level'},
        "USER_2": {'user_role': 'engineering'},
        "USER_3": {'user_role': 'finance'},
        "USER_4": {'user_role': 'marketing'},
        "USER_5": {'user_role': 'general'}
    }

    def get_user_role(user_id):
        if user_id in USER_DATABASE:
            return USER_DATABASE[user_id]['user_role']
        else:
            return 'general'

    print("\n\n")
    print("="*100)
    print("Welcome to Information Assistant for FinSolve Technologies")
    print("\n(Type 'exit' to close this application)")
    print("-"*100)

    session_thread_id = str(uuid7())
    
    while True:
        try:
            print("\nPlease ask your question : ")

            user_question = input()

            if user_question.lower() in ['exit']:
                break
                
            response = chat(user_question, get_user_role(current_user), session_thread_id)

            print(f"\nASSISTANT : \n{response}\n")
            print("----------"*10)

        except KeyboardInterrupt:
            break