import re
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.render import render_text_description


template = """You are a powerful chatbot to collect user's data input. 
    The user's data input must be return with these format:
    - Price: 
    - Time:
    - Review star:
    - Job name:
    
    **For example**
    user's data input: Price i want is 300k, the time take 2 hours, i need 3 stars view and the last one is babysitter job
    That the Final answer should be:
    Final answer: 
    - Price: 300k
    - Time: 2 hours
    - Review star: 3 
    - Job name: babysitter
    
    You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

PATTERNS = {
    "Price":       re.compile(r"Price\s*[:\-]?\s*([\d.,]+\s*[kKmM]?)", re.IGNORECASE),
    "Time":      re.compile(r"Time\s*[:\-]?\s*([\d.,]+\s*(giờ|phút|tiếng)?)", re.IGNORECASE),
    "Review star":  re.compile(r"Review star\s*[:\-]?\s*(\d+)\s*sao", re.IGNORECASE),
    "Job name": re.compile(r"Job name\s*[:\-]?\s*([A-Za-zÀ-Ỹ0-9 ]+)", re.IGNORECASE),
}

def input_data():
    inp = input("\nInput: ") + "\n"
    return inp

def parse_input(text: str) -> str:
    info = {}
    for field, pattern in PATTERNS.items():
        m = pattern.search(text)
        if m:
            info[field] = m.group(1).strip()
    if not info:
        return "Not extracted information yet"
    return "\n".join(f"{k}: {v}" for k, v in info.items())

def ask_field(field: str) -> str:
    return f"\nPlease provide value for “{field}”."


parse_fc = StructuredTool.from_function(
    func=parse_input,
    name="parse_input",
    description="Parse input user to have format "
)

ask_f = StructuredTool.from_function(
    func=ask_field,
    name="ask_field",
    description="Ask user if it not have enough information"
)

inp_fc = StructuredTool.from_function(
    func=input_data,
    name="Input missing data",
    description="Input missing field data"
)

tools = [parse_fc, ask_f, inp_fc]

llm = ChatOpenAI(model="gpt-4o")

# Construct the ReAct agent
agent = create_react_agent(llm, tools=tools, prompt=prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools, handle_parsing_errors=True)


user_query = input("❓ Enter your request: ")
response = agent_executor.invoke({"input": user_query})
