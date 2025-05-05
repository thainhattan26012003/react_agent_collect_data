import re
from saving import save_to_json
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

price_schema = ResponseSchema(name="price", description="Price of job")
time_schema = ResponseSchema(name="time", description="Time of job")
review_star_schema = ResponseSchema(name="review_star", description="Review star")
job_name_schema = ResponseSchema(name="job_name", description="Job name")

response_schemas = [price_schema,
                   time_schema,
                   review_star_schema,
                   job_name_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)

format_instructions = output_parser.get_format_instructions()


template = """You are a powerful chatbot to collect user's data input. 
    
    You must return your *Final Answer* exactly in the JSON format below, with NO extra text before or after:

    {format_instructions}
    
    If the user's input lacks any required field, you must explicitly request that field from the user using the 'ask_field' action.
    
    The user's data input must be return with these format:
    - Price: 
    - Time:
    - Review star:
    - Job name:
    
    ---

    **Example for your understanding (do not use this as actual input):**

    Example user input: "Price I want is 300k, the time take 2 hours, I need 3 stars view and the last one is babysitter job"

    Example Final Answer:
    - Price: 300k
    - Time: 2 hours
    - Review star: 3
    - Job name: babysitter

    ---

When answering, process ONLY the actual question provided below as input.

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
    "Price": re.compile(r"(\d+[.,]?\d*\s*[kKmM]?)", re.IGNORECASE),
    "Time": re.compile(r"(\d+[.,]?\d*\s*(hours|hour?))", re.IGNORECASE),
    "Review star": re.compile(r"(\d+)\s*(review star|review stars|stars|star?)", re.IGNORECASE),
    "Job name": re.compile(r"(?:job is|job's|job name is|as|i work as|as a)\s*([A-Za-zÀ-Ỹ0-9 ]+)", re.IGNORECASE),
}

def input_data():
    inp = input("\nInput: ") + "\n"
    return inp

def parse_input(text: str) -> dict:
    info = {}
    for field, pattern in PATTERNS.items():
        match = pattern.search(text)
        info[field] = match.group(1).strip() if match else None
    return "\n".join(f"{k}: {v}" for k, v in info.items())

def ask_field(field: str) -> str:
    return f"\nPlease provide value for “{field}”\n."

parse_fc = StructuredTool.from_function(
    func=parse_input,
    name="parse_input",
    description="Extract structured data from user input as JSON."
)

ask_fc = StructuredTool.from_function(
    func=ask_field,
    name="ask_field",
    description="Ask user if it not have enough information"
)

inp_fc = StructuredTool.from_function(
    func=input_data,
    name="Input missing data",
    description="Input missing field data"
)

tools = [parse_fc, ask_fc, inp_fc]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Construct the ReAct agent
agent = create_react_agent(llm, tools=tools, prompt=prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools, handle_parsing_errors=True)


user_query = input("❓ Enter your request: ")
response = agent_executor.invoke({"input": user_query,
                                  "format_instructions": format_instructions})



final_output = response["output"] if "output" in response else response
print("\n🎯 Final Structured Output:\n", final_output)

    
save_to_json(final_output, "data.json")