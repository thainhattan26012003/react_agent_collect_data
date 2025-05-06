import os
from dotenv import load_dotenv
from saving import save_to_json
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.callbacks import StdOutCallbackHandler

load_dotenv(".env")

price_schema = ResponseSchema(name="price", description="Price of job")
time_schema = ResponseSchema(name="time", description="Time of job")
review_star_schema = ResponseSchema(name="review_star", description="Review star")
job_name_schema = ResponseSchema(name="job_name", description="Job name")

response_schemas = [price_schema, time_schema, review_star_schema, job_name_schema]

output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas=response_schemas
)

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


def parse_input_with_llm(text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a AI assistant system to extract structure information from user's input\n"
                "Must return 4 line follow by this format below, dont add anything"
                "If there are not value for some field, dont return that"
                "\n- Price: <price>\n"
                "- Time: <time>\n"
                "- Review star: <review_star>\n"
                "- Job name: <job_name>\n"
            ),
        },
        {"role": "user", "content": text},
    ]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    data = llm.invoke(messages).content

    return data


parse_fc = StructuredTool.from_function(
    func=parse_input_with_llm,
    name="parse_input",
    description="Extract structured data from user input as JSON.",
)


def input_data():
    inp = input("\nInput: ") + "\n"
    return inp


inp_fc = StructuredTool.from_function(
    func=input_data, name="Input missing data", description="Input missing field data"
)


def ask_field(field: str) -> str:
    return f"\nPlease provide value for “{field}”\n."


ask_fc = StructuredTool.from_function(
    func=ask_field,
    name="ask_field",
    description="Ask user if it not have enough information",
)


tools = [parse_fc, ask_fc, inp_fc]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Construct the ReAct agent
agent = create_react_agent(llm, tools=tools, prompt=prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent, verbose=True, tools=tools, handle_parsing_errors=True
)


user_query = input("❓ Enter your request: ")
response = agent_executor.invoke(
    {"input": user_query, "format_instructions": format_instructions}
)



# # Lấy các bước trung gian
# steps = response.get("intermediate_steps", [])
# print("step:", steps)

# # In ra Action Input (nếu có)
# for step in steps:
#     if "actions" in step:
#         for action in step["actions"]:
#             if action.tool == "ask_field":  # Chỉ in ra khi tool là ask_field
#                 print(f"Yêu cầu nhập: {action.tool_input}")

# # In ra kết quả cuối cùng
# print("\nKết quả cuối cùng:")
# print(response["output"])