import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from saving import save_to_json

load_dotenv(".env")


class Agent:
    def __init__(self):
        
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.deepseek_llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0
        )

        self.response_schemas = [
            ResponseSchema(name="price", description="Price of job"),
            ResponseSchema(name="time", description="Time of job"),
            ResponseSchema(name="review_star", description="Review star"),
            ResponseSchema(name="job_name", description="Job name"),
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas=self.response_schemas
        )

        self.tools = [
            StructuredTool.from_function(self.parse_input_with_llm, name="parse_input", description="Extract structured data from user input as JSON."),
            StructuredTool.from_function(self.ask_field, name="ask_field", description="Ask user if not enough information provided."),
            StructuredTool.from_function(self.input_data, name="input_missing_data", description="Input missing field data"),
        ]

        self.prompt = PromptTemplate.from_template(
            open("Prompts/agent_prompt.txt").read()
        )

        self.parse_prompt = open("Prompts/parse_prompt.txt").read()

        self.agent = create_react_agent(
            llm=self.llm, 
            tools=self.tools, 
            prompt=self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent, verbose=True, tools=self.tools, handle_parsing_errors=True, return_intermediate_steps=True
        )


    def parse_input_with_llm(self, text: str) -> dict:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": self.parse_prompt},
            {"role": "user", "content": text},
        ]
        return llm.invoke(messages).content

    def input_data(self):
        return input("\nInput: ")

    def ask_field(self, field: str) -> str:
        return f"Please provide value for '{field}'."


    def execute(self, query):
        result = self.executor.invoke(
            {"input": query, "format_instructions": self.output_parser.get_format_instructions()}
        )
        return result["output"], result["intermediate_steps"]


if __name__ == "__main__":
    agent = Agent()

    user_query = input("❓ Enter your request: ")
    output, intermediate_steps = agent.execute(user_query)
    
    print("\nIntermediate Steps:")
    for step in intermediate_steps:
        print(f"\nAction: {step[0]}")
        print(f"Observation: {step[1]}")
    
    print("\nFinal Output:", output)

    save_to_json(output, "data.json")
    print(output)