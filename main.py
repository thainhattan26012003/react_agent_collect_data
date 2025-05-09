import os
import json
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from saving import save_to_json
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv(".env")


class Agent:
    def __init__(self):
        
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.parse_llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.response_schemas = [
            ResponseSchema(name="Công việc", description="Công việc cần tìm"),
            ResponseSchema(name="Thời gian", description="Thời gian cần tìm"),
            ResponseSchema(name="Địa chỉ", description="Địa chỉ cần tìm"),
            ResponseSchema(name="Số năm kinh nghiệm", description="Số năm kinh nghiệm cần tìm"),
            ResponseSchema(name="Số sao đánh giá", description="Số sao đánh giá cần tìm"),
            ResponseSchema(name="Mức giá", description="Mức giá cần tìm")
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
            open("Prompts/agent_prompt.txt", encoding="utf-8").read()
        )

        self.parse_prompt = open("Prompts/parse_prompt.txt", encoding="utf-8").read()

        self.agent = create_react_agent(
            llm=self.llm, 
            tools=self.tools, 
            prompt=self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent, verbose=True, tools=self.tools, handle_parsing_errors=True
        )


    def _convert_text_to_json(self, text: str) -> dict:
        """Convert colon-separated text format to JSON dictionary."""
        result = {}
        # Split by lines and process each line
        for line in text.strip().split('\n'):
            # Split by first colon
            if ':' in line:
                key, value = line.split(':', 1)
                # Clean up key and value
                key = key.strip()
                value = value.strip()
                # Convert key to match schema names
                key_mapping = {
                    'Công việc': 'Công việc',
                    'Thời gian': 'Thời gian',
                    'Địa chỉ': 'Địa chỉ',
                    'Số năm kinh nghiệm': 'Số năm kinh nghiệm',
                    'Số sao đánh giá': 'Số sao đánh giá',
                    'Mức giá': 'Mức giá'
                }
                if key in key_mapping:
                    result[key_mapping[key]] = value
        return result

    def parse_input_with_llm(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": self.parse_prompt},
            {"role": "user", "content": text},
        ]
        
        with get_openai_callback() as parse_cb:
            response = self.parse_llm.invoke(messages).content
            
            print(f"\n[parse_input] Token usage:")
            print(f"Prompt tokens: {parse_cb.prompt_tokens}")
            print(f"Completion tokens: {parse_cb.completion_tokens}")
            print(f"Total tokens: {parse_cb.total_tokens}")
            print(f"Total cost: ${parse_cb.total_cost}\n")
            
            try:
                # Try to parse the response as JSON
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, try to convert text to JSON
                try:
                    return self._convert_text_to_json(response)
                except Exception as e:
                    raise ValueError(f"Could not parse response: {response}\nError: {str(e)}")

    def input_data(self):
        return input("\nInput: ")

    def ask_field(self, field: str) -> str:
        return f"Please provide value for '{field}'."

    def execute(self, query):
        with get_openai_callback() as cb_parse:
            self.parse_llm.callbacks = [cb_parse]

            parsed = self.parse_llm.invoke([
                {"role":"system","content": self.parse_prompt},
                {"role":"user","content": query}
            ]).content

            # Reset callback
            self.parse_llm.callbacks = []

        print("=== Parse token usage ===")
        print(f" Prompt tokens:    {cb_parse.prompt_tokens}")
        print(f" Completion tokens:{cb_parse.completion_tokens}")
        print(f" Total tokens:     {cb_parse.total_tokens}")
        print(f" Total cost:       ${cb_parse.total_cost}\n")

        with get_openai_callback() as cb_exec:
            self.llm.callbacks = [cb_exec]

            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )

            result = executor.invoke({
                "input": query,
                "format_instructions": self.output_parser.get_format_instructions()
            })

            # Reset lại callbacks
            self.llm.callbacks = []

        print("=== Execute token usage ===")
        print(f" Prompt tokens:    {cb_exec.prompt_tokens}")
        print(f" Completion tokens:{cb_exec.completion_tokens}")
        print(f" Total tokens:     {cb_exec.total_tokens}")
        print(f" Total cost:       ${cb_exec.total_cost}\n")

        return result["output"]


if __name__ == "__main__":
    agent = Agent()

    user_query = input("❓ Enter your request: ")
    output = agent.execute(user_query)

    save_to_json(output, "data.json")
    print(output)