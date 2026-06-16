from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool


def _first_line(value: str) -> str:
    """Use only the first line of model output as tool input."""
    return value.strip().splitlines()[0].strip()


def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        expression = _first_line(expression)
        allowed = set("0123456789+-*/(). ")
        if not all(char in allowed for char in expression):
            return "Error calculating: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"


def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        text = _first_line(text)
        format_type, content = text.split(": ", 1)
        format_type = format_type.strip().lower()
        if format_type == "uppercase":
            return content.upper()
        if format_type == "lowercase":
            return content.lower()
        if format_type == "titlecase":
            return content.title()
        return f"Error formatting text: unknown format type '{format_type}'"
    except Exception as e:
        return f"Error formatting text: {str(e)}"


tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description=(
            "A simple calculator that can add, subtract, multiply, or divide numbers. "
            "Input should be a mathematical expression like '2 + 2' or '15 / 3'."
        ),
    ),
    Tool(
        name="TextFormatter",
        func=format_text,
        description=(
            "Format text to uppercase, lowercase, or title case. "
            "Input should be in format: '[format_type]: [text]' "
            "where format_type is 'uppercase', 'lowercase', or 'titlecase'."
        ),
    ),
]

prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
Answer the following questions as best you can. You have access to the following tools:

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

Here are examples:

Question: What is 2 + 2?
Thought: I should use the calculator for this math problem.
Action: Calculator
Action Input: 2 + 2
Observation: 4
Thought: I now know the final answer
Final Answer: 4

Question: Can you convert 'hello world' to uppercase?
Thought: I should use the text formatter.
Action: TextFormatter
Action Input: uppercase: hello world
Observation: HELLO WORLD
Thought: I now know the final answer
Final Answer: HELLO WORLD

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# Free local language model (no API key required).
# First run downloads the model (~700MB).
llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "temperature": 0.0,
        "return_full_text": False,
    },
)

prompt = PromptTemplate.from_template(prompt_template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

print("=== Tool smoke tests ===")
print(f"Calculator: 25 + 63 = {calculator('25 + 63')}")
print(f"TextFormatter: {format_text('uppercase: hello world')}")
print(f"TextFormatter: {format_text('titlecase: langchain is awesome')}")

test_questions = [
    "What is 25 + 63?",
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7",
    "titlecase: langchain is awesome",
]

print("\n=== Agent tests ===")
for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"input": question})
    print(f"Final Answer: {result['output']}")
