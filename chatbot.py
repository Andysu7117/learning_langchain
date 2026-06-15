from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import HuggingFacePipeline
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from typing import List, Optional


CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful assistant. Reply to the user's latest message with one answer only.
Do not write "Human:", "AI:", "User:", or "Assistant:" in your reply.
Do not continue the conversation or invent new questions.

Conversation so far:
{history}
User: {input}
Assistant:""",
)


def clean_response(text: str) -> str:
    """Remove fake follow-up turns that small models often hallucinate."""
    text = text.strip()
    turn_markers = [
        "\nHuman:",
        "\nAI:",
        "\nUser:",
        "\nAssistant:",
        "\n\nHuman:",
        "\n\nUser:",
    ]
    for marker in turn_markers:
        if marker in text:
            text = text.split(marker)[0]

    for marker in ("Human:", "User:"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    return text.strip()


class CleanHuggingFacePipeline(HuggingFacePipeline):
    """HuggingFace pipeline that trims hallucinated multi-turn output."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> LLMResult:
        result = super()._generate(
            prompts, stop=stop, run_manager=run_manager, **kwargs
        )
        for generation_list in result.generations:
            for generation in generation_list:
                generation.text = clean_response(generation.text)
        return result


def create_llm():
    """Create a free local language model (no API key required)."""
    return CleanHuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.2,
            "return_full_text": False,
        },
    )


def create_conversation(verbose: bool = False):
    """Create a conversation chain with empty memory."""
    history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=history,
        human_prefix="User",
        ai_prefix="Assistant",
    )
    return ConversationChain(
        llm=create_llm(),
        memory=memory,
        prompt=CHAT_PROMPT,
        verbose=verbose,
    )


def chat_simulation(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses."""
    print("\n=== Beginning Chat Simulation ===")

    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {user_input}")

        response = conversation.invoke(input=user_input)
        print(f"AI: {response['response']}")

    print("\n=== End of Chat Simulation ===")


def run_cli_demo():
    history = ChatMessageHistory()
    history.add_user_message("Hello, my name is Alice.")
    history.add_ai_message("Hello, Alice. How can I help you today?")

    print("Initial Chat History:")
    for message in history.messages:
        sender = "Human" if isinstance(message, HumanMessage) else "AI"
        print(f"{sender}: {message.content}")

    memory = ConversationBufferMemory(
        chat_memory=history,
        human_prefix="User",
        ai_prefix="Assistant",
    )
    conversation = ConversationChain(
        llm=create_llm(),
        memory=memory,
        prompt=CHAT_PROMPT,
        verbose=True,
    )

    test_inputs = [
        "My favorite color is blue.",
        "I enjoy hiking in the mountains.",
        "What activities would you recommend for me?",
        "What was my favorite color again?",
        "Can you remember both my name and my favorite color?",
    ]

    chat_simulation(conversation, test_inputs)

    print("\nFinal Memory Contents:")
    print(conversation.memory.buffer)


if __name__ == "__main__":
    run_cli_demo()
