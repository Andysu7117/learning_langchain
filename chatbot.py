from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage

# 1. Set up a free local language model (no API key required).
# First run downloads the model (~700MB).
llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 64,
        "temperature": 0.2,
        "return_full_text": False,
    },
)

# 2. Create a simple conversation with chat history
history = ChatMessageHistory()

# Add some initial messages (optional)
history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hello, Alice. How can I help you today?")

# 3. Print the current conversation history
print("Initial Chat History:")
for message in history.messages:
    sender = "Human" if isinstance(message, HumanMessage) else "AI"
    print(f"{sender}: {message.content}")

# 4. Set up a conversation chain with memory
memory = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 5. Function to simulate a conversation
def chat_simulation(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses"""
    print("\n=== Beginning Chat Simulation ===")
    
    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {user_input}")
        
        # Get response from the conversation chain
        response = conversation.invoke(input=user_input)
        
        # Print the AI's response
        print(f"AI: {response['response']}")
    
    print("\n=== End of Chat Simulation ===")

# 6. Test with a series of related questions
test_inputs = [
    "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?"
]

chat_simulation(conversation, test_inputs)

# 7. Examine the conversation memory
print("\nFinal Memory Contents:")
print(conversation.memory.buffer)

# 8. Create a new conversation with a different type of memory (optional)
# Try implementing ConversationSummaryMemory or another type of memory
