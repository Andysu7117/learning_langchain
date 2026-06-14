def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# IBM WatsonX imports
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain  # Still using this for backward compatibility

def llm_model(prompt_txt, params=None):
    
    model_id = "ibm/granite-4-h-small"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    # Set up credentials for WatsonxLLM
    url = "https://us-south.ml.cloud.ibm.com"
    api_key = "your api key here"
    project_id = "skills-network"

    credentials = {
        "url": url,
        # "api_key": api_key
        # uncomment the field above and replace the api_key with your actual Watsonx API key
    }
    
    # Create LLM directly
    granite_llm = WatsonxLLM(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response

params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "Once upon a time in a distant galaxy"

zero_shot_prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""

# Getting a reponse from the model with the provided prompt and new parameters
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

response = llm_model(zero_shot_prompt, params)
print(f"prompt: {zero_shot_prompt}\n")
print(f"response : {response}\n")

params = {
    "max_new_tokens": 10,
}

few_shot_prompt = """Here are few examples of classifying emotions in statements:

            Statement: 'I just won my first marathon!'
            Emotion: Joy
            
            Statement: 'I can't believe I lost my keys again.'
            Emotion: Frustration
            
            Statement: 'My best friend is moving to another country.'
            Emotion: Sadness
            
            Now, classify the emotion in the following statement:
            Statement: 'That movie was so scary I had to cover my eyes.’
            

"""

response = llm_model(few_shot_prompt, params)
print(f"prompt: {few_shot_prompt}\n")
print(f"response : {response}\n")

params = {
    "max_new_tokens": 512,
    "temperature": 0.5,
}

chain_of_thought_prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
            How many apples are there now?’

            Break down each step of your calculation

"""
response = llm_model(chain_of_thought_prompt, params)
print(f"prompt: {chain_of_thought_prompt}\n")
print(f"response : {response}\n")
