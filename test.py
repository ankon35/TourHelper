import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the .env file to access the API key
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = api_key

# Initialize the OpenAI model with LangChain (use updated import)
llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)  # Use GPT-4 for this example

# Create a prompt template (this is optional and can be customized for more complex inputs)
prompt_template = "Write a one-sentence bedtime story about a unicorn."

# Wrap the prompt template in a list of messages with role and content
messages = [{"role": "user", "content": prompt_template}]

# Generate the output from the LLM
response = llm.invoke(messages)

# Print the result (access the content of the response)
print(response.content.strip())
