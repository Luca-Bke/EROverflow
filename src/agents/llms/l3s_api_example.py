import os
from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

os.environ["OPENAI_BASE_URL"] = os.getenv(
    "LLMHUB_HOST", "https://brrr.kbs.uni-hannover.de/v1")
os.environ["OPENAI_API_KEY"] = os.getenv("LLMHUB_APIKEY")

client = OpenAI()

completion = client.chat.completions.create(
    model="vllm/gpt-oss:120b-mxfp4",
    messages=[
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": "Hello, who are you?"},
    ],
)
print(completion.choices[0].message.content)
