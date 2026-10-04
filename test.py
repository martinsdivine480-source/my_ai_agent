import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

response = client.models.generate_content(
    model=model,
    contents="Say hello and tell me that you are connected successfully.",
)

print(response.text)