from google import genai
from config import GEMINI_API

client = genai.Client(api_key=GEMINI_API)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)