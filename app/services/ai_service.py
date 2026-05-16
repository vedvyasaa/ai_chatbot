from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_ai(message):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=message
    )

    return response.choices[0].message.content
