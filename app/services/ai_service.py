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

    # Defensive checks: some API responses may be None or missing fields
    if response is None:
        raise RuntimeError("AI service returned no response")
    # Try to extract text from several common response shapes.
    # 1) Attribute-style: response.choices[0].message.content
    try:
        choices = getattr(response, 'choices', None)
        if choices and len(choices) > 0:
            first = choices[0]
            return getattr(first, 'message').content
    except Exception:
        pass

    # 2) Dict-like: response['choices'][0]['message']['content']
    try:
        if isinstance(response, dict):
            choices = response.get('choices')
            if choices and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    # common shapes: {'message': {'content': '...'}} or {'text': '...'}
                    msg = first.get('message')
                    if isinstance(msg, dict) and 'content' in msg:
                        return msg.get('content')
                    if 'text' in first:
                        return first.get('text')

            # groq or other services might return outputs: [{'content': '...'}]
            outputs = response.get('outputs')
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                out0 = outputs[0]
                if isinstance(out0, dict):
                    # try several keys
                    for key in ('content', 'text', 'output'):
                        if key in out0:
                            return out0.get(key)
    except Exception:
        pass

    # As a last resort, try to stringify the response so the app gets something useful
    try:
        return str(response)
    except Exception:
        raise RuntimeError("AI service returned an unreadable response")
