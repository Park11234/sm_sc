from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def openAiModel():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def makeMsg(system,user ):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages

def openAiModelArg(model, msgs):
    print(model)
    print(msgs)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=msgs
    )
    return response.choices[0].message.content

def geminiModel():
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model