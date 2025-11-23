import json
from fastapi import FastAPI, HTTPException
from groq import Groq
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Literal, List
import datetime
import uuid

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Charger ton JSON
with open("personal_data.json", "r", encoding="utf-8") as f:
    PERSONAL_DATA = json.load(f)

PROMPTS = {
    "fr": """
    Tu es un assistant IA qui répond uniquement avec les informations suivantes :
    {data}

    Ne répond JAMAIS avec des infos inexistantes.
    Réponds à cette question de manière claire, naturelle et amicale: {question}
    
    Si la réponse ne se trouve pas dans les données ci-dessous, réponds :
    Je ne peux pas répondre votre question car c'est une information qui ne lie pas moi
    """,
    "en": """
    You are an AI assistant that only answers with the following information:
    {data}

    NEVER answer with non-existent information.
    Answer this question clearly, naturally, and in a friendly manner: {question}
    
    If the answer is not in the data below, answer:
    I cannot answer your question because it is information that does not bind me
    """
}

DATA = {
    "fr": PERSONAL_DATA,
    "en": PERSONAL_DATA
}

ASKS_FILE = "asks.json"


class Query(BaseModel):
    question: str
    language: Literal["en", "fr"] = "fr"


class Ask(BaseModel):
    id: str
    question: str
    date: str


def read_asks() -> List[Ask]:
    try:
        with open(ASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_asks(asks: List[dict]):
    with open(ASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(asks, f, indent=4)


@app.post("/ask")
async def ask(query: Query):
    question_lower = query.question.lower().strip().rstrip(".,;!?")

    greetings_fr = {
        "merci": "De rien !",
        "merci beaucoup": "Avec plaisir !",
        "merci bien": "Je vous en prie !",
        "bonjour": "Bonjour ! Comment puis-je vous aider ?",
        "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
        "salut": "Salut ! Comment puis-je vous aider ?",
        "coucou": "Coucou ! Comment puis-je vous aider ?",
        "au revoir": "Au revoir !",
        "à bientôt": "À bientôt !",
        "super": "Ravi de l'entendre !",
        "top": "Parfait !",
        "bien": "Parfait !",
        "génial": "Super !",
        "excellent": "Parfait !",
        "ok": "Très bien !",
        "parfait": "Parfait !",
        "cool": "Cool !",
    }

    greetings_en = {
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "thank you very much": "You're welcome!",
        "hello": "Hello! How can I help you?",
        "hi": "Hi! How can I help you?",
        "hey": "Hey! How can I help you?",
        "good morning": "Good morning! How can I help you?",
        "good afternoon": "Good afternoon! How can I help you?",
        "good evening": "Good evening! How can I help you?",
        "bye": "Goodbye!",
        "goodbye": "Goodbye!",
        "see you": "See you soon!",
        "super": "Glad to hear it!",
        "top": "Perfect!",
        "good": "Great!",
        "awesome": "Awesome!",
        "great": "Great!",
        "ok": "Alright!",
        "perfect": "Perfect!",
        "cool": "Cool!",
    }

    greetings = greetings_fr if query.language == "fr" else greetings_en

    if question_lower in greetings:
        return {"answer": greetings[question_lower]}

    # Save the question
    asks = read_asks()
    new_ask = {
        "id": str(uuid.uuid4()),
        "question": query.question,
        "date": datetime.datetime.now().isoformat()
    }
    asks.append(new_ask)
    write_asks(asks)

    prompt = PROMPTS[query.language].format(
        data=json.dumps(DATA[query.language], indent=2),
        question=query.question
    )

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return {"answer": answer}


@app.get("/asks", response_model=List[Ask])
async def get_asks():
    return read_asks()


@app.get("/asks/{ask_id}", response_model=Ask)
async def get_ask(ask_id: str):
    asks = read_asks()
    for ask in asks:
        if ask["id"] == ask_id:
            return ask
    raise HTTPException(status_code=404, detail="Ask not found")


@app.delete("/asks/{ask_id}")
async def delete_ask(ask_id: str):
    asks = read_asks()
    ask_to_delete = None
    for ask in asks:
        if ask["id"] == ask_id:
            ask_to_delete = ask
            break

    if ask_to_delete is None:
        raise HTTPException(status_code=404, detail="Ask not found")

    asks.remove(ask_to_delete)
    write_asks(asks)
    return {"message": "Ask deleted successfully"}
