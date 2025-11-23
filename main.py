import json
from fastapi import FastAPI, HTTPException
from groq import Groq
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Literal, List, Optional
import datetime
import uuid

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Charger ton JSON
with open("personal_data.json", "r", encoding="utf-8") as f:
    PERSONAL_DATA = json.load(f)

SYSTEM_PROMPTS = {
    "fr": """
    Tu es un assistant IA qui se base uniquement sur les informations personnelles suivantes pour répondre.
    Données personnelles :
    {data}

    Règles strictes :
    - Ne réponds JAMAIS avec des informations qui ne proviennent pas des données personnelles ci-dessus.
    - Réponds toujours de manière claire, naturelle et amicale.
    - Si la réponse à une question ne se trouve pas dans les données, réponds : "Je ne peux pas répondre à votre question car cette information ne me concerne pas."
    - Tu dois tenir compte de l'historique de la conversation pour donner des réponses cohérentes.
    """,
    "en": """
    You are an AI assistant who answers based solely on the following personal information.
    Personal data:
    {data}

    Strict rules:
    - NEVER answer with information that does not come from the personal data above.
    - Always respond in a clear, natural, and friendly manner.
    - If the answer to a question is not found in the data, answer: "I cannot answer your question because this information does not concern me."
    - You must take the conversation history into account to give coherent answers.
    """
}


ASKS_FILE = "asks.json"
CONVERSATION_HISTORY_FILE = "conversation_history.json"


def read_history() -> dict:
    try:
        with open(CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_history(history: dict):
    with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


class Query(BaseModel):
    question: str
    language: Literal["en", "fr"] = "fr"
    session_id: Optional[str] = None


class Answer(BaseModel):
    answer: str
    session_id: str


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


@app.post("/ask", response_model=Answer)
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
        "cool": "Cool !"
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

    session_id = query.session_id or str(uuid.uuid4())

    if question_lower in greetings:
        return {"answer": greetings[question_lower], "session_id": session_id}

    # --- Conversation History and Context Management ---

    # 1. Load full history and get current session's history
    history = read_history()
    session_history = history.get(session_id, [])

    # 2. Create the system prompt
    system_prompt = SYSTEM_PROMPTS[query.language].format(
        data=json.dumps(PERSONAL_DATA, indent=2)
    )

    # 3. Build the message list for the API
    messages = [{"role": "system", "content": system_prompt}]
    # Add the last 10 messages (5 turns) to keep the context relevant and payload small
    messages.extend(session_history[-10:])
    messages.append({"role": "user", "content": query.question})

    # 4. Call the Groq API
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=messages
    )
    answer = response.choices[0].message.content

    # 5. Update and save the conversation history
    session_history.append({"role": "user", "content": query.question})
    session_history.append({"role": "assistant", "content": answer})
    history[session_id] = session_history
    write_history(history)

    # --- (Optional) Keep logging individual asks ---
    asks = read_asks()
    new_ask = {
        "id": str(uuid.uuid4()),
        "question": query.question,
        "date": datetime.datetime.now().isoformat()
    }
    asks.append(new_ask)
    write_asks(asks)

    # 6. Return the answer and session_id
    return {"answer": answer, "session_id": session_id}


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
