import re
from groq import Groq
from config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

CLASSIFIER_MODEL = "allam-2-7b"

CASUAL_PATTERNS = {
    "greeting": re.compile(
        r"^(hi|hello|hey|yo|sup|howdy|hii+|heyy+|namaste|good\s*(morning|afternoon|evening)|gm|gn)\s*[!.]*$",
        re.IGNORECASE,
    ),
    "thanks": re.compile(
        r"^(thanks?|thank\s*you|thx|ty|tysm|tq|shukriya|dhanyavaad|appreciate\s*it|cheers)\s*[!.]*$",
        re.IGNORECASE,
    ),
    "great_response": re.compile(
        r"^(great|nice|awesome|amazing|cool|perfect|excellent|good|wonderful|fantastic|brilliant|superb|ok|okay|got\s*it|understood|noted)\s*[!.]*$",
        re.IGNORECASE,
    ),
    "bye": re.compile(
        r"^(bye|goodbye|see\s*ya|see\s*you|later|take\s*care|good\s*night|gn|tata|alvida)\s*[!.]*$",
        re.IGNORECASE,
    ),
    "how_are_you": re.compile(
        r"^(how\s*(are\s*you|r\s*u)|how'?s\s*it\s*going|what'?s\s*up|kya\s*haal|kaise\s*ho)\s*[!.?]*$",
        re.IGNORECASE,
    ),
    "who_are_you": re.compile(
        r"^(who\s*are\s*you|what\s*are\s*you|tell\s*me\s*about\s*(yourself|you)|tum\s*kaun\s*ho|aap\s*kaun\s*hai)\s*[!.?]*$",
        re.IGNORECASE,
    ),
    "help": re.compile(
        r"^(help|help\s*me|what\s*can\s*you\s*do|commands|options|menu|kya\s*karsakte\s*ho)\s*[!.?]*$",
        re.IGNORECASE,
    ),
}

CASUAL_RESPONSES = {
    "greeting": [
        "Hey there! 👋 I'm Padhobeta, your study buddy. What can I help you with today?",
        "Hello! 😊 Welcome back! Ready to learn something new? Just ask me anything!",
        "Hi! Great to see you! What topic shall we explore today?",
    ],
    "thanks": [
        "You're welcome! 😊 That's what I'm here for. Got more questions? I'm all ears!",
        "Happy to help! 🎓 Feel free to ask anything else — I'm always here for you!",
        "Anytime! 🙌 Don't hesitate to come back whenever you need help!",
    ],
    "great_response": [
        "Thanks! That means a lot! 😄 Got any more questions? I'd love to help!",
        "Glad you liked it! 🎉 What else can I help you with?",
        "Awesome! Keep the questions coming — I'm here to make learning fun! 🚀",
    ],
    "bye": [
        "Goodbye! 👋 Keep learning and stay curious. See you next time!",
        "Take care! 🌟 Come back anytime you need help. Happy studying!",
        "Bye! 📚 Remember, every question makes you smarter. See you soon!",
    ],
    "how_are_you": [
        "I'm doing great, thanks for asking! 😊 More importantly, how can I help YOU today?",
        "All good on my end! 🎓 I'm ready to tackle any question you throw at me!",
        "I'm fantastic! 🚀 What about you? Need help with any study topics?",
    ],
    "who_are_you": [
        "I'm Padhobeta — your AI study buddy! 🎓 I can help you with academic questions from your uploaded documents, or search the web if needed. What would you like to learn?",
        "I'm Padhobeta, an AI educational assistant! 📚 I'm here to make studying easier. Upload a document and ask me anything!",
    ],
    "help": [
        "Here's what I can do! 🚀\n\n"
        "📄 **Upload a document** — PDF, DOCX, or PPTX\n"
        "💬 **Ask me anything** about your uploaded material\n"
        "🌐 **Web search** — If the answer isn't in your document, I'll search the web for you!\n"
        "🧠 **Smart routing** — Easy questions get quick answers, tough ones get deep analysis\n\n"
        "Just type your question and I'll handle the rest!",
    ],
}

CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for an educational chatbot called "Padhobeta".

Your ONLY job is to determine if a user's question is related to academics, education, learning, or the content of an uploaded document.

Reply with EXACTLY one word:
- EDUCATIONAL — if the question is about studies, academics, science, math, history, literature, programming, technology, concepts, definitions, explanations, homework, exams, or any learning topic.
- NOT_EDUCATIONAL — if the question is about weather, sports, entertainment, politics, personal advice, jokes, recipes, shopping, or anything clearly unrelated to education and learning.

When in doubt, lean towards EDUCATIONAL. Be lenient with edge cases."""


def detect_casual(query: str) -> str | None:
    """Detect casual messages (greetings, thanks, etc.) and return the category."""
    cleaned = query.strip()
    for category, pattern in CASUAL_PATTERNS.items():
        if pattern.match(cleaned):
            return category
    return None


def get_casual_response(category: str) -> str:
    """Return a random response for a casual category."""
    import random
    return random.choice(CASUAL_RESPONSES[category])


def classify_query(query: str) -> tuple[bool, float]:
    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        raw = response.choices[0].message.content.strip()
        result = raw.upper().strip()

        if "NOT_EDUCATIONAL" in result or "NOT EDUCATIONAL" in result:
            return False, 0.95
        elif "EDUCATIONAL" in result:
            return True, 0.95
        else:
            match = re.search(r"EDUCATIONAL|NOT_EDUCATIONAL", result)
            if match:
                return match.group() == "EDUCATIONAL", 0.9
            return True, 0.5

    except Exception as e:
        print(f"[QueryClassifier] Error: {e}")
        return True, 0.5
