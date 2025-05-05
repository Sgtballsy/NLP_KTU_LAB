import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nltk.download('punkt')

data = {
    "greeting": [
        "hello", "hi", "hey", "good morning", "good evening"
    ],
    "goodbye": [
        "bye", "see you later", "goodbye", "take care"
    ],
    "thanks": [
        "thanks", "thank you", "much appreciated"
    ],
    "about_bot": [
        "who are you", "what can you do", "tell me about yourself", "what is your name"
    ],
    "python_help": [
        "how to define a function in python", 
        "what is a list comprehension", 
        "explain python loops", 
        "how does a dictionary work"
    ]
}

responses = {
    "greeting": ["Hi there!", "Hello!", "Hey! How can I help you?"],
    "goodbye": ["Goodbye!", "See you soon!", "Take care!"],
    "thanks": ["You're welcome!", "Anytime!", "Glad I could help!"],
    "about_bot": ["I'm a chatbot powered by machine learning!", "I can help you with basic Python and general questions."],
    "python_help": ["In Python, you define a function using 'def function_name():'", 
                    "List comprehensions provide a concise way to create lists.",
                    "Loops in Python include for-loops and while-loops.",
                    "Dictionaries are key-value pairs in Python."]
}

X = []
y = []

for intent, examples in data.items():
    for example in examples:
        X.append(example)
        y.append(intent)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

def chatbot_response(user_input):
    predicted_intent = model.predict([user_input])[0]
    return random.choice(responses[predicted_intent])

print("Bot: Hello! Ask me something (type 'exit' to quit).")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user_input))
