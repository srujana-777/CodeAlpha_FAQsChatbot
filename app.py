import streamlit as st
from faqs import faqs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# ---------- TEXT PREPROCESSING ----------
def preprocess(text):
    text = text.lower()
    for ch in string.punctuation:
        text = text.replace(ch, "")
    return text

# ---------- FAQ DATA ----------
questions = list(faqs.keys())
answers = list(faqs.values())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):
    cleaned_input = preprocess(user_input)

    if cleaned_input in ["hi", "hello", "hey"]:
        return "Hello! 😊 How can I help you with the CodeAlpha internship?"

    if cleaned_input in ["good morning", "morning"]:
        return "Good morning ☀️ Hope you have a great day!"

    if cleaned_input in ["good evening", "evening"]:
        return "Good evening 🌆 How can I assist you?"

    if cleaned_input in ["good night", "night"]:
        return "Good night 🌙 Take care!"

    if cleaned_input in ["thanks", "thank you", "thankyou"]:
        return "You're welcome! 😊 Happy to help."

    if cleaned_input in ["bye", "goodbye", "see you"]:
        return "Goodbye 👋 Wishing you all the best!"

    user_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    index = similarity.argmax()

    if similarity[0][index] < 0.2:
        return "Sorry, I couldn't understand your question."

    return answers[index]

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="CodeAlpha FAQ Chatbot", page_icon="🤖")
st.title("🤖 CodeAlpha FAQ Chatbot")
st.write("Ask me anything about the CodeAlpha Internship")

user_input = st.text_input("Type your question:")

if st.button("Send"):
    if user_input:
        reply = chatbot_response(user_input)
        st.success(reply)
    else:
        st.warning("Please enter a question.")
