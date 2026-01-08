import tkinter as tk
from tkinter import ttk
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

    # Greetings
    if cleaned_input in ["hi", "hello", "hey"]:
        return "Hello! 😊 How can I help you with the CodeAlpha internship?"

    # Time-based greetings
    if cleaned_input in ["good morning", "morning"]:
        return "Good morning ☀️ Hope you have a great day!"

    if cleaned_input in ["good evening", "evening"]:
        return "Good evening 🌆 How can I assist you?"

    if cleaned_input in ["good night", "night"]:
        return "Good night 🌙 Take care!"

    # Thanks
    if cleaned_input in ["thanks", "thank you", "thankyou"]:
        return "You're welcome! 😊 Happy to help."

    # Goodbye
    if cleaned_input in ["bye", "goodbye", "see you"]:
        return "Goodbye 👋 Wishing you all the best!"

    # FAQ matching
    user_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    index = similarity.argmax()

    if similarity[0][index] < 0.2:
        return "Sorry, I couldn't understand your question."
    return answers[index]
# ---------- TKINTER UI ----------
root = tk.Tk()
root.title("CodeAlpha FAQ Chatbot")
root.geometry("520x600")
root.configure(bg="#f4f6f8")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("TEntry", font=("Segoe UI", 11))

header = tk.Label(
    root,
    text="🤖 CodeAlpha FAQ Chatbot",
    font=("Segoe UI", 16, "bold"),
    bg="#4f46e5",
    fg="white",
    pady=12
)
header.pack(fill="x")

chat_frame = tk.Frame(root, bg="#f4f6f8")
chat_frame.pack(expand=True, fill="both", padx=10, pady=10)

chat_area = tk.Text(
    chat_frame,
    wrap="word",
    state="disabled",
    font=("Segoe UI", 11),
    bg="white"
)
chat_area.pack(expand=True, fill="both")

input_frame = tk.Frame(root, bg="#f4f6f8")
input_frame.pack(fill="x", padx=10, pady=10)

user_input = ttk.Entry(input_frame)
user_input.pack(side="left", expand=True, fill="x", padx=(0, 10))

send_btn = ttk.Button(input_frame, text="Send")
send_btn.pack(side="right")

# ---------- CHAT FUNCTION ----------
def send_message(event=None):
    msg = user_input.get().strip()
    if not msg:
        return

    chat_area.config(state="normal")
    chat_area.insert("end", f"You: {msg}\n")
    reply = chatbot_response(msg)
    chat_area.insert("end", f"Bot: {reply}\n\n")
    chat_area.config(state="disabled")
    chat_area.see("end")

    user_input.delete(0, "end")

send_btn.config(command=send_message)
user_input.bind("<Return>", send_message)

# ---------- WELCOME MESSAGE ----------
chat_area.config(state="normal")
chat_area.insert(
    "end",
    "Bot: Hello! Ask me anything about the CodeAlpha internship.\n\n"
)
chat_area.config(state="disabled")

root.mainloop()
