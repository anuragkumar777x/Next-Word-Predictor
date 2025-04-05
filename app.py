import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Streamlit page config
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

model = load_model('next_word_lstm.h5')

with open('mytokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predicts the next 'n' words based on the input text
def predict_next_n_words(model, tokenizer, text, max_sequence_len, n=1):
    for _ in range(n):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                next_word = word
                break
        
        text += " " + next_word
    return text

# UI styling and profile section
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Google Sans', 'Roboto', sans-serif;
            background: linear-gradient(to right, #f3f4f6, #e8f0fe);
            color: #202124;
        }

        .header {
            text-align: center;
            padding: 20px 20px 10px 20px;
            margin-bottom: 10px;
        }

        .header h1 {
            font-size: 3.5rem;
            color: #1a73e8;
            font-weight: 800;
        }

        .header p {
            font-size: 1.5rem;
            color: #5f6368;
        }

        .typing-box {
            padding: 40px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        .prediction-text {
            background: #d2e3fc;
            color: #174ea6;
            font-weight: 700;
            padding: 20px 28px;
            border-radius: 16px;
            font-size: 26px;
            margin-top: 30px;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }

        .footer {
            margin-top: 70px;
            text-align: center;
            font-size: 18px;
            color: #5f6368;
        }

        .footer a {
            color: #1a73e8;
            text-decoration: none;
            font-weight: bold;
        }

        .footer img {
            vertical-align: middle;
            margin-right: 6px;
        }

        .stSelectbox label, .stTextInput label {
            color: #202124 !important;
            font-weight: 700;
            font-size: 24px;
            margin-bottom: 10px;
        }

        .stTextInput input {
            background-color: #ffffff !important;
            color: #202124;
            font-size: 24px;
            border-radius: 12px;
            border: 1px solid #dadce0;
            padding: 16px;
        }

        .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border-radius: 12px;
            border: 1px solid #dadce0;
        }

        .stTextInput:after {
            content: "Press Enter to apply";
            display: block;
            color: #5f6368;
            font-size: 16px;
            margin-top: 6px;
            font-style: italic;
            font-weight: 500;
        }

        @media screen and (max-width: 768px) {
            .stTextInput input {
                font-size: 20px;
                padding: 14px;
            }

            .stTextInput:after {
                font-size: 14px;
            }

            .stSelectbox label, .stTextInput label {
                font-size: 20px;
            }
        }
    </style>

    <div class="header">
        <h1>Next Word Predictor</h1>
        <p>Predictive typing made simple with AI</p>
    </div>
""", unsafe_allow_html=True)

num_words = st.selectbox("How many words do you want to predict?", options=[1, 2, 3, 4, 5], index=2)

input_text = st.text_input("Start typing your sentence:")

if input_text.strip():
    max_sequences_len = model.input_shape[1] + 1
    prediction = predict_next_n_words(model, tokenizer, input_text, max_sequences_len, n=num_words)
    st.markdown(f"""
        <div class='prediction-text'>
            💡 <span style="font-weight: 600; font-family: 'Segoe UI', sans-serif;">Suggestion:</span> {prediction}
        </div>
    """, unsafe_allow_html=True)

# Footer/Profile section
st.markdown("""
    <div class="footer">
        Built by <strong><a href="https://github.com/anuragkumar777x" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" height="20" style="vertical-align: middle; margin-right: 5px;">Anurag</a></strong>
    </div>
""", unsafe_allow_html=True)