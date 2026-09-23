import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Streamlit page configuration
# MUST be before any other st command
# --------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📝",
    layout="centered"
)


# --------------------------------------------------
# Load model, tokenizer and max length
# --------------------------------------------------
@st.cache_resource
def load_files():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_files()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_next_word(text):

    # Convert input text into sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Check if words are present in tokenizer
    if len(sequence[0]) == 0:
        return None

    # Pad sequence
    sequence = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="pre"
    )

    # Predict
    prediction = model.predict(sequence, verbose=0)

    # Get index of highest probability
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Convert index back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return None


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📝 Next Word Predictor")

st.write(
    "Enter a sentence and the LSTM model will predict the next word."
)

st.divider()


text = st.text_input(
    "Enter your text:",
    placeholder="Example: I am going"
)


if st.button("Predict Next Word"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        next_word = predict_next_word(text)

        if next_word:

            st.success(
                f"Predicted Next Word: **{next_word}**"
            )

            st.write("### Complete Sentence")

            st.write(
                f"{text} **{next_word}**"
            )

        else:

            st.error(
                "The entered words were not found in the tokenizer."
            )


st.divider()

st.caption("LSTM-based Next Word Prediction")

