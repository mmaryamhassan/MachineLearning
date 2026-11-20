import streamlit as st
import pickle

st.set_page_config(page_title="Real vs Fake News Classifier", layout="wide")
st.title("📰 Real vs. Fake News Classifier (KNN)")

@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("best_knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, knn = load_artifacts()

st.subheader("Test a Headline")
user_input = st.text_input("Enter a news headline below:")

if st.button("Classify Headline"):
    if user_input.strip() == "":
        st.warning("Please enter a valid headline.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = knn.predict(vec)[0]
        prob = knn.predict_proba(vec)[0]

        if prediction == 1:
            st.success(f"**Prediction:** REAL News (Confidence: {prob[1]*100:.1f}%)")
        else:
            st.error(f"**Prediction:** FAKE News (Confidence: {prob[0]*100:.1f}%)")
