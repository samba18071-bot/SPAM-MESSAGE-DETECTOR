"""
app.py
Streamlit app for Spam Message Detection using two pre-trained pickle files:
  - spam.pkl        -> the trained MultinomialNB classifier
  - vectorizer.pkl   -> the fitted TfidfVectorizer/CountVectorizer used in training

The classifier alone cannot accept raw text -- it needs the SAME fitted
vectorizer used at training time to convert text into numeric features first.
"""

import os
import pickle

import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Spam Message Detector", page_icon="📩", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ---------------------------------------------------------
# Cached loading of model + vectorizer (loads once, not on every rerun)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


missing = [p for p in (MODEL_PATH, VECTORIZER_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "Missing file(s): " + ", ".join(missing) +
        ".\n\nMake sure both `spam.pkl` and `vectorizer.pkl` are in the same "
        "folder as app.py. See the notebook snippet in the chat to export "
        "vectorizer.pkl if you don't have it yet."
    )
    st.stop()

model, vectorizer = load_artifacts()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ About")
st.sidebar.markdown(
    "This app predicts whether a message is **Spam** or **Ham** "
    "using your pre-trained model (`spam.pkl`)."
)

# Track prediction history across reruns
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("📩 Spam Message Detector")
st.write("Enter a message below to check whether it's **Spam** or **Ham (Not Spam)**.")

with st.form("spam_form"):
    message = st.text_area("✉️ Message text", height=120, placeholder="Type or paste a message here...")
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    if not message.strip():
        st.warning("Please enter a message before predicting.")
    else:
        vect_input = vectorizer.transform([message])
        prediction = model.predict(vect_input)[0]

        # Handle both label styles: 0/1 or "ham"/"spam"
        is_spam = prediction == 1 or str(prediction).lower() == "spam"
        label = "🚨 SPAM" if is_spam else "✅ HAM (Not Spam)"

        if is_spam:
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect_input)[0]
            col1, col2 = st.columns(2)
            col1.metric("Ham confidence", f"{proba[0] * 100:.2f}%")
            col2.metric("Spam confidence", f"{proba[1] * 100:.2f}%")

        st.session_state.history.append({
            "message": message,
            "prediction": "Spam" if is_spam else "Ham"
        })

# ---------------------------------------------------------
# Prediction history
# ---------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("🕘 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    csv_data = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history as CSV", csv_data, "prediction_history.csv", "text/csv")

# ---------------------------------------------------------
# Batch prediction from CSV
# ---------------------------------------------------------
st.divider()
st.subheader("📂 Batch Prediction (CSV Upload)")
st.caption("CSV should have a column named 'message'.")

batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    if "message" not in batch_df.columns:
        st.error("CSV must contain a 'message' column.")
    else:
        vect_batch = vectorizer.transform(batch_df["message"].astype(str))
        preds = model.predict(vect_batch)
        batch_df["prediction"] = [
            "Spam" if (p == 1 or str(p).lower() == "spam") else "Ham" for p in preds
        ]
        st.dataframe(batch_df[["message", "prediction"]], use_container_width=True)

        out_csv = batch_df[["message", "prediction"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download batch predictions", out_csv, "batch_predictions.csv", "text/csv")