import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🤖",
    layout="centered"
)

API_URL = "http://fastapi-backend:8000/predict"

# --- UI Elements ---
st.title("Find out the sentiment of a review...")
st.markdown("Enter a review or any text below to classify its sentiment as **Positive**, **Neutral**, or **Negative**.")

# User input text 
with st.form("sentiment_form"):
    user_input = st.text_area("Enter Text:", height=150, placeholder="The product was fantastic and the battery life was superb!")
    submit_button = st.form_submit_button(label="Analyze Sentiment")

# --- Form Submission Logic ---
if submit_button:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            payload = {"rid": "streamlit_user_1", "text": user_input}
            
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()  

                result = response.json()

                # --- Display Results ---
                st.subheader("Analysis Results")
                
                sentiment = result.get("sentiment_word", "unknown").capitalize()
                confidence = result.get("confidence_score", 0)

                # Display sentiment with a colored emoji
                if sentiment == "Positive":
                    st.success(f"**Sentiment: Positive** 👍")
                elif sentiment == "Negative":
                    st.error(f"**Sentiment: Negative** 👎")
                else:
                    st.info(f"**Sentiment: Neutral** 😐")

                # Display confidence score as a progress bar
                st.write("Confidence Score:")
                st.progress(confidence)
                st.write(f"{confidence:.2%}")

                
                with st.expander("Show Detailed Probabilities"):
                    st.bar_chart({
                        "Negative": result.get("negative_prob", 0),
                        "Neutral": result.get("neutral_prob", 0),
                        "Positive": result.get("positive_prob", 0),
                    })
                    st.write(f"**Model Entropy:** {result.get('entropy', 0):.4f}")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the Sentiment Analysis API. Please ensure the backend service is running. Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

