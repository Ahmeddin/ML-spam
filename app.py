import streamlit as st
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="✉️",
    layout="centered"
)

# 1. Resource Loading Layer
@st.cache_resource
def load_artifacts():
    """Loads the trained model and vectorizer once and caches them to save memory."""
    if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
        return None, None
        
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        
    return vectorizer, model

vectorizer, model = load_artifacts()

# 2. UI Layout
st.title("✉️ Intelligence-Driven Email Spam Classifier")
st.write("Enter the textual body of an email below to analyze whether it is classified as a legitimate message (Ham) or Spam.")
st.markdown("---")

# Verify artifacts are present before rendering functional inputs
if model is None or vectorizer is None:
    st.error("🚨 Missing System Artifacts! Please execute `Create.py` first to train and generate your classification models.")
else:
    # Text input area
    email_input = st.text_area(
        "Email Content text input:", 
        height=200, 
        placeholder="Paste the email content body here..."
    )
    
    # Classify layout button
    if st.button("Analyze & Classify Email", type="primary"):
        cleaned_input = email_input.strip()
        
        if not cleaned_input:
            st.warning("⚠️ Input field empty. Please provide textual content to run analysis.")
        elif len(cleaned_input) < 10:
            st.warning("⚠️ Input string too short. Please provide a more complete email structure for a reliable prediction.")
        else:
            with st.spinner("Executing structural vectorization and inference..."):
                # 3. Vectorization & Inference
                transformed_text = vectorizer.transform([cleaned_input])
                prediction = model.predict(transformed_text)[0]
                probabilities = model.predict_proba(transformed_text)[0]
                
                # Extract specific probability metric based on prediction result
                confidence = probabilities[prediction] * 100

                st.markdown("### **Classification Output Result:**")
                
                # 4. Result Metrics Presentation
                if prediction == 1:
                    st.error(f"🚨 **SPAM DETECTED** (Confidence Level: {confidence:.2f}%)")
                    st.info("💡 *Advice: This email exhibits statistical patterns common in phishing or unsolicited advertisement feeds. Avoid interaction.*")
                else:
                    st.success(f"✅ **HAM / LEGITIMATE EMAIL** (Confidence Level: {confidence:.2f}%)")
                    st.info("💡 *Advice: This text exhibits high-probability characteristics of typical, standard communication.*")