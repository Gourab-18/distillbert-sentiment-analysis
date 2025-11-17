#!/usr/bin/env python3
"""
Task 9: Streamlit UI Development & Interactive Demo

A user-friendly web interface for the DistilBERT sentiment analysis model
with integrated SHAP explanations for model interpretability.

Features:
- Real-time sentiment prediction with confidence scores
- Token-level SHAP explanations showing word importance
- Interactive visualization of model decisions
- Example inputs for quick testing
- Clean, professional UI design

Usage:
    streamlit run app_streamlit.py

    Then open http://localhost:8501 in your browser
"""

import streamlit as st
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from PIL import Image
import io
import time


# Page configuration
st.set_page_config(
    page_title="DistilBERT Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load the fine-tuned DistilBERT model and create SHAP explainer."""
    print("Loading DistilBERT model...")
    model_path = "models/distilbert-sentiment/best_model"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Use MPS if available (Apple Silicon), otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Create pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        top_k=None,
        truncation=True,
        max_length=512
    )

    # Create SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(sentiment_pipeline)

    print("✅ Model and explainer loaded successfully!")
    return model, tokenizer, sentiment_pipeline, explainer


def predict_sentiment(text: str, pipeline):
    """
    Predict sentiment for input text.

    Args:
        text: Input text to analyze
        pipeline: Sentiment analysis pipeline

    Returns:
        Tuple of (label, confidence, positive_prob, negative_prob)
    """
    if not text or len(text.strip()) == 0:
        return None, 0.0, 0.0, 0.0

    # Get prediction
    outputs = pipeline(text[:512])  # Truncate to max length

    # Parse results
    if isinstance(outputs[0], list):
        probs = {item['label']: item['score'] for item in outputs[0]}
    else:
        probs = {outputs[0]['label']: outputs[0]['score']}

    # Extract probabilities
    positive_prob = probs.get('LABEL_1', probs.get('POSITIVE', 0.5))
    negative_prob = probs.get('LABEL_0', probs.get('NEGATIVE', 1 - positive_prob))

    # Determine label and confidence
    if positive_prob > negative_prob:
        label = "Positive"
        confidence = positive_prob
    else:
        label = "Negative"
        confidence = negative_prob

    return label, confidence, positive_prob, negative_prob


def generate_shap_visualization(text: str, explainer):
    """
    Generate SHAP explanation visualization for the input text.

    Args:
        text: Input text to explain
        explainer: SHAP explainer

    Returns:
        PIL Image of SHAP visualization
    """
    if not text or len(text.strip()) == 0:
        return None

    # Truncate text if too long
    words = text.split()
    if len(words) > 400:
        text = ' '.join(words[:400])

    try:
        # Generate SHAP values
        shap_values = explainer([text])

        # Create custom visualization
        fig, ax = plt.subplots(figsize=(14, 3))

        # Get tokens and SHAP values for positive class
        tokens = shap_values.data[0]
        values = shap_values.values[0, :, 1]  # Positive class

        # Create color map: blue (negative) to red (positive)
        cmap = LinearSegmentedColormap.from_list('shap', ['#3b82f6', '#ffffff', '#ef4444'])

        # Normalize values for coloring
        abs_max = max(abs(values.min()), abs(values.max())) if len(values) > 0 else 1
        if abs_max == 0:
            abs_max = 1

        # Plot tokens with background colors
        x_pos = 0
        for i, (token, value) in enumerate(zip(tokens, values)):
            # Normalize value to [-1, 1]
            norm_value = value / abs_max

            # Get color
            color = cmap(0.5 + norm_value * 0.5)

            # Draw token with background
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color,
                            edgecolor='none', alpha=0.7)
            ax.text(x_pos, 0.5, token, fontsize=11, ha='left', va='center',
                   bbox=bbox_props, family='monospace')

            # Update position
            token_width = len(token) * 0.011 + 0.02
            x_pos += token_width

        # Set axis properties
        ax.set_xlim(-0.05, x_pos + 0.05)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add title
        ax.text(0, 0.95, 'Token Importance (Red = Positive, Blue = Negative)',
               fontsize=10, fontweight='bold', va='top')

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)

    except Exception as e:
        st.error(f"Error generating SHAP visualization: {e}")
        return None


# Main app
def main():
    # Header
    st.title("🎭 DistilBERT Sentiment Analysis")
    st.markdown("""
    Analyze the sentiment of any text using a fine-tuned DistilBERT model.
    Get instant predictions with **token-level explanations** powered by SHAP.

    **Model Performance:** 92.65% accuracy on IMDb test set | Trained on 20K movie reviews
    """)

    # Load model (cached)
    with st.spinner("Loading model and SHAP explainer..."):
        model, tokenizer, sentiment_pipeline, explainer = load_model()

    # Sidebar
    with st.sidebar:
        st.header("💡 Example Inputs")
        st.markdown("Click an example to use it:")

        examples = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "Terrible experience. The service was slow and the food was cold. Would not recommend.",
            "The product works as expected. Nothing special but does the job.",
            "I'm blown away by how good this is! Exceeded all my expectations. Highly recommended!",
            "Waste of money. Poor quality and broke after one use. Very disappointed.",
        ]

        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.text_input = example

        st.markdown("---")
        st.markdown("""
        ### About
        - **Architecture:** DistilBERT (distilbert-base-uncased)
        - **Training Data:** IMDb movie reviews (25K samples)
        - **Test Accuracy:** 92.65%
        - **Improvement:** +4.87 percentage points

        **Technology Stack:** Transformers · PyTorch · SHAP · Streamlit
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter Your Text")
        text_input = st.text_area(
            "Type or paste text here to analyze sentiment...",
            value=st.session_state.get('text_input', ''),
            height=150,
            key="main_text_input"
        )

        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")

    with col2:
        st.subheader("Results")
        result_container = st.container()

    # Analyze sentiment
    if analyze_btn and text_input:
        start_time = time.time()

        with st.spinner("Analyzing sentiment..."):
            # Get prediction
            label, confidence, pos_prob, neg_prob = predict_sentiment(text_input, sentiment_pipeline)

            if label:
                with result_container:
                    # Display sentiment
                    if label == "Positive":
                        st.success(f"😊 **{label}**")
                    else:
                        st.error(f"😞 **{label}**")

                    # Display confidence
                    st.metric("Confidence", f"{confidence:.1%}")

                    # Display probabilities
                    st.markdown("**Probability Distribution:**")
                    st.progress(pos_prob, text=f"Positive: {pos_prob:.1%}")
                    st.progress(neg_prob, text=f"Negative: {neg_prob:.1%}")

                elapsed_time = time.time() - start_time
                st.info(f"⏱️ Analysis completed in {elapsed_time:.2f} seconds")

        # Generate SHAP visualization
        st.markdown("---")
        st.subheader("🔬 Model Explanation (SHAP)")
        st.markdown("""
        See which words influenced the model's prediction:
        - 🔴 **Red** tokens push toward **Positive** sentiment
        - 🔵 **Blue** tokens push toward **Negative** sentiment
        - Intensity indicates strength of influence
        """)

        with st.spinner("Generating SHAP explanation..."):
            shap_image = generate_shap_visualization(text_input, explainer)
            if shap_image:
                st.image(shap_image, use_container_width=True)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze")

    # Footer
    st.markdown("---")
    st.markdown("""
    *Built as part of a sentiment analysis project demonstrating NLP best practices.*
    """)


if __name__ == "__main__":
    main()
