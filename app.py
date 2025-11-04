import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
from collections import Counter 


import model_utils
import processing

st.set_page_config(page_title="Speech Confidence Analyzer", page_icon="🎤", layout="centered")
st.title("🎙️ Speech Confidence Analyzer")
st.markdown("""
Upload or record an audio clip.
The model (fine-tuned from **HuBERT**) will analyze your tone and classify it as:
- **Confident**
- **Nervous**
- **Monotonous**
""")

# --- Load models ---
feature_extractor, model, le = model_utils.load_model_and_processor()

# --- State management ---
if 'audio_source' not in st.session_state:
    st.session_state.audio_source = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'recorder_file' not in st.session_state:
    st.session_state.recorder_file = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📁 Upload Audio"

# Function to clean up old temp files
def cleanup_temp_file():
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.remove(st.session_state.temp_file_path)
        except Exception as e:
            st.warning(f"Could not remove old temp file: {e}")
    st.session_state.audio_source = None
    st.session_state.temp_file_path = None
    st.session_state.recorder_file = None

# Proceed only if models loaded
if model is not None and feature_extractor is not None and le is not None:
    active_tab = st.radio(
        "Choose input method",
        ["📁 Upload Audio", "🎤 Record Voice"],
        horizontal=True,
        label_visibility="collapsed",
        key='active_tab'
    )
    st.markdown("---")

    # --- Upload ---
    if active_tab == "📁 Upload Audio":
        uploaded_file = st.file_uploader("Upload your voice clip", type=["wav", "mp3", "flac"])
        if uploaded_file is not None:
            cleanup_temp_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                st.session_state.temp_file_path = tmp.name
            st.session_state.audio_source = st.session_state.temp_file_path
            st.session_state.recorder_file = None
            st.audio(st.session_state.audio_source)

    # --- Record from mic ---
    if active_tab == "🎤 Record Voice":
        if not st.session_state.recorder_file:
            st.info("👇 Click the button below to start recording your voice.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                audio_bytes = audio_recorder(
                    text="🎙️ Start Recording",
                    recording_color="#e74c3c",
                    neutral_color="#2ecc71",
                    icon_name="microphone",
                    pause_threshold=30.0,
                    sample_rate=16000,
                )
        else:
            audio_bytes = None

        if audio_bytes:
            cleanup_temp_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                st.session_state.temp_file_path = tmp.name
            st.session_state.audio_source = st.session_state.temp_file_path
            st.session_state.recorder_file = st.session_state.temp_file_path
            st.rerun()

        if st.session_state.recorder_file and st.session_state.audio_source == st.session_state.recorder_file:
            st.success("✅ Recording finished! Listen below or discard.")
            st.audio(st.session_state.audio_source)
            if st.button("Discard & Record Again"):
                cleanup_temp_file()
                st.rerun()

    # Analyze button
    if st.session_state.audio_source:
        if st.button("🔎 Analyze My Speech"):
            audio_path_to_analyze = st.session_state.audio_source
            result = None
            waveform = None
            
            with st.spinner("Preprocessing audio (trimming, denoising, quality check... 🎧)"):
                waveform, sr, error_message = processing.load_and_preprocess_audio(audio_path_to_analyze)

            if error_message:
                st.error(f"**Analysis Failed:** {error_message}")            

            elif waveform is None:
                st.error("An unknown error occurred during preprocessing. Please try again.")
            
            else:
                st.subheader("1. Preprocessing & Feature Extraction")
                st.caption("Processed Audio (Denoised & Trimmed)")
                st.audio(waveform, sample_rate=sr)

                st.caption("Acoustic Features (for demo)")
                with st.spinner("Calculating acoustic features..."):
                    acoustic_feats = processing.extract_acoustic_features(waveform, sr)
                
                cols = st.columns(2)
                cols[0].metric("Average Pitch (Hz)", acoustic_feats.get("Average Pitch (Hz)", "N/A"))
                cols[1].metric("RMS Energy (Volume)", acoustic_feats.get("RMS Energy (Volume)", "N/A"))

                st.subheader("2. Confidence Analysis")
                with st.spinner("Analyzing your tone..."):
                    result = model_utils.predict_audio(waveform, sr, model, feature_extractor, le)

                if result:
                    label = result["final_label"]
                    conf = result["confidence"]
                    probs = result["avg_probs"]
                    
                    st.success(f"**Overall, you sound {label}!**")
                    if label.lower() == "confident":
                        st.info("**Great job!** Your overall tone reflects clarity and assertiveness.")
                    elif label.lower() == "nervous":
                        st.warning("**Keep practicing!** Your tone was identified as nervous. Try slowing down, breathing steadily, and projecting your voice.")
                    elif label.lower() == "monotonous":
                        st.warning("**Try to engage more!** Your tone sounds monotonous. Focus on adding variation in pitch and energy to sound more engaging.")

                    st.metric(f"Overall {label.capitalize()} Probability", f"{conf:.2f}%")

                    st.subheader("Full Probability Breakdown")
                    classes = list(le.classes_)
                    prob_df = pd.DataFrame({
                        "Tone": classes,
                        "Probability": [f"{p*100:.1f}%" for p in probs]
                    })
                    st.dataframe(
                        prob_df, 
                        hide_index=True, 
                        width='stretch'
                    )

                    if result["chunk_predictions"] is not None:
                        st.subheader("Analysis Over Time")
                        st.caption("Your audio was split into 8-second chunks. Here is the breakdown of how often each tone appeared.")
                        
                        chunk_labels = [le.classes_[p] for p in result["chunk_predictions"]]
                        total_chunks = len(chunk_labels)
        
                        counts = Counter(chunk_labels)
                        
                        num_labels = len(le.classes_)
                        cols = st.columns(num_labels)
                        
                        for i, class_label in enumerate(le.classes_):
                            percent = (counts.get(class_label, 0) / total_chunks) * 100
                            cols[i].metric(class_label, f"{percent:.1f}%")
            # Clean up and reset
            cleanup_temp_file()
            st.info("Analysis complete. Upload or record a new file.")