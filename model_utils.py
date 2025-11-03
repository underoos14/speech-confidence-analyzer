import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import joblib
from processing import pad_or_crop_audio

MODEL_DIR = "distilhubert_confidence_fast"
MAX_LENGTH = 16000 * 8  # 8 sec
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_and_processor():
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
        model = HubertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        le = joblib.load("label_encoder.pkl")
        return feature_extractor, model, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure the model files ('distilhubert_confidence_fast') and 'label_encoder.pkl' are in the correct directory.")
        return None, None, None

def predict_audio(waveform, sr, model, feature_extractor, le):
    if len(waveform) > MAX_LENGTH:
        st.info("Running extended analysis...")
        return _analyze_long_audio(waveform, sr, model, feature_extractor, le)
    else:
        st.info("Running standard analysis...")
        return _predict_short_audio(waveform, sr, model, feature_extractor, le)

def _predict_short_audio(waveform, sr, model, feature_extractor, le):    
    # Fix: Pad or center-crop to MAX_LENGTH for deterministic results
    processed_wave = pad_or_crop_audio(waveform, MAX_LENGTH)
    
    # Extract features
    inputs = feature_extractor(processed_wave, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in inputs.items()}).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100
        
    return {
        "final_label": pred_label,
        "confidence": confidence,
        "avg_probs": probs,
        "chunk_predictions": None, # No chunks for short 
    }

def _analyze_long_audio(waveform, sr, model, feature_extractor, le, chunk_duration=8, overlap=1):
    chunk_size = int(chunk_duration * sr)
    step_size = int((chunk_duration - overlap) * sr)
    
    chunks = []
    for start in range(0, len(waveform) - step_size, step_size):
        end = start + chunk_size
        if end > len(waveform):
            end = len(waveform)
        chunk = waveform[start:end]
        
        if len(chunk) < sr * 2: # skip fragments < 2s
            continue
            
        # Pad the chunk if it's shorter than the chunk_size
        processed_chunk = pad_or_crop_audio(chunk, chunk_size)
        chunks.append(processed_chunk)

    if not chunks:
        st.error("Could not split audio into valid chunks.")
        return None

    st.write(f"Audio split into {len(chunks)} overlapping chunks.")

    all_probs = []
    all_preds = []

    for chunk in chunks:
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**{k: v.to(DEVICE) for k, v in inputs.items()}).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)
        all_preds.append(np.argmax(probs))

    if not all_probs:
        st.error("Could not process audio chunks.")
        return None

    # Aggregate results by averaging probabilities
    avg_probs = np.mean(all_probs, axis=0)
    final_idx = np.argmax(avg_probs)
    final_label = le.inverse_transform([final_idx])[0]
    confidence = avg_probs[final_idx] * 100

    return {
        "final_label": final_label,
        "confidence": confidence,
        "avg_probs": avg_probs,
        "chunk_predictions": all_preds,
    }