import librosa
import numpy as np
import noisereduce as nr
from librosa.effects import trim
import pandas as pd

TARGET_SR = 16000
RMS_TARGET = 0.05
TRIM_DB = 30
MIN_WAVE_LEN = 1600 

MIN_DURATION = 0.5 # seconds
MAX_SILENCE_PCT = 0.7 # 70%
MIN_SNR = 20 # dB

def estimate_snr(y, sr, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0
    signal_power = np.mean(rms**2)
    noise_floor = np.percentile(rms**2, 10) 
    snr = 10 * np.log10(signal_power / (noise_floor + 1e-12))
    return snr

def load_and_preprocess_audio(path):
    try:
        # Load (without duration cap for quality checks)
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Duration
        if duration < MIN_DURATION:
            return None, TARGET_SR, f"Audio is too short. Please upload/record at least {MIN_DURATION}s."

        # Silence Pct 
        # We'll trim the audio and see how much was left
        y_trimmed, trim_indices = trim(y, top_db=TRIM_DB)
        
        # If nothing is left after trimming, it's all silence
        if len(y_trimmed) == 0:
            return None, TARGET_SR, "Audio is completely silent. Please record again."
            
        trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)
        silence_pct = 1.0 - (trimmed_duration / duration)
        
        if silence_pct > MAX_SILENCE_PCT:
            return None, TARGET_SR, f"Audio is mostly silent ({silence_pct*100:.0f}%). Please record again."

        # SNR (Noise) 
        snr = estimate_snr(y, sr=sr)
        if snr < MIN_SNR:
            return None, TARGET_SR, f"Audio is too noisy (SNR: {snr:.1f} dB). Please record in a quieter place."

        # Check length of trimmed
        if len(y_trimmed) < MIN_WAVE_LEN:
            return None, TARGET_SR, "No speech detected after trimming silence."

        # Apply noise reduction
        y_reduced = nr.reduce_noise(y=y_trimmed, sr=sr)
        y_reduced = np.nan_to_num(y_reduced)

        # Normalize RMS loudness
        rms = np.sqrt(np.mean(y_reduced**2))
        if rms > 1e-5: # Avoid division by zero
            y_normalized = y_reduced * (RMS_TARGET / rms)
        else:
            y_normalized = y_reduced

        return y_normalized, TARGET_SR, None 

    except Exception as e:
        print(f"Error processing audio file {path}: {e}")
        return None, TARGET_SR, "Error: Could not read the audio file."

def pad_or_crop_audio(y, target_length):
    if len(y) > target_length:
        # Center crop
        start = (len(y) - target_length) // 2
        y = y[start : start + target_length]
    elif len(y) < target_length:
        # Pad with zeros
        padding = np.zeros(target_length - len(y))
        y = np.concatenate([y, padding])
    
    return y

def extract_acoustic_features(y, sr):

    try:
        # RMS Energy (Volume) 
        rms_mean = np.mean(librosa.feature.rms(y=y))
        
        # Pitch 
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        # Select pitches with non-zero magnitude
        pitch_vals = pitches[mags > np.median(mags)]
        pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
        
        return {
            "Average Pitch (Hz)": f"{pitch_mean:.2f}",
            "RMS Energy (Volume)": f"{rms_mean:.4f}",
        }
    except Exception as e:
        print(f"Error extracting acoustic features: {e}")
        return {"Error": "Could not calculate acoustic features."}