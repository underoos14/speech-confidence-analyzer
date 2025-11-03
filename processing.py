import librosa
import numpy as np
import noisereduce as nr
from librosa.effects import trim

TARGET_SR = 16000
RMS_TARGET = 0.05  
TRIM_DB = 30       
MIN_WAVE_LEN = 1600 

def load_and_preprocess_audio(path):
    """
    Loads, resamples, and applies all robust preprocessing steps
    from your EDA notebook (File 1).
    
    Returns:
        (waveform, sr) or (None, None) if audio is invalid.
    """
    try:
        #Load and resample
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        #Trim silence 
        y_trimmed, _ = trim(y, top_db=TRIM_DB)
        
        #Check for empty audio
        if len(y_trimmed) < MIN_WAVE_LEN:
            print("Audio too short after trimming.")
            return None, TARGET_SR

        # Apply noise reduction
        y_reduced = nr.reduce_noise(y=y_trimmed, sr=sr)
        y_reduced = np.nan_to_num(y_reduced)

        # Normalize RMS loudness
        rms = np.sqrt(np.mean(y_reduced**2))
        if rms > 1e-5: # Avoid division by zero
            y_normalized = y_reduced * (RMS_TARGET / rms)
        else:
            y_normalized = y_reduced

        return y_normalized, TARGET_SR

    except Exception as e:
        print(f"Error processing audio file {path}: {e}")
        return None, TARGET_SR

def pad_or_crop_audio(y, target_length):
    """
    Ensures an audio clip is exactly target_length.
    - If longer: center-crops
    - If shorter: pads with zeros
    
    This fixes the "random crop" bug.
    """
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
        
        # --- Pitch ---
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        # Select pitches with non-zero magnitude
        pitch_vals = pitches[mags > np.median(mags)]
        pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
        
        # --- MFCCs (a few for demo) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Get the mean of the 2nd, 3rd, and 4th MFCCs
        mfcc_mean = np.mean(mfcc[1:4], axis=1)
        
        return {
            "Average Pitch (Hz)": f"{pitch_mean:.2f}",
            "RMS Energy (Volume)": f"{rms_mean:.4f}",
            "MFCC_2_Mean": f"{mfcc_mean[0]:.2f}",
            "MFCC_3_Mean": f"{mfcc_mean[1]:.2f}",
            "MFCC_4_Mean": f"{mfcc_mean[2]:.2f}"
        }
    except Exception as e:
        print(f"Error extracting acoustic features: {e}")
        return {"Error": "Could not calculate acoustic features."}