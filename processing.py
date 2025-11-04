import librosa
import numpy as np
import noisereduce as nr
from librosa.effects import trim

TARGET_SR = 16000
RMS_TARGET = 0.05  
TRIM_DB = 30       
MIN_WAVE_LEN = 1600 

MIN_DURATION = 0.5 
MAX_SILENCE_PCT = 0.8 
MIN_SNR = 5

def estimate_snr(y, sr, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, sr=sr, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0
    signal_power = np.mean(rms**2)
    noise_floor = np.percentile(rms**2, 10)
    snr = 10 * np.log10(signal_power / (noise_floor + 1e-12))
    return snr

def load_and_preprocess_audio(path):
    try:
        #Load and resample
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        duration = librosa.get_duration(y=y, sr=sr)
        if duration < MIN_DURATION:
            return None, TARGET_SR, f"Audio is too short. Please upload/record at least {MIN_DURATION}s."

        # Using a simple energy-based silence check
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if rms.size == 0:
             return None, TARGET_SR, "Audio is empty or corrupt."
        
        # Find a low RMS value to count as "silence"
        silence_threshold_rms = np.percentile(rms, 20) 
        silence_pct = np.sum(rms < silence_threshold_rms) / len(rms)
        if silence_pct > MAX_SILENCE_PCT:
            return None, TARGET_SR, f"Audio is mostly silent ({silence_pct*100:.0f}%). Please record again."

        # Check 3: SNR (Noise)
        snr = estimate_snr(y, sr=sr)
        if snr < MIN_SNR:
            return None, TARGET_SR, f"Audio is too noisy (SNR: {snr:.1f} dB). Please record in a quieter place."

        #Trim silence 
        y_trimmed, _ = trim(y, top_db=TRIM_DB)
        
        #Check for empty audio
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

        return y_normalized, TARGET_SR

    except Exception as e:
        print(f"Error processing audio file {path}: {e}")
        return None, TARGET_SR, "Error processing audio file"

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
        
        #  Pitch 
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