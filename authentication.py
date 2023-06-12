import librosa 
import numpy as np

def extract_mfcc(audio_path):
    # Lade Audio-Datei und berechne MFCCs
    audio, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(audio, sr=44, n_mfcc=13)

    # Berechne den Durchschnitt entlang der Zeitachse
    mfcc_mean = np.mean(mfcc, axis=1)

    # Normalisiere den Durchschnitt auf den Bereich von 0 bis 1
    mfcc_mean_normalized = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))

    return mfcc_mean_normalized


