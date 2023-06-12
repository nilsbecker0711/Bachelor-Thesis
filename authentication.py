import librosa 
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc(audio_path):
    # Lade Audio-Datei und berechne MFCCs
    audio, sr = librosa.load(audio_path)
    #audio = np.frombuffer(audio, dtype=np.int16)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    #mfcc = librosa.feature.mfcc()

    # Berechne den Durchschnitt entlang der Zeitachse
    mfcc_mean = np.mean(mfcc, axis=1)

    # Normalisiere den Durchschnitt auf den Bereich von 0 bis 1
    mfcc_mean_normalized = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))
    #print(mfcc_mean_normalized)
    return mfcc_mean_normalized, sr, audio

def plot_mel_spectrogram(audio_path):
    mfcc_mean_normalized, sr, audio = extract_mfcc(audio_path)

    # Berechne das Mel-Spektrogramm
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=2048, hop_length=512, n_mels=128, fmax=sr/2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Zeichne das Mel-Spektrogramm
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

plot_mel_spectrogram("samples/cloned/Sample_Nils_1.wav")


