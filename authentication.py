import librosa 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score
import os

def extract_mfcc(audio_path):
    '''
    This method extracts the mel frequency cepstral coefficients (mfcc) from a given path to an audio file
    :param audio_path: path to audio file in .wav format
    :return: nbormalized frequency cepstral coefficients, sampling rate and audiodata
    '''
    # Lade Audio-Datei und berechne MFCCs
    audio, sr = librosa.load(audio_path)
    #audio = np.frombuffer(audio, dtype=np.int16)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)


    # Berechne den Durchschnitt entlang der Zeitachse
    mfcc_mean = np.mean(mfcc, axis=1)

    # Normalisiere den Durchschnitt auf den Bereich von 0 bis 1
    mfcc_mean_normalized = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))
    #print(mfcc_mean_normalized)
    return mfcc_mean_normalized, sr, audio

def plot_mel_spectrogram(audio_path):
    '''
    Shows a visual representation of Mel Frequencys, by first calculating and then plotting them
    :param: audio_path path to audio file
    '''
   
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

def gender_classification(data_path, audio_path, test_path):
    '''
    Trains a KNN Model to differentiate between male and female speakers.
    :param data_path: link to an excel file containing paths to audio files and the speakers gender
    '''

    voice_samples = pd.read_excel(data_path, usecols=[1,6])
    #print(voice_samples)
    features = []
    genders = []

    for index, sample in voice_samples.iterrows():
        #print(sample)
        audio_path_detail, gender = sample
        mfcc_features, sr, path = extract_mfcc(audio_path+audio_path_detail)
        features.append(mfcc_features)
        genders.append(gender)

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(features, genders, test_size=0.2, random_state=42)

    classifier = SVC()
    classifier.fit(X_train, y_train)
    print("Training finished")
    
    try:
       dirname = os.path.dirname(__file__)
       filename = os.path.join(dirname, 'models/svm_model.jl')
       joblib.dump(classifier, filename )
    except Exception as e:
        print("File not found")
        print(e)


    new_mfcc_features,_,_ = extract_mfcc(test_path)

    # Klassifiziere die neue Audio-Datei
    try:
        predicted_label1 = classifier.predict([new_mfcc_features])
        print(f"{test_path} is a sample of a {predicted_label1[0]}")
  
    except Exception as e:
        print("prediction1 failed")
        print(e)
        

    # Evaluierung auf dem Testset
    accuracy = classifier.score(X_test, y_test)
    print("Classifier Accuracy: {:.2f}%".format(accuracy * 100))



#plot_mel_spectrogram("samples/commonvoice/common_voice_en_37046960.mp3")
gender_classification("samples/commonvoice/info/Filtered.xlsx", "samples/commonvoice/" , "samples/cloned/Sample_Nils_1.wav")

