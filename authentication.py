import librosa 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pandas as pd
import joblib
import os
from datetime import datetime

dirname = os.path.dirname(__file__)

def extract_mfcc(audio_path):
    '''
    This method extracts the mel frequency cepstral coefficients (mfcc) from a given path to an audio file
    :param audio_path: path to audio file in .wav format
    :return: nbormalized frequency cepstral coefficients, sampling rate and audiodata
    '''
    audio, sr = librosa.load(audio_path)
    #audio = np.frombuffer(audio, dtype=np.int16)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)

    mfcc_mean = np.mean(mfcc, axis=1)

   
    mfcc_mean_normalized = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))
    #print(mfcc_mean_normalized)
    return mfcc_mean_normalized, sr, audio


def plot_mel_spectrogram(audio_path):
    
    '''
    Shows a visual representation of Mel Frequencys, by first calculating and then plotting them
    :param: audio_path path to audio file
    '''
   
    mfcc_mean_normalized, sr, audio = extract_mfcc(audio_path)


    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=2048, hop_length=512, n_mels=128, fmax=sr/2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    #Plotting
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def save_classifier(classifier):
    '''
    This method saves a given classifier to the models subfolder with current datetime as destinction method. The datetime format is D/M/Y_H_M
    :param classifier: classifier to save (As it is only SVC)
    :return: path to file
    '''

    try:
       now = datetime.now()
       date = now.strftime("%d_%m_%Y_%H_%M")
       filename = os.path.join(dirname, f'models/svc_model{date}.jl')
       joblib.dump(classifier, filename )
       return filename
    except Exception as e:
         print(e)
         return None
    
def load_classifier(path, from_models=False):
    '''
   Load a pre-trained classifier
   :param path: path to the classifier
   :param from_models: if True the model to be loaded is from the models subfolder allowing a shorter path to be given
   :return: the pre-trained classifier or None if an Exception is raised 
    '''

    if from_models:
        path = os.path.join(dirname,f"models/{path}")   
    try:
        classifier=joblib.load(path)
        return classifier
    except Exception as e:
        print(e)
        return None


def train_gender_classification(data_path, audio_path):
    '''
    Trains a SVC to differentiate between male and female speakers.
    :param data_path: link to an excel file containing paths to audio files and the speakers gender
    :param audio_path: path to folder, where audio files are stored
    :return: trained gender classifier
    '''
    data = os.path.join(dirname, data_path)
    voice_samples = pd.read_excel(data, usecols=[1,6])
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
  

    #Actual Training
    classifier = train_classifier(features, genders)
    return classifier

def train_speaker_classification(data_path, audio_path):
    '''
    Trains a SVC to differentiate between speakers.
    :param data_path: link to an excel file containing paths to audio files and distict speaker ID's
    :param audio_path: path to folder, where audio files are stored
    :return: trained gender classifier
    '''
    data = os.path.join(dirname, data_path)
    voice_samples = pd.read_excel(data, usecols=[0,1])
    speakers = []
    features = []
    i=0
    print(f"Feature Extraction started: {datetime.now()}")
    for index, sample in voice_samples.iterrows():
        if i==10:
            pass
        speaker_ID, audio_path_detail = sample
        mfcc_features, sr, path = extract_mfcc(audio_path+audio_path_detail)
        speakers.append(speaker_ID)
        features.append(mfcc_features)  
        i=i+1    
    print(f"Feature Extraction completed: {datetime.now()}")

    classifier = train_classifier(features, speakers)
    return classifier


def train_classifier(features, criteria):

    X_train, X_test, y_train, y_test = train_test_split(features, criteria, test_size=0.2)
 
    classifier = SVC()

    param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf']
    }

    print(f"Parameter Search Started (Using all CPU Cores in parallel): {datetime.now()}")
    grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Parameter Search Completed: {datetime.now()}")

    params = grid_search.best_params_
    print(f'Beste Parameter: {params}')

   
    classifier = SVC(C=params['C'], kernel=params['kernel'])
    print("Training started")
    classifier.fit(X_train, y_train)
    print("Training finished")
    accuracy = classifier.score(X_test, y_test)
    print("Classifier Accuracy: {:.2f}%".format(accuracy * 100))

    return classifier

def predict_single_gender(classifier, audio_path):
    '''
    Predict for a single sample if the person is a male or female based on a given trained classifier
    :param classifier: a prerained classifier 
    :param audio_path: path to the audio file to be classified
    :return: If no exception is raised -> A String with the name of the sample and the predicted gender, True
             If an exception is raised -> A String with a short notice of the failed prediction, False
    '''
    new_mfcc_features,_,_ = extract_mfcc(audio_path)
    try:
        predicted_label1 = classifier.predict([new_mfcc_features])
        return(f"{audio_path} is a sample of a {predicted_label1[0]}", True)
  
    except Exception as e:
        print(e)
        return("Prediction failed", False)

def predict_single_speaker(classifier, audio_path):
    '''
    Predict for a single sample who the speaker is
    :param classifier: a prerained classifier 
    :param audio_path: path to the audio file to be classified
    :return: If no exception is raised -> A String with the name of the sample and the predicted speaker, True
             If an exception is raised -> A String with a short notice of the failed prediction, False
    '''
    new_mfcc_features,_,_ = extract_mfcc(audio_path)
    try:
        predicted_label1 = classifier.predict([new_mfcc_features])
        return(f"{audio_path} is a sample of a {predicted_label1[0]}", True)
  
    except Exception as e:
        print(e)
        return("Prediction failed", False)


#model1 = save_classifier(train_speaker_classification("samples\commonvoice\info\Filtered.xlsx", "samples/commonvoice/"))  
#print(model1)
classifier = load_classifier("svc_model10_07_2023_21_08.jl", from_models=True)
print(predict_single_speaker(classifier, "samples\cloned\Sample_Nils_1.wav"))