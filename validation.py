import speech_recognition

import authenticator as svc
import os


recognizer = speech_recognition.Recognizer()

model_path = os.path.join(svc.dirname, "models/svc_model24_07_2023_16_08")
classifiers = []
names = []
clone_clf = svc.load_classifiers(os.path.join(svc.dirname, "models/svc_model24_07_2023_16_17/clone.jl"))
for filename in os.listdir(model_path):
    if filename[len(filename)-2:] == "jl":#only take models
        classifiers.append(svc.load_classifiers(os.path.join(model_path, filename)))
        names.append(filename)
def validate (audio):
    name = 'temp.wav'
    with open(name, "wb") as f:
        f.write(audio.get_wav_data())
    pred = svc.predict_single_speaker(classifiers=classifiers, clone_classifier=clone_clf, audio_path=name)
    return pred
       