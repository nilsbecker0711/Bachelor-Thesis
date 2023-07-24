import speech_recognition
import sounddevice
import svc_tuning as svc
import numpy as np
import wave
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


while True:
    print("start")
    try:
        with speech_recognition.Microphone() as mic:
            #recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            #print(audio.sample_rate)
            
            name = 'temp.wav'
            with open(name, "wb") as f:
                f.write(audio.get_wav_data())
            pred = svc.predict_single_speaker(classifiers=classifiers, clone_classifier=clone_clf, audio_path=name)
            print(pred)
            #os.remove(name)
            
            text = recognizer.recognize_google(audio)
            text = text.lower()         
            print(text)
            
            if text == 'exit':
                break
            
    except Exception as e:
        recognizer = speech_recognition.Recognizer()
        print(e)
        continue
    break