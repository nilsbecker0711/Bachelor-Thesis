import pyaudio
import wave
import numpy as np
import svc_tuning as svc
import os


model_path = os.path.join(svc.dirname, "models/svc_model24_07_2023_14_23")
classifiers = []
names = []
clone_clf = svc.load_classifiers(os.path.join(svc.dirname, "models/svc_model24_07_2023_14_36/clone.jl"))
for filename in os.listdir(model_path):
    if filename[len(filename)-2:] == "jl":#only take models
        classifiers.append(svc.load_classifiers(os.path.join(model_path, filename)))
        names.append(filename)


def record_audio_to_wav(output_filename, max_duration=10, silence_threshold=12, chunk_size=1024, sample_format=pyaudio.paInt16, channels=2, sample_rate=44100):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=sample_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    frames = []
    silent_frames = 0

    print("Recording...")

    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        # Konvertieren Sie das binäre Audio in ein Array von NumPy, um das Stillelevel zu überprüfen
        audio_np = np.frombuffer(data, dtype=np.int16)

        # Wenn das Stillelevel unterhalb des Schwellenwerts liegt, erhöhen Sie den Zähler.
        # Wenn das Stillelevel wieder über den Schwellenwert steigt, setzen Sie den Zähler zurück.
        #print(np.max(np.abs(audio_np)))
        if np.max(np.abs(audio_np)) < silence_threshold:
            silent_frames += 1
            #print("silent")
        else:
            #print("non silent")
            silent_frames = 0
        # Beenden Sie die Aufnahme, wenn die maximale Aufnahmedauer erreicht ist oder eine ausreichende Stille erreicht wurde.
        if ((len(frames) / sample_rate)*1000 >= max_duration) or (silent_frames >= (sample_rate / chunk_size)):
            break

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Speichern Sie die aufgezeichneten Frames in eine WAV-Datei
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(sample_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    output_file = "temp_audio.wav"
    record_audio_to_wav(output_file)
    pred = svc.predict_single_speaker(classifiers=classifiers, clone_classifier=clone_clf, audio_path=output_file)
    print(pred)
