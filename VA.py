import configparser
import logging
import logic.processor as prc
import speech_recognition as sr
import sounddevice
import validation as v
    
def start_text_interface(cfg):
    logging.info("Starting text interface. Type `stop` to finish")
    command = None
    while command != "stop":
        print("ask> ", end="")
        command = input()
        result = prc.process(command, cfg, logging)
        print(result)



def start_voice_interface(cfg):
    """
    Starts the voice loop
    :param cfg:
    :return:
    """
    # https://github.com/MycroftAI/mimic1
    # https://pypi.org/project/SpeechRecognition/

    keywords = cfg['audio']['keywords'].split()
    stopword = cfg['audio']['stopword']
    recognizer_type = cfg['audio']['recognizer']
    logging.info(f"Keyword = {keywords}, stopword = {stopword}, Recognizer = {recognizer_type}")
    command = None
    ibm_auth, ibm_recognizer = None, None
    # use this code: https://stackoverflow.com/questions/48777294/python-app-listening-for-a-keyword-like-cortana
    # get audio from the microphone
    r = sr.Recognizer()
    # r.dynamic_energy_threshold = False

    while command != stopword:
        # wait for a trigger word
        with sr.Microphone() as source:
            print("Waiting for a word ...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            # test the data comes
            logging.info(f"Len: {len(audio.frame_data)}, Data: {audio.frame_data[:16]}")
            


        # simple way to replace if you don't have a micro
        # with sr.AudioFile("/mnt/c/dev/voice.wav") as af:
        #     audio = r.record(af)
        
        
        try:
            recognized = r.recognize_google(audio)
           
            logging.info(f"word `{recognized}` captured")
        
            # if this was a keyword - process a command
            if recognized.strip().lower() in keywords:
                validation = v.validate(audio)
                if not validation[1]:
                    logging.error("Cloned Voice Detected! Access Denied")
                    return
                else:
                    ids = cfg['authorized']['id'].split()
                    logging.info(validation[0][0])
                    if f'{validation[0][0]}' not in ids:
                        logging.error("Unauthorized Access!")
                        return
                    logging.info(f'Access Granted for ID {validation[0][0]}')
                print('Waiting for a command...')
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    command_audio = r.listen(source)
                    command = r.recognize_google(command_audio) #test 24.04.
                    command = command.lower() #test 26.04.
                    
                   
                # exit on stop word
                print(command)
                if command == stopword:
                    logging.info('Exit on stopword')
                    break
                result = prc.process(command, cfg, logging)
                # TODO: SPEAK THE RESULT
                logging.info(result)
        except sr.UnknownValueError:
            logging.error("Could not understand audio")
        except sr.RequestError as e:
            logging.error(f"Could not request results: {e}")
        pass


if __name__ == "__main__":
    sounddevice.query_devices()
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    logging.info("Pacifier is starting")
    config = configparser.ConfigParser()
    config.read('config.ini')
    if 'ui' not in config:
        logging.error("Config section [ui] is missing")
        exit(1)
    mode = config['ui']['mode']
    if 'text' == mode:
        start_text_interface(config)
    elif 'voice' == mode:
        start_voice_interface(config)
    else:
        logging.error("unknown ui mode")
        exit(2)
    logging.info("Smoothly finishing")
 
