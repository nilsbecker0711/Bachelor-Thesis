# Larisa
Voice assistant - education project for 
[applied AI](https://github.com/hsu-ai-course/hsu.ai/) course.


Initially designed for Raspberry Pi model B (the old one) this project contains some derived limitations and hacks:
- run `install_pyaudio_with_patch.sh` with `su` rights to make pyaudio work in Raspbian OS (according to [this](https://stackoverflow.com/questions/59006083/how-to-install-portaudio-on-pi-properly))
- not sure you can make it run in Windows...

## Install guide
1. Copy `config.ini.template` to `config.ini`. Tune `config.ini`. What is important? If you use voice, choose correct playback and recording devices. Call `sounddevice.query_devices()` in python to learn their ids. For voice recognition use services, Wit and IBM are easy to setup.
2. Specify host IP and port if you use web version (coming soon).
3. Run `install_pyaudio_with_patch.sh` with superuser to make pyaudio work in Raspbian.
4. Run `setup.sh` to install binary dependencies.
6. Install `pip3 install -r requirements.txt` to bring python3 dependencies.

## Run
Launch `python3 larisa.py`. By default the system triggers to `John` name. Say or type `John` ... wait ... `stop` to finish.
