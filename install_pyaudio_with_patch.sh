# apt-get install remove libportaudio2
apt-get install libasound2-dev
git clone -b alsapatch https://github.com/gglockner/portaudio
cd portaudio
./configure && make
make install
ldconfig
cd ..
