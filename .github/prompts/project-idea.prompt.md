# Voice isolation ai

The project aims to train ai on the specific person's voice and be able to isolate it in the audio recording. That will eventually be used in the microphone stream, but for now let's concentrate on training the ai and basic processing. The main problem that this AI tries to solve is "My friend has a brother who always talks or yells in the background, as well as the family that can be noisy sometimes. So he has to isolate his voice from everything else." My friend has recordings of his voice and voices of his family, and we can add some random noise files like static noise, random sounds without his voice.

1. The project has a folder VOICE with my friend's voice files
2. The project has a folder NOISE with some other person's voice files and noise audio files

## The project workflow

- takes an audio file and decomposes it using Short Fourier Transform
- feeds audio to ai that will generate a mask
- applies the mask to isolate my friend's voice
- output cleared audio file

## Project requirements

- pytorch and if needed numpy
- it should have a configuration for the sliding window size (30ms, 500ms, 2s)
- it should have a part for training the ai and a part for using ai
- no need to create an implementation for the microphone integration, for now, only simple audio file input and audio file output
- keep in mind, audio files provided in training data are of various length
- keep in mind, audio files are separated (VOICE, NOISE) and have to be combined in various ways for training (ONLY NOISE, ONLY VOICE, COMBINATION, NONE)
- keep in mind, audio formats are (mp3, wav)
- keep everything documented
- create a quickstart
- keep in mind, the sounds might be of various frequencies
- the AI should be CNN
