# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:00:11 2018

@author: PARS PARDAZ CO.1
"""

from scipy.io import wavfile
from speechpy import processing
from speechpy import feature
from speechpy import functions

fs, data = wavfile.read('d:\waves\clean2.wav')


#-------------------------------------------------------
    """PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
#-------------------------------------------------------
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

fs=44100
duration = 5  # seconds
myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
print ("Recording Audio")
sd.wait()
print ("Audio recording complete , Play Audio")
sd.play(myrecording, fs)
sd.wait()
print ("Play Audio Complete")
#-------------------------------------------------------

import scipy.io.wavfile as wav
import numpy as np
import speechpy


file_name = 'd:\waves\clean2.wav'
#os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
fs, signal = wav.read(file_name)
#signal = signal[:,0]

# Example of pre-emphasizing.
signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

# Example of staching frames
frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
         zero_padding=True)

# Example of extracting power spectrum
power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
print('power spectrum shape=', power_spectrum.shape)

############# Extract MFCC features #############
mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)

############# Extract logenergy features #############
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
print('logenergy features=', logenergy.shape)