# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:45:55 2018

@author: PARS PARDAZ CO.1
"""

#=================== td_utils =============================
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 44100 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds
#================================================================
#=============   trigger word detection -Training ===============
#================================================================
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython

from datetime import datetime

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

# GRADED FUNCTION: is_overlapping

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    
    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False
    
    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    ### END CODE HERE ###

    return overlap


# GRADED FUNCTION: insert_audio_clip

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    ### START CODE HERE ### 
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)
    
    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    ### END CODE HERE ###
    
    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time

# GRADED FUNCTION: insert_ones

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    ### END CODE HERE ###
    
    return y

# GRADED FUNCTION: create_training_example

def create_training_example(background, activates, negatives, index):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Set the random seed
#    np.random.seed(18)
    micro=datetime.now().microsecond
    np.random.seed(micro%100)

    
    # Make background quieter
    background = background - 20

    ### START CODE HERE ###
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []
    ### END CODE HERE ###
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    ### START CODE HERE ### (≈ 3 lines)
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)
    ### END CODE HERE ###

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    ### START CODE HERE ### (≈ 2 lines)
    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    ### END CODE HERE ###
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    filename = "./training_examples/train" +str(int(index))+ ".wav"
    background.export(filename, format="wav")

    y_int = y.astype("int")
    y_filename = "./training_examples/train" +str(int(index))+ "_ones.txt"
    np.savetxt(y_filename, y_int, fmt="%d")

    print("File "+filename+" was saved in your directory.")
    #file_handle = background.export("train"+str(int(index)) + ".wav", format="wav")
    #print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    #    x = graph_spectrogram(filename)
    return

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

activates, negatives, backgrounds = load_raw_audio()

#for i in range(10):
#    create_training_example(backgrounds[0], activates, negatives, i)
#    create_training_example(backgrounds[1], activates, negatives, i+10)
'''   
#--------------------------------------------------------------------------    
b=1
a=1
n=1
number_of_activates = a
for a_index in range(9):
    background = (backgrounds[b])
    previous_segments = []
    y = np.zeros((1, Ty))
    random_indices = [a_index] #np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end)
    number_of_negatives = n #np.random.randint(0, 3)
    for n_index in range(10):
        new_previous_segments = list(previous_segments)
        newbackground = background
        random_indices = [n_index] #np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[j] for j in random_indices]
        for random_negative in random_negatives:
            newbackground, _ = insert_audio_clip(newbackground, random_negative, new_previous_segments)            
        newbackground = match_target_amplitude(newbackground, -20.0)
        xdate=str(datetime.now())
        xdate = xdate.replace(":","_")
        xdate = xdate.replace(".","_")
        xdate = xdate.replace(" ","_")
        filename = "./training_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+".wav"
        newbackground.export(filename, format="wav")
        y_int = y.astype("int")
        y_filename = "./training_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+ "_ones.txt"
        np.savetxt(y_filename, y_int, fmt="%d")
        print(filename)'''
#--------------------------------------------------------------------------
'''         
b=1
a=2
n=2
number_of_activates = a

for a_index in range(5):
    background = (backgrounds[b])
    previous_segments = []
    y = np.zeros((1, Ty))
    random_indices = [a_index, a_index+1]
    #np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end)
    number_of_negatives = n #np.random.randint(0, 3)

    for n_index in range(5):
        new_previous_segments = list(previous_segments)
        newbackground = background
        random_indices = [n_index, n_index+1] #np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[j] for j in random_indices]
        for random_negative in random_negatives:
            newbackground, _ = insert_audio_clip(newbackground, random_negative, new_previous_segments)            
        newbackground = match_target_amplitude(newbackground, -20.0)
        xdate=str(datetime.now())
        xdate = xdate.replace(":","_")
        xdate = xdate.replace(".","_")
        xdate = xdate.replace(" ","_")
        filename = "./training_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+".wav"
        newbackground.export(filename, format="wav")
        y_int = y.astype("int")
        y_filename = "./training_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+ "_ones.txt"
        np.savetxt(y_filename, y_int, fmt="%d")
        print(filename)
'''        
#------------------------ Evaluation Examples --------------------------------------------------        
b=1
a=3
n=5
number_of_activates = a

for a_index in range(3):
    background = (backgrounds[b])
    previous_segments = []
    y = np.zeros((1, Ty))
    random_indices = [ a_index+2, a_index, a_index+5]
    #np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end)
    number_of_negatives = n #np.random.randint(0, 3)

    for n_index in range(4):
        new_previous_segments = list(previous_segments)
        newbackground = background
        random_indices = [ n_index, n_index+1, n_index+2, n_index+3, n_index+5] #np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[j] for j in random_indices]
        for random_negative in random_negatives:
            newbackground, _ = insert_audio_clip(newbackground, random_negative, new_previous_segments)            
        newbackground = match_target_amplitude(newbackground, -20.0)
        xdate=str(datetime.now())
        xdate = xdate.replace(":","_")
        xdate = xdate.replace(".","_")
        xdate = xdate.replace(" ","_")
        filename = "./eval_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+".wav"
        newbackground.export(filename, format="wav")
        y_int = y.astype("int")
        y_filename = "./eval_examples/train_b" +str(int(b))+"_a"+ str(int(a))+"_n"+str(int(n))+"_"+xdate+ "_ones.txt"
        np.savetxt(y_filename, y_int, fmt="%d")
        print(filename)
#--------------------------------------------------------------------------        
'''
for i in range(1):
    filename = "./training_examples/train" +str(int(i))+ ".wav"
    rate, data = get_wav_info(filename)
    y_filename = "./training_examples/train" +str(int(i))+ "_ones.txt"
    y_int=np.loadtxt(y_filename, dtype="int")
    y=y.astype("float64")
 '''   

