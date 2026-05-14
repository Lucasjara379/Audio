from pathlib import Path
import librosa
import numpy as np
import sys

if not Path("audio_clips").exists():
    print("No folder called audio_clips exists in working directory.")
    sys.exit()

# get the audio files loaded into script
filename1 = input("Input first audio file: ")
filename2 = input("Input second audio file: ")
audio1, audio2 = Path("audio_clips") / filename1
y1, sr1 = librosa.load(audio1)
y2, sr2 = librosa.load(audio2)

# convert to mfcc which is apparently good for noticing similarities between songs
mfcc1 = librosa.feature.mfcc(y1, sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y2, sr2, n_mfcc=13)

# Use the dynamic time warping algorithm for similarity
D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2)
raw_distance = D[-1, -1]

# Find the normalisation constant
N = np.shape(mfcc1)[1]
M = np.shape(mfcc2)[1]

# Normalise and evaluate similarity constant between 0 and 1
normalised_distance = raw_distance / (N + M)
similarity = 1 / (1 + normalised_distance)