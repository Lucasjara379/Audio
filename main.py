from pathlib import Path
import librosa
import numpy as np
import sys


p = Path("audio_clips")
if not p.is_dir():
    print("No folder called audio_clips exists in working directory.")
    sys.exit(1)

# get the audio files loaded into script
filename1 = input("Input first audio file that's inside audio_clips: ")
audio1 = p / filename1
if not audio1.is_file():
    raise FileNotFoundError(
        "No audio clip exists with filename that was inputted."
    )

filename2 = input("Input second audio file that's also in audio_clips: ")
audio2 = p / filename2
if not (p / filename2).is_file():
    raise FileNotFoundError(
        "No audio clip exists with filename that was inputted."
    )

y1, sr1 = librosa.load(str(audio1), sr=22050)
y2, sr2 = librosa.load(str(audio2), sr=22050)

# convert to mfcc which is apparently good for noticing similarities between songs
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

# Use the dynamic time warping algorithm for similarity
D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2)
raw_distance = D[-1, -1]

# Normalise and evaluate similarity constant between 0 and 1
avg_cost = raw_distance / len(wp)
T = 20
similarity = 1 / (1 + avg_cost / T)
print(f"The similarity of the two audio clips has a value of:\n{similarity}")