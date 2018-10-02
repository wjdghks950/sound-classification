import librosa
import numpy as np
import glob as g

FILE_EXT='*.wav'

class FeatureParser():
    def __init__(self, file_ext=FILE_EXT):
        self.file_ext = file_ext

    def extract_feature(path):
        Y, sample_rate = librosa.load(path)
        stft = np.abs(librosa.stft(Y)
        mfcc = np.mean(librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(Y, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast9S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effect.harmonic(Y), sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz

    def parse_audio_files(filepath):
