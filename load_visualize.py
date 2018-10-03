import glob
import os
import librosa
import librosa.display as display
import librosa.core as core
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

# LoadPlot loads and plots sound waves, spectograms and log-power spectograms using matplotlib.pyplot
class LoadPlot():
    def __init__(self, i, figsize=(10, 12), dpi=100, x=0.5, y=0.5, fontsize=5):
        self.i = 1
        self.figsize = figsize
        self.dpi = dpi
        self.x = x
        self.y = y
        self.fontsize = fontsize

    # Load the sound files from the specified file path
    def load_sound_files(self, path):
        raw_sounds = []
        for fp in path:
            Y, sr = librosa.load(fp)
            raw_sounds.append(Y)
        return raw_sounds

    def plot_waves(self, sound_names, raw_sounds):
        i = self.i
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        for n, f in zip(sound_names, raw_sounds):
            plt.subplot(10, 1, i)
            display.waveplot(np.array(f), sr=22050)
            plt.title(n.title())
            i += 1
        plt.suptitle("Figure 1: Waveplot", x=self.x, y=self.y, fontsize=self.fontsize)
        plt.tight_layout()
        plt.show()

    def plot_specgram(self, sound_names, raw_sounds):
        i = self.i
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        for n, f in zip(sound_names, raw_sounds):
            plt.subplot(10, 1, i)
            #plt.specgram - plot a specgram
            specgram(np.array(f), Fs=22050)
            plt.title(n.title())
            i+=1
        plt.suptitle("Figure 2: Spectogram", x=self.x, y=self.y, fontsize=self.fontsize)
        plt.tight_layout()
        plt.show()

    def plot_log_power_specgram(self, sound_names, raw_sounds):
        i = self.i
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        for n, f in zip(sound_names, raw_sounds):
            plt.subplot(10, 1, i)
            D = core.amplitude_to_db(np.abs(librosa.stft(f))**2, ref=np.max)
            """ref_power parameter deprecated after librosa 0.6.0
                and librosa.core.logamplitude has been removed; replaced by amplitude_to_db"""
            #D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
            display.specshow(D, x_axis='time', y_axis='log')
            plt.title(n.title())
            i+=1
        plt.suptitle("Figure 3: Log power spectrogram", x=self.x, y=self.y, fontsize=self.fontsize)
        plt.show()

def main():
    sound_file_paths = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav",
                    "46669-4-0-35.wav","89948-5-0-0.wav","40722-8-0-4.wav",
                    "103074-7-3-2.wav","106905-8-0-0.wav","108041-9-0-4.wav"]

    sound_names = ["air conditioner","car horn","children playing",
                "dog bark","drilling","engine idling", "gun shot",
                "jackhammer","siren","street music"]

    loader = LoadPlot(1)
    raw_sounds = loader.load_sound_files('sample_data/' + i for i in sound_file_paths)

    #loader.plot_waves(sound_names, raw_sounds)
    #loader.plot_specgram(sound_names, raw_sounds)
    loader.plot_log_power_specgram(sound_names, raw_sounds)

if __name__ == '__main__':
    main()
