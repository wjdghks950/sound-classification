import librosa
import pickle
import os
import numpy as np
import glob as g
import tqdm
import tarfile
import requests
from urllib.request import urlretrieve, urlopen
from os.path import isfile, isdir

FILE_EXT='*.wav'
FILE_NAME='./UrbanSound8k.tar.gz'
FILE_URL='https://www.google.com/url?q=https://goo.gl/8hY5ER&sa=D&ust=1538505137084000&usg=AFQjCNEFbxtZWZdlAWlAX0LVnZTty_y2HQ'
DATAPATH='/media/jeonghwan/Seagate Expansion Drive/UrbanSound8K/audio'

def DownloadData(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as f:
        # get request
        print('Downloading...')
        response = requests.get(url)
        # write to file
        f.write(response.content)
        print('Download complete')

    if not isdir(DATAPATH):
        path = os.getcwd()
        path = path + '/data'
        print('Creating ' + path + ' directory for Urban Sound Dataset')
        try:
            os.mkdir(path)
        except:
            print('Failed to create the following directory:{}'.format(path))
        else:
            print('{} directory created'.format(path))

        if isfile(FILE_NAME):
            with tarfile.open(FILE_NAME) as tar:
                tar.extractall(path)
                tar.close()
            

class FeatureParser():
    def __init__(self, file_ext=FILE_EXT):
        self.file_ext = file_ext

    def windows(self, data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size / 2)

    def extract_feature(self, path, pickle_exists=False):
        if pickle_exists is False:
            Y, sample_rate = librosa.load(path)
        else:
            pass

        stft = np.abs(librosa.stft(Y))
        mfcc = np.mean(librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(Y, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(Y), sr=sample_rate).T, axis=0)
        return mfcc, chroma, mel, contrast, tonnetz

    # Extracting and preprocessing features for ConvNet
    def extract_CNNfeature(self, parent_dir, sub_dirs, file_ext=FILE_EXT, bands = 60, frames = 41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        labels = []
        if not isfile('audio_CNNdataset.pickle'):
            for label, sub_dir in enumerate(sub_dirs):
                for fn in g.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                    sound_clip, sr = librosa.load(fn)
                    lbl = fn.split('/')[7].split('-')[1]
                    for (start, end) in self.windows(sound_clip, window_size):
                        if(len(sound_clip[start:end]) == window_size):
                            signal = sound_clip[start:end]
                            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                            logspec = librosa.core.amplitude_to_db(melspec)
                            logspec = logspec.T.flatten()[:, np.newaxis].T
                            log_specgrams.append(logspec)
                            labels.append(lbl)

            log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
            features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
            for i in range(len(features)):
                features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        
        return np.array(features), np.array(labels, dtype=np.int)

    def parse_audio_files(self, parent_dir, sub_dirs, file_ext=FILE_EXT):
        features, labels = np.empty((0, 193)), np.empty(0)

        if not isfile('audio_dataset.pickle'):
            for label, sub_dir in enumerate(sub_dirs):
                print('Subdirectory path:{}'.format(sub_dir))
                for fn in g.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                    try:
                        if os.path.exists(fn):
                            mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(fn)
                        else:
                            raise ValueError('Error while extracting feature from the file; at parse_audio_files')
                    except ValueError as err:
                        print(err.args)

                    ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
                    features = np.vstack([features, ext_features])
                    labels = np.append(labels, fn.split('/')[7].split('-')[1])

        return np.array(features), np.array(labels, dtype = np.int)

    def one_hot_encode(self, labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode

def main():
    f = FeatureParser()
    #DownloadData(FILE_URL, FILE_NAME)


if __name__ == '__main__':
    main()
