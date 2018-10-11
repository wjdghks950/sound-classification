import librosa
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
DATAPATH='/media/jeonghwan/Seagate Expansion Drive/UrbanSound8K'

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

    def extract_feature(self, path):
        Y, sample_rate = librosa.load(path)
        stft = np.abs(librosa.stft(Y))
        mfcc = np.mean(librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(Y, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(Y), sr=sample_rate).T, axis=0)
        return mfcc, chroma, mel, contrast, tonnetz

    def parse_audio_files(self, parent_dir, sub_dirs, file_ext=FILE_EXT):
        features, labels = np.empty((0, 193)), np.empty(0) 
        for label, sub_dir in enumerate(sub_dirs):
            print('Subdirectory path:{}'.format(sub_dir))
            for fn in g.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                print('***PATH:{}***'.format(fn))
                try:
                    if os.path.exists(fn):
                        mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(fn)
                        print('MFCC:{}, chroma:{}'.format(mfcc, chroma))
                    else:
                        raise ValueError('Error while extracting feature from the file; at parse_audio_files')
                except ValueError as err:
                    print(err.args)

                ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split('/')[7].split('-')[1])
                print('Class number:{}'.format(fn.split('/')[7].split('-')[1]))
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
