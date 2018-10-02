import librosa
import os
import numpy as np
import glob as g
import tqdm

FILE_EXT='*.wav'
FILE_NAME=''
FILE_URL='https://www.google.com/url?q=https://goo.gl/8hY5ER&sa=D&ust=1538505137084000&usg=AFQjCNEFbxtZWZdlAWlAX0LVnZTty_y2HQ'
DATAPATH='./data'

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

class FeatureParser():
    def __init__(self, file_ext=FILE_EXT):
        self.file_ext = file_ext

    def DownloadData(datapath):
        # If dataset is not already donwloaded yet, download it
        print('Checking for downloaded dataset...')

        if not isfile(FILE_NAME):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='Urban Sound Dataset') as progressbar:
                urlretrieve(FILE_URL, 'UrbanSound8K.tar.gz',progressbar.hook)

        else:
            pass

        if not isdir(DATAPATH):
           with tarfile.open(FILE_NAME) as tar:
               tar.extractall()
               tar.close()

    def extract_feature(path):
        Y, sample_rate = librosa.load(path)
        stft = np.abs(librosa.stft(Y))
        mfcc = np.mean(librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(Y, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effect.harmonic(Y), sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz

    def parse_audio_files(parent_dir, sub_dir, file_ext=FILE_EXT):
        features, labels = np.empty((0, 193)), np.empty(0) #TODO:Initialize with normal distribution values
        for label, sub_dir in enumerate(sub_dir):
            for fn in g.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try:
                    mfcc, chroma, mel, contrast, tonnetz = extract_feature(fn)
                except Exception:
                    print('Error while extracting feature from the file; at parse_audio_files', fn)
                    continue
                extfeatures = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split('/')[2].split('-')[1])
        return np.array(features), np.array(labels, dtype = np.int)

    def one_hot_encode(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros(n_labels, n_unique_labels)
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode

def main():
    f = FeatureParser()
    f.DownloadData(DATAPATH)

if __name__ == '__main__':
    main()
