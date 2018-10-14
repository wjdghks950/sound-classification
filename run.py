import tensorflow as tf
import numpy as np
import pickle
from feature_extract import FeatureParser
from train_layers import FeedForward
from os.path import isfile

NUM_CLASS=10
LEARNING_RATE=1e-2
PARENT_DIR='/media/jeonghwan/Seagate Expansion Drive/UrbanSound8K/audio'

def main():

    f = FeatureParser()

    parent_dir = PARENT_DIR
    sub_dir = ['fold1', 'fold2', 'fold3']

    print('Parsing audio files...')
    print('Extracting features...')
    features, labels = f.parse_audio_files(parent_dir, sub_dir)

    labels = f.one_hot_encode(labels)
    print('Length of features:{}'.format(len(features)))
 
    train_test_split = np.random.rand(len(features)) < 0.70
    print('Train_test_split index:{}'.format(train_test_split))
    train_x = features[train_test_split]
    train_y = labels[train_test_split]
    test_x = features[~train_test_split]
    test_y = labels[~train_test_split]

    if not isfile('audio_dataset_nn.pickle'):
        data_dict = {'tr_features': train_x,
                     'tr_labels': train_y,
                     'ts_features': test_x,
                     'ts_labels': test_y}

        with open('audio_dataset_nn.pickle', 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('audio_dataset_nn.pickle', 'rb') as handle:
            data = pickle.load(handle)

    with tf.Session() as sess:
        print('Shape of test_x:{}'.format(sess.run(tf.shape(data['ts_features']))))

    print('TRAIN_X:{}\nTEST_X:{}'.format(data['tr_features'], data['ts_features']))
    print('FEATURE SHAPE:{}'.format(features.shape[1]))
    # Initialize the feed-forward model
    model = FeedForward(features.shape[1], NUM_CLASS, LEARNING_RATE)
    model.train_layers(data['tr_features'], data['tr_labels'], data['ts_features'], data['ts_labels'])


if __name__ == '__main__':
    main()
