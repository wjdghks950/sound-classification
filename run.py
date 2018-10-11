import tensorflow as tf
import numpy as np
from feature_extract import FeatureParser
from train_layers import FeedForward

NUM_CLASS=10
LEARNING_RATE=1e-2
PARENT_DIR='/media/jeonghwan/Seagate Expansion Driver/UrbanSound8K/audio'

def main():

    f = FeatureParser()

    parent_dir = PARENT_DIR
    sub_dir = ['fold1', 'fold2', 'fold3']

    features, labels = f.parse_audio_files(parent_dir, sub_dir)

    labels = f.one_hot_encode(labels)

    train_test_split = np.random.rand(len(features)) < 0.70
    print('Train_test_split index:{}'.format(train_test_split))
    train_x = features[train_test_split]
    train_y = labels[train_test_split]
    test_x = features[~train_test_split]
    test_y = labels[~train_test_split]

    with tf.Session() as sess:
        print('Shape of test_y:{}'.format(sess.run(tf.shape(test_y))))

    print('TRAIN_X:{}\nTEST_X:{}'.format(train_x, test_x))
    print('FEATURE SHAPE:{}'.format(features.shape[1]))
    # Initialize the feed-forward model
    model = FeedForward(features.shape[1], NUM_CLASS, LEARNING_RATE)
    model.train_layers(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
