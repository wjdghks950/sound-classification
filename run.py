import tensorflow as tf
from feature_extract import FeatureParser
from train_layers import FeedForward

NUM_CLASS=10
LEARNING_RATE=1-e2

def main():

    f = FeatureParser()

    parent_dir = 'data'
    sub_dir = ['fold1', 'fold2', 'fold3']

    features, labels = f.parse_audio_files(parent_dir, sub_dir)

    labels = f.one_hot_encode(labels)

    train_test_split = np.random.rand(len(features)) < 0.70
    train_x = features[train_test_split]
    train_y = labels[train_test_split]
    test_x = features[~train_test_split]
    test_y = labels[~train_test_split]

    # Initialize the feed-forward model
    model = FeedForward(features.shape[1], NUM_CLASS, LEARNING_RATE)
    model.train_layers(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
