import tensorflow as tf
import tensorflow.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from feature_extract import FeatureParser
from sklearn.metrics import precision_recall_fscore_support

training_epochs = 1000

class FeedForward():
    def __init__(self, n_dim, n_classes, lr):
        self.opt = {}
        self.opt['n_dim'] = n_dim
        self.opt['n_classes'] = n_classes
        self.opt['learning_rate'] = lr
        self.opt['std'] = 1 / np.sqrt(n_dim)
        self.opt['num_hidden1'] = 500
        self.opt['num_hidden2'] = 1000
        self.opt['num_hidden3'] = 700


    def train_layers(self, train_x, train_y, test_x, test_y):
        params = {}

        X = tf.placeholder(tf.float32, [None, self.opt['n_dim']])
        Y = tf.placeholder(tf.float32, [None, self.opt['n_classes']])
        keep_prob = tf.placeholder(tf.float32) #for dropout

        params['W1'] = tf.Variable(tf.random_normal([self.opt['n_dim'], self.opt['num_hidden1']], mean = 0, stddev=self.opt['std']))
        params['b1'] = tf.Variable(tf.random_normal([self.opt['num_hidden1']], mean = 0, stddev=self.opt['std']))
        params['a1'] = nn.sigmoid(tf.matmul(X, params['W1']) + params['b1'])
        params['dropout1'] = nn.dropout(params['a1'], keep_prob)

        params['W2'] = tf.Variable(tf.random_normal([self.opt['num_hidden1'], self.opt['num_hidden2']], mean = 0, stddev=self.opt['std']))
        params['b2'] = tf.Variable(tf.random_normal([self.opt['num_hidden2']], mean=0, stddev=self.opt['std']))
        params['a2'] = nn.relu(tf.matmul(params['dropout1'], params['W2']) + params['b2'])
        params['dropout2'] = nn.dropout(params['a2'], keep_prob)

        params['W3'] = tf.Variable(tf.random_normal([self.opt['num_hidden2'], self.opt['num_hidden3']], mean = 0, stddev=self.opt['std']))
        params['b3'] = tf.Variable(tf.random_normal([self.opt['num_hidden3']], mean=0, stddev=self.opt['std']))
        params['a3'] = nn.tanh(tf.matmul(params['dropout2'], params['W3']) + params['b3'])
        params['dropout3'] = nn.dropout(params['a3'], keep_prob)

        params['outW'] = tf.Variable(tf.random_normal([self.opt['num_hidden3'], self.opt['n_classes']], mean=0, stddev=self.opt['std']))
        params['outb'] = tf.Variable(tf.random_normal([self.opt['n_classes']], mean=0, stddev=self.opt['std']))

        out = nn.softmax(tf.matmul(params['dropout3'], params['outW']) + params['outb'])

        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(out), reduction_indices=[1]))
        optimizer = tf.train.AdamOptimizer(self.opt['learning_rate']).minimize(cost)

        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)
        y, y_pred = None, None

        #reshape labels into a one hot vector
        f = FeatureParser()
        train_y = f.one_hot_encode(train_y)
        test_y = f.one_hot_encode(test_y)        

        print('TRAIN_ONE_HOT_LABEL{}'.format(train_y))

        print('Training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                _, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={X:train_x, Y:train_y, keep_prob: 0.5})
                cost_history = np.append(cost_history, loss)
                if epoch % 50 == 0:
                    print('Epoch#', epoch, 'Cost:', loss, 'Train acc.:', acc)
            
            y_pred = sess.run(tf.argmax(out, 1), feed_dict={X: test_x, keep_prob: 1.0})
            y = sess.run(tf.argmax(test_y, 1))

            print("Test accuracy: ", round(sess.run(accuracy, feed_dict={X: test_x, Y: test_y, keep_prob:1.0}), 3))

        fig = plt.figure(figsize=(10, 8))
        plt.plot(cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.axis([0, training_epochs, 0, np.max(cost_history)])
        plt.show()

        precision, recall, f_score, s = precision_recall_fscore_support(y, y_pred, average='micro')
        print('F score:', round(f_score, 3)) 
