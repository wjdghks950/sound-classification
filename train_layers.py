import tensorflow as tf
import tensorflow.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

training_epochs = 500
learning_rate = 0.01

class FeedForward():
    def __init__(self, n_dim, n_classes, lr):
        self.opt = {}
        self.opt['n_dim'] = n_dim
        self.opt['n_classes'] = n_classes
        self.opt['learning_rate'] = lr
        self.opt['std'] = 1 / np.sqrt(n_dim)
        self.opt['num_hidden1'] = 200
        self.opt['num_hidden2'] = 300


    def train_layers(self, train_x, train_y, test_x, test_y):
        params = {}

        X = tf.placeholder(tf.float32, [None, self.opt['n_dim']])
        Y = tf.placeholder(tf.float32, [None, self.opt['n_classes']])

        params['W1'] = tf.Variable(tf.random_normal([self.opt['n_dim'], self.opt['num_hidden1']], mean = 0, stddev=self.opt['std']))
        params['b1'] = tf.Variable(tf.random_normal([self.opt['num_hidden1']], mean = 0, stddev=self.opt['std']))
        params['a1'] = nn.sigmoid(tf.matmul(X, params['W1']) + params['b1'])

        params['W2'] = tf.Variable(tf.random_normal([self.opt['num_hidden1'], self.opt['num_hidden2']], mean = 0, stddev=self.opt['std']))
        params['b2'] = tf.Variable(tf.random_normal([self.opt['num_hidden2']], mean=0, stddev=self.opt['std']))
        params['a2'] = nn.tanh(tf.matmul(params['a1'], params['W2']) + params['b2'])

        params['outW'] = tf.Variable(tf.random_normal([self.opt['num_hidden2'], self.opt['n_classes']], mean=0, stddev=self.opt['std']))
        params['outb'] = tf.Variable(tf.random_normal([self.opt['n_classes']], mean=0, stddev=self.opt['std']))

        out = nn.softmax(tf.matmul(params['a2'], params['outW']) + params['outb'])

        cost = -tf.reduce_sum(Y * tf.log(out))
        optimizer = tf.train.GradientDescentOptimizer(self.opt['learning_rate']).minimize(cost)

        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)
        y, y_pred = None, None

        y_shape = tf.shape(train_y)
        #reshape labels into a one hot vector
        tr_onehot_lbl = np.eye(self.opt['n_classes'])[train_y] #TODO

        print('TRAIN_ONE_HOT_LABEL{}'.format(tr_onehot_lbl))

        print('Training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                _, cost = sess.run([optimizer, cost], feed_dict={X:train_x, Y:train_y})
                cost_history = np.append(cost_history, cost)

            y_pred = sess.run(tf.argmax(out, 1), feed_dict={X: test_x})
            y = sess.run(tf.argmax(test_y, 1))

            print("Test accuracy: ", round(session.run(accuracy, feed_dict={X: test_x, Y: test_y}), 3))

        fig = plt.figure(figsize=(10, 8))
        plt.plot(cost_history)
        plt.ylabel("Cost")
        plt.xlabel("Iterations")
        plt.axis([0, training_epochs, 0, np.max(cost_history)])
        plt.show()

        precision, recall, f_score, s = precision_recall_fscore_support(y, y_pred, average='micro')
        print('F score:', round(f_score, 3)) 
