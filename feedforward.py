import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o,w_h2):
    h = tf.nn.relu(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    h_2=tf.nn.relu(tf.matmul(h, w_h2))
    return tf.matmul(h_2, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

trX = train_images/255.0
trY = test_images/255.0
trY = train_labels
reY = test_labels

X = tf.placeholder("float", [None, 100*100])
Y = tf.placeholder("float", [None, 104])

w_h = init_weights([100*100, 625]) # create symbolic variables
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 104])

py_x = model(X, w_h, w_o,w_h2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.AdamOptimizer(1e-1).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        for start, end in zip(range(0, len(trX), 200), range(200, len(trX)+1, 200)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))