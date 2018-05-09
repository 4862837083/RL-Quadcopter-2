import numpy as np
import tensorflow as tf
from functools import partial


class Actor(object):
    def __init__(self, n_observation, n_action, name='actor_net'):
        self.n_observation = n_observation
        self.n_action = n_action
        self.name = name
        self.sess = None
        self.build_model()
        self.build_train()

    def build_model(self):
        activation = tf.nn.elu
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0)
        default_dense = partial(tf.layers.dense, \
                                activation=activation, \
                                kernel_initializer=kernel_initializer, \
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name):
            observation = tf.placeholder(tf.float32, shape=[None, self.n_observation])
            hid1 = default_dense(observation, 128)
            hid1 = tf.layers.batch_normalization(hid1)
            hid1 = tf.layers.dropout(hid1, 0.3)
            hid2 = default_dense(hid1, 128)
            hid2 = tf.layers.batch_normalization(hid2)
            hid2 = tf.layers.dropout(hid2, 0.3)
            hid3 = default_dense(hid2, 64)
            action = default_dense(hid3, self.n_action, activation=tf.nn.tanh, use_bias=False)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.observation, self.action, self.trainable_vars = observation, action, trainable_vars

    def build_train(self, learning_rate=0.0001):
        with tf.variable_scope(self.name) as scope:
            action_grads = tf.placeholder(tf.float32, [None, self.n_action])
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                var_grads = tf.gradients(self.action, self.trainable_vars, -action_grads)
                train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(var_grads, self.trainable_vars))
        self.action_grads, self.train_op = action_grads, train_op

    def predict_action(self, obs_batch):
        return self.action.eval(session=self.sess, feed_dict={self.observation: obs_batch})

    def train(self, obs_batch, action_grads):
        batch_size = len(action_grads)
        self.train_op.run(session=self.sess,
                          feed_dict={self.observation: obs_batch, self.action_grads: action_grads / batch_size})

    def set_session(self, sess):
        self.sess = sess

    def get_trainable_dict(self):
        return {var.name[len(self.name):]: var for var in self.trainable_vars}