import tensorflow as tf
from functools import partial

class Critic(object):
    def __init__(self, n_observation, n_action, name='critic_net'):
        self.n_observation = n_observation
        self.n_action = n_action
        self.name = name
        self.sess = None
        self.build_model()
        self.build_train()

    def build_model(self):
        activation = tf.nn.relu
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0)
        default_dense = partial(tf.layers.dense, \
                                activation=activation, \
                                kernel_initializer=kernel_initializer, \
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name) as scope:
            observation = tf.placeholder(tf.float32, shape=[None, self.n_observation])
            action = tf.placeholder(tf.float32, shape=[None, self.n_action])
            hid1 = default_dense(observation, 64)
            hid2 = default_dense(action, 64)
            hid3 = tf.concat([hid1, hid2], axis=1)
            #hid3 = tf.layers.batch_normalization(hid3)
            hid3 = tf.layers.dropout(hid3, 0.7)
            hid4 = default_dense(hid3, 128)
            hid4 = tf.layers.dropout(hid4, 0.7)
            hid5 = default_dense(hid4, 64)
            Q = default_dense(hid5, 1, activation=None)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.observation, self.action, self.Q, self.trainable_vars = observation, action, Q, trainable_vars

    def build_train(self, learning_rate=0.001):
        with tf.variable_scope(self.name) as scope:
            Qexpected = tf.placeholder(tf.float32, shape=[None, 1])
            #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            loss = tf.losses.mean_squared_error(Qexpected, self.Q)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)
        self.Qexpected, self.train_op = Qexpected, train_op
        self.action_grads = tf.gradients(self.Q, self.action)[0]

    def predict_Q(self, obs_batch, action_batch):
        return self.Q.eval(session=self.sess, \
                           feed_dict={self.observation: obs_batch, self.action: action_batch})

    def compute_action_grads(self, obs_batch, action_batch):
        return self.action_grads.eval(session=self.sess, \
                                      feed_dict={self.observation: obs_batch, self.action: action_batch})

    def train(self, obs_batch, action_batch, Qexpected_batch):
        self.train_op.run(session=self.sess, \
                          feed_dict={self.observation: obs_batch, self.action: action_batch,
                                     self.Qexpected: Qexpected_batch})

    def set_session(self, sess):
        self.sess = sess

    def get_trainable_dict(self):
        return {var.name[len(self.name):]: var for var in self.trainable_vars}