import numpy as np
import tensorflow as tf
from functools import partial
from collections import namedtuple, deque
from .actor import Actor
from .critic import Critic
import math
import copy


# Reference:
# https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2%20(Pendulum-v0).ipynb

class Memory(object):
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=128):
        """Randomly sample a batch of experiences from memory."""
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]


def UONoise(dim, theta, mu, sigma):
    theta = theta
    mu = mu
    sigma = sigma
    state = np.zeros(dim)
    while True:
        state += theta * (mu - state) + sigma * np.random.randn(dim)
        yield state

# a = UONoise(4, 0.15, 5, 0.1)
# for idx in range(20):
#     print(next(a))

class AsyncNets(object):
    def __init__(self, class_name, n_state, n_action):
        class_ = eval(class_name)
        self.net = class_(n_state, n_action, name=class_name)
        self.target_net = class_(n_state, n_action, name='{}_target'.format(class_name))
        self.TAU = tf.placeholder(tf.float32, shape=None)
        self.sess = None
        self.__build_async_assign()

    def __build_async_assign(self):
        net_dict = self.net.get_trainable_dict()
        target_net_dict = self.target_net.get_trainable_dict()
        keys = net_dict.keys()
        async_update_op = [target_net_dict[key].assign((1 - self.TAU) * target_net_dict[key] + self.TAU * net_dict[key]) \
                           for key in keys]
        self.async_update_op = async_update_op

    def async_update(self, tau=0.01):
        self.sess.run(self.async_update_op, feed_dict={self.TAU: tau})

    def set_session(self, sess):
        self.sess = sess
        self.net.set_session(sess)
        self.target_net.set_session(sess)

    def get_subnets(self):
        return self.net, self.target_net


class DDPG(object):
    def __init__(self,
                 task,
                 UONoise_para=None,
                 max_episode=500,
                 gamma=0.99,
                 tau=0.01,
                 memory_size=10000,
                 batch_size=64,
                 memory_warmup=128,
                 max_explore_eps=200,
                 ):
        ## Task related
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.last_state = task.reset()
        self.curr_episode = 0
        self.UONoise_para = UONoise_para if UONoise_para is not None else {'theta': 0.15, 'mu': 0, 'sigma': 0.2}

        # Model Parameters
        self.gamma = gamma
        self.tau = tau
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_warmup = memory_warmup
        self.max_episode = max_episode
        self.max_explore_eps = max_explore_eps

        # Init Actor, Critic
        tf.reset_default_graph()
        self.actorAsync = AsyncNets('Actor', self.state_size, self.action_size)
        self.actor, self.actor_target = self.actorAsync.get_subnets()
        self.criticAsync = AsyncNets('Critic', self.state_size, self.action_size)
        self.critic, self.critic_target = self.criticAsync.get_subnets()
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        self.actorAsync.set_session(self.sess)
        self.criticAsync.set_session(self.sess)
        self.memory = Memory(memory_size)
        self.noise = UONoise(self.action_size, **self.UONoise_para)

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def step(self, state=None, is_train=True):
        # Save experience / reward
        if state is None:
            state = self.last_state
        action = self.actor.predict_action(np.reshape(state, [-1, self.state_size]))[0]

        if is_train:
            if self.curr_episode < self.max_explore_eps:  # exploration policy
                epsilon = self.curr_episode / self.max_explore_eps + 0.01
                action =  action * epsilon + (1 - epsilon) * next(self.noise)

            action_scale = (np.clip(action, -1, 1) + 1)/2 * (self.action_high - self.action_low) + self.action_low
            next_state, reward, done = self.task.step(action_scale)
            self.memory.add(state, action, reward, next_state, done)
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.memory_warmup:
                self.learn()
        else:
            action_scale = (np.clip(action, -1, 1) + 1) / 2 * (self.action_high - self.action_low) + self.action_low
            next_state, reward, done = self.task.step(action_scale)
        self.last_state = copy.copy(next_state)
        return action_scale, next_state, reward, done

    def learn(self):
        memory_batch = self.memory.sample(self.batch_size)
        extract_mem = lambda k: np.array([item[k] for item in memory_batch])
        obs_batch = extract_mem(0)
        action_batch = extract_mem(1)
        reward_batch = extract_mem(2)
        next_obs_batch = extract_mem(3)
        done_batch = extract_mem(4)
        action_next = self.actor_target.predict_action(next_obs_batch,
                                                       )
        Q_next = self.critic_target.predict_Q(next_obs_batch, action_next)[:, 0]
        Qexpected_batch = reward_batch + self.gamma * (1 - done_batch) * Q_next  # target Q value
        Qexpected_batch = np.reshape(Qexpected_batch, [-1, 1])
        # train critic
        self.critic.train(obs_batch, action_batch, Qexpected_batch)
        # train actor
        action_grads = self.critic.compute_action_grads(obs_batch, action_batch)
        self.actor.train(obs_batch, action_grads)
        # async update
        self.actorAsync.async_update(self.tau)
        self.criticAsync.async_update(self.tau)

    def reset_episode(self):
        self.noise = UONoise(self.action_size, **self.UONoise_para)
        state = self.task.reset()
        self.last_state = state
        return state

    def train(self):
        self.curr_episode = 0
        iteration = 0
        episode_score = 0
        episode_steps = 0
        max_score = -math.inf
        state = self.reset_episode()
        while self.curr_episode < self.max_episode:
            #print('\riter {}, ep {}'.format(iteration, self.curr_episode), end='')
            lst_act, state, reward, done = self.step()
            episode_score += reward
            episode_steps += 1
            iteration += 1
            if done:
                print('\r iter {}, ep {} ,{} score {:8f}, best {:8f} time {}'.format(
                    iteration, self.curr_episode,
                    lst_act, episode_score, max_score, self.task.gettime()),
                end='')

                state = self.reset_episode()
                self.curr_episode += 1
                if episode_score > max_score:
                    max_score = episode_score
                episode_score = 0
                episode_steps = 0



