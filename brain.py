# CartPole-v0
import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(
        self,
        n_actions,
        space_shape,
        batch_size = 32,
        learning_rate = 0.01,
        epsilon = 0.9,
        gamma = 0.9,
        target_replace_iter = 100,
        replay_memory_size = 2000,
        output_graph = False
        ):
        ''' Hyper parameters '''
        self.n_actions = n_actions
        self.space_shape = space_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter

        # Save the hyper parameters
        self.params = self.__dict__.copy()

        # Replay memory
        self.D = deque(maxlen=replay_memory_size)

        ''' Tensorflow placeholders '''
        self.observation = tf.placeholder(tf.float32, [None, space_shape], name='observation')
        self.action = tf.placeholder(tf.int32, [None, ], name='action')
        self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
        self.observation_ = tf.placeholder(tf.float32, [None, space_shape], name='observation_')

        # Set seeds
        tf.set_random_seed(1)
        np.random.seed(1)

        with tf.variable_scope('q'):        # evaluation network
            l_eval = tf.layers.dense(self.observation, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(l_eval, n_actions, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('q_next'):   # target network, not to train
            l_target = tf.layers.dense(self.observation_, 10, tf.nn.relu, trainable=False)
            self.q_next = tf.layers.dense(l_target, n_actions, trainable=False)

        q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1)    # shape=(None, )

        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)                    # shape=(None, )

        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.scalar('loss', loss)
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.observation: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        self.D.append((s, a, r, s_))

    def learn(self):
        if not hasattr(self, 'learning_step_counter'):
            self.learning_step_counter = 0

        # Update target net
        if self.learning_step_counter % self.target_replace_iter == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        self.learning_step_counter += 1

        # Learning
        batch_size = min(len(self.D), self.batch_size)
        batch_indeces = np.random.randint(0, len(self.D), batch_size)
        batch_observation = deque(maxlen=batch_size)
        batch_action = deque(maxlen=batch_size)
        batch_reward = deque(maxlen=batch_size)
        batch_observation_ = deque(maxlen=batch_size)
        for j in batch_indeces:
            observation_j, action_j, reward_j, observation_j_ = self.D[j]
            batch_observation.append(observation_j)
            batch_action.append(action_j)
            batch_reward.append(reward_j)
            batch_observation_.append(observation_j_)

        _, summary = self.sess.run([self.train_op, self.summary_op], {self.observation: batch_observation, self.action: batch_action, self.reward: batch_reward, self.observation_: batch_observation_})
        self.writer.add_summary(summary, self.learning_step_counter)

    def show_parameters(self):
        ''' Helper function to show the hyper parameters '''
        for key, value in self.params.items():
            print key, '=', value
