# CartPole-v0
import numpy as np
import tensorflow as tf

class DQNAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        batch_size = 32,
        learning_rate = 0.01,
        epsilon = 0.9,
        gamma = 0.9,
        target_replace_iter = 100,
        memory_capacity = 2000
        ):
        ''' Hyper parameters '''
        self.n_actions = n_actions
        self.n_states = n_states
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter

        # Save the hyper parameters
        self.params = self.__dict__.copy()

        # Initialize zero memory [s, [a, r], s_]
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))

        ''' Tensorflow placeholders '''
        self.state = tf.placeholder(tf.float32, [None, n_states])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.reward = tf.placeholder(tf.float32, [None, ])
        self.state_ = tf.placeholder(tf.float32, [None, n_states])

        # Set seeds
        tf.set_random_seed(1)
        np.random.seed(1)

        with tf.variable_scope('q'):        # evaluation network
            l_eval = tf.layers.dense(self.state, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(l_eval, n_actions, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('q_next'):   # target network, not to train
            l_target = tf.layers.dense(self.state_, 10, tf.nn.relu, trainable=False)
            self.q_next = tf.layers.dense(l_target, n_actions, trainable=False)

        q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1)    # shape=(None, )

        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)                    # shape=(None, ), q for current state

        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.state: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

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
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :self.n_states]
        b_a = b_memory[:, self.n_states].astype(int)
        b_r = b_memory[:, self.n_states+1]
        b_s_ = b_memory[:, -self.n_states:]
        self.sess.run(self.train_op, {self.state: b_s, self.action: b_a, self.reward: b_r, self.state_: b_s_})

    def show_parameters(self):
        ''' Helper function to show the hyper parameters '''
        for key, value in self.params.items():
            print key, '=', value
