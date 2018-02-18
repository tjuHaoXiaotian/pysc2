# -*- coding: UTF-8 -*-
import tensorflow as tf


class Critic(object):

    GAMMA = 0.95
    learning_rate = 1e-3
    TAU = 0.01 # soft replacement

    def __init__(self, sess, action_dim, state_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim[0],self.state_dim[1],self.state_dim[2]], name='state_input')
        self.actor_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actor_input')

        self.Q_value_label_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Q_value_label_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope("Critic"):
            # online actor
            self.online_q_outputs = self._build_net(self.state_input, self.actor_input, "online_q", trainable=True)
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Critic/online_q')
            # target actor
            self.target_q_outputs = self._build_net(self.state_input, self.actor_input, "target_q", trainable=False)
            self.target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Critic/target_q')

        # define hard replacement and soft replacement
        self._build_update_graph()
        # define the target label of current Q value
        self._build_td_target_graph()
        # define the cost function
        self._build_cost_graph()
        # define the gradient of Q to actor_output
        self._build_gradient_graph()

    def _build_net(self, state_inputs, actor_input, scope, trainable):
        '''
        :param state_inputs:
        :param actor_input:
        :param scope:
        :param trainable:
        :return:
        '''
        with tf.variable_scope(scope):
            # input: 84 * 84 * 3
            with tf.variable_scope("conv1"):
                # conv 1
                filter1 = self._weight_variable([8,8,self.state_dim[2],16], trainable, name="filter1")
                b1 = self._bias_variable([16], trainable, name="bias1")
                conv1 = tf.nn.relu(self._conv2d(state_inputs, filter1, stride=[1,4,4,1]) + b1)
                # conv1: 20 * 20 * 16
            with tf.variable_scope("conv2"):
                # conv 2
                filter2 = self._weight_variable([4, 4, 16, 32], trainable, name="filter2")
                b2 = self._bias_variable([32], trainable, name="bias2")
                conv2 = tf.nn.relu(self._conv2d(conv1, filter2, stride=[1, 2, 2, 1]) + b2)
                # conv2: 9 * 9 * 32
                # max pooling
                max_pool2 = self._max_pooling(conv2)
                # max_pool2: 5 * 5 * 32 = 800
            with tf.variable_scope("full_con"):
                flat = tf.reshape(max_pool2,[-1,6*6*32])
                # TODO: 这里只是把动作拼到 dense 后的 state 上了（尝试先处理，在拼接）
                agent_dense = tf.concat([flat, actor_input], axis=1)
                w_full = self._weight_variable([6*6*32 + self.action_dim, 1024], trainable, name="w3")
                b_full = self._bias_variable([1024], trainable, name="bias3")
                agent_full1 = tf.nn.relu(tf.matmul(agent_dense, w_full) + b_full)
                # full_con: 1024
            with tf.variable_scope("full_con2"):
                w_full2 = self._weight_variable([1024, 128], trainable, name="w4")
                b_full2 = self._bias_variable([128], trainable, name="bias4")
                agent_full2 = tf.nn.sigmoid(tf.matmul(agent_full1, w_full2) + b_full2)
                # full_con2: 128
            with tf.variable_scope("ouput"):
                w_outout = self._weight_variable([128, 1], trainable, name="w5")
                b_output = self._bias_variable([1], trainable, name="bias5")
                out_Q = tf.matmul(agent_full2, w_outout) + b_output
                # ouput: 1
        # return tf.squeeze(out_Q)
        return out_Q

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = [[tf.assign(at, aon)]
                             for aon, at, in zip(self.online_q_net_vars, self.target_q_net_vars)]

        # target net soft replacement
        self.soft_replace = [[tf.assign(at, (1 - self.TAU) * at + self.TAU * aon)]
                             for aon, at, in zip(self.online_q_net_vars, self.target_q_net_vars)]

    def _build_td_target_graph(self):
        self.td_target = tf.where(self.terminal,
                                  self.reward,
                                  self.reward + self.GAMMA * self.target_q_outputs)

    def _build_cost_graph(self):
        self.cost = tf.reduce_mean(tf.square(self.Q_value_label_input - self.online_q_outputs))

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def _build_gradient_graph(self):
        gradient_temp = tf.gradients(self.online_q_outputs, self.actor_input)
        # print("Q 对 actions 的梯度是：", gradient_temp[0].get_shape())
        # output's shape is [1,batch_size,4]????
        # self.gradient = tf.reshape(gradient_temp, (-1, self.action_dim))
        self.gradient = gradient_temp[0]

    def operation_get_TDtarget(self, action_next, state_next, reward, terminal):
        '''
        Training to get td target
        :param action_next:  target actor output
        :param state_next:
        :param reward:
        :param terminal:
        :return:
        '''
        return self.sess.run(self.td_target, feed_dict={self.actor_input: action_next, # 应该是 target actor output
                                                        self.state_input: state_next,
                                                        self.reward: reward,
                                                        self.terminal: terminal})

    def operation_critic_learn(self, TDtarget, state, action):
        '''
        Training the critic network
        :param TDtarget: the target label (calculated by self.operation_get_TDtarget())
        :param action: the batch action input which is sampled from the replay_buffer
        :param state: the batch sate input which is sampled from the replay_buffer
        :return:
        '''
        _ = self.sess.run(self.train,
                          feed_dict={self.Q_value_label_input: TDtarget,
                                     self.state_input: state,
                                     self.actor_input: action
                                     })

    def operation_get_gradient(self, state, action):
        '''
        Calculate the gradient from Q to actor ouput
        :param state: state: the batch sate input which is sampled from the replay_buffer
        :param action: actor ouput (calculated from batch state: state)
        :return:
        '''
        return self.sess.run(self.gradient, feed_dict={self.actor_input: action,
                                                       self.state_input: state})

    def operation_update_TDnet_compeletely(self):
        '''
        hard replacement
        :return:
        '''
        self.sess.run(self.hard_replace)

    def operation_soft_update_TDnet(self):
        '''
        soft replacement
        :return:
        '''
        self.sess.run(self.soft_replace)

    def _weight_variable(self, shape, trainable, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name)
    def _bias_variable(self, shape, trainable, name):
        return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape), trainable=trainable, name = name)
    def _conv2d(self, x, w, stride = (1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')
    def _max_pooling(self, x, ksize = (1, 2, 2, 1), strides = (1,2,2,1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')

