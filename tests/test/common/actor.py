# -*- coding: UTF-8 -*-
import tensorflow as tf


# 每个 unit 动作都一样
class Actor(object):
    TAU = 0.01 # soft replacement
    learning_rate = 1e-4

    def __init__(self, sess, action_dim, state_dim, action_bound):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim[0],self.state_dim[1],self.state_dim[2]], name="stae_input")
        self.action_gradient_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name="action_gradient_input")

        with tf.variable_scope("Actor"):
            # online actor
            self.online_action_outputs = self._build_net(self.state_input,"online_actor",trainable=True)
            self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Actor/online_actor')
            # target actor : 输入的是 S' 输出 a'
            self.target_action_outputs = self._build_net(self.state_input,"target_actor",trainable=False)
            self.target_policy_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Actor/target_actor')

        self._build_update_graph()
        self._build_cost_graph()



    def _build_net(self, state_inputs, scope, trainable):
        with tf.variable_scope(scope):
            # input: 84 * 84 * 3
            with tf.variable_scope("conv1"):
                # conv 1
                filter1 = self._weight_variable([8, 8, self.state_dim[2], 16], trainable, name="filter1")
                b1 = self._bias_variable([16], trainable, name="bias1")
                conv1 = tf.nn.relu(self._conv2d(state_inputs, filter1, stride=[1, 4, 4, 1]) + b1)
            # conv1: 20 * 20 * 16
            with tf.variable_scope("conv2"):
                # conv 2
                filter2 = self._weight_variable([4, 4, 16, 32], trainable,name="filter2")
                b2 = self._bias_variable([32], trainable,name="bias2")
                conv2 = tf.nn.relu(self._conv2d(conv1, filter2, stride=[1, 2, 2, 1]) + b2)
                # max pooling
                max_pool2 = self._max_pooling(conv2)
            # conv2: 9 * 9 * 32
            # max_pool2: 5 * 5 * 32 = 800
            with tf.variable_scope("full_con"):
                flat = tf.reshape(max_pool2, [-1, 6 * 6 * 32])
                w_full = self._weight_variable([6 * 6 * 32, 1024], trainable, name="w3")
                b_full = self._bias_variable([1024], trainable, name="bias3")
                full1 = tf.nn.relu(tf.matmul(flat, w_full) + b_full)
            # full_con: 1024
            with tf.variable_scope("ouput"):
                # ouput: 3
                w_outout = self._weight_variable([1024, self.action_dim], trainable, name="w4")
                b_output = self._bias_variable([self.action_dim], trainable, name="bias4")
                out = tf.nn.sigmoid(tf.matmul(full1, w_outout) + b_output)
                scaled_a = tf.multiply(out, self.action_bound, name="scaled_action")

        # return tf.squeeze(scaled_a)
        return scaled_a

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = [[tf.assign(at, aon)]
                             for aon, at, in zip(self.online_policy_net_vars, self.target_policy_net_vars)]

        # target net soft replacement
        self.soft_replace = [[tf.assign(at, (1 - self.TAU) * at + self.TAU * aon)]
                             for aon, at, in zip(self.online_policy_net_vars, self.target_policy_net_vars)]

    def _build_cost_graph(self):
        # batch 维度上求平均
        self.cost = tf.reduce_mean(
            # action 维度上求和
            tf.reduce_sum(self.action_gradient_input * self.online_action_outputs, axis=1)
        )  # [batch_size,1]*[batch_size,1]

        # must has a negtive learning_rate (to maximize the loss function)
        self.train = tf.train.AdamOptimizer(-self.learning_rate).minimize(self.cost)


    def operation_get_action_to_environment(self,state):
        '''
        Calculate the online action output
        :param state: online state our batch state
        :return:
        '''
        # TODO: 这里添加了压缩维度
        return self.sess.run(tf.squeeze(self.online_action_outputs),feed_dict={self.state_input:state})

    def operation_get_action_to_TDtarget(self, state):
        '''
        Calculate the target actor net action output (just used for the training the Q network)
        :param state: net state batch (sampled from the replay buffer)
        :return:
        '''
        return self.sess.run(self.target_action_outputs, feed_dict={self.state_input: state})

    def operation_actor_learn(self,gradient,state):
        '''
        Traning the actor network
        :param gradient: the gradient from Q to actor ouput action(note: the actor ouput is calculated from state batch),
        :param state: state batch (sampled from the replay buffer)
        :return:
        '''
        self.sess.run(self.train,feed_dict={self.action_gradient_input:gradient,self.state_input:state})

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