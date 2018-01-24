# -*- coding: UTF-8 -*-
import tensorflow as tf
from pysc2.agents.ddpg.common import *

DDPG_CFG = tf.app.flags.FLAGS  # alias

# 每个 unit 动作都一样
class Actor(object):
    def __init__(self, sess, action_dim, state_dim, use_batch_norm):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_batch_norm = use_batch_norm
        self.is_training = tf.placeholder(tf.bool, name="is_training")


        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim * 14], name="actor_state_input")
        self.action_gradient_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name="action_gradient_input")

        with tf.variable_scope("Actor"):
            # online actor
            self.online_action_outputs = self._build_net(self.state_input,"online_actor",trainable=True)
            self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Actor/online_actor')

            self.online_policy_net_vars_by_name = {var.name.strip('Actor/online'): var
                                                   for var in self.online_policy_net_vars}

            # target actor : 输入的是 S' 输出 a'
            self.target_action_outputs = self._build_net(self.state_input,"target_actor",trainable=False)
            self.target_policy_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Actor/target_actor')

            self.target_policy_net_vars_by_name = {var.name.strip('Actor/target'): var
                                                   for var in self.target_policy_net_vars}


        self._build_update_graph()
        self._build_cost_graph()



    def _build_net(self, state_inputs, scope, trainable):
        # input layer： 8 + 8 * 13 = 104 + 8 = 112
        # hidden layer 1: 1024
        # hidden layer 2: 128
        # output layer: 3
        with tf.variable_scope(scope):
            layer1 = self._fully_connected(state_inputs,[8 * 13 + 8, 1024],[1024],activation_fn=tf.nn.elu,variable_scope_name="layer1",trainable=trainable)
            layer2 = self._fully_connected(layer1,[1024, 256],[256],activation_fn=tf.nn.elu,variable_scope_name="layer2",trainable=trainable)
            layer3 = self._fully_connected(layer2,[256, 128],[128],activation_fn=tf.nn.elu,variable_scope_name="layer3",trainable=trainable)
            out = self._fully_connected(layer3,[128,self.action_dim],[self.action_dim],activation_fn=tf.nn.sigmoid,variable_scope_name="out",trainable=trainable)
        return out

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                        self.target_policy_net_vars_by_name)

        # target net soft replacement
        self.soft_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                        self.target_policy_net_vars_by_name)

    def _build_cost_graph(self):
        # batch 维度上求平均
        self.cost = tf.reduce_mean(
            # action 维度上求和
            tf.reduce_sum(self.action_gradient_input * self.online_action_outputs, axis=1)
        )  # [batch_size,1]*[batch_size,1]
        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Actor')):
                # must has a negtive learning_rate (to maximize the loss function)
                self.train = tf.train.AdamOptimizer(-DDPG_CFG.learning_rate).minimize(self.cost)
        else:
            # must has a negtive learning_rate (to maximize the loss function)
            self.train = tf.train.AdamOptimizer(-DDPG_CFG.learning_rate).minimize(self.cost)


    def operation_get_action_to_environment(self, state, is_training):
        '''
        Calculate the online action output
        :param state: online state our batch state
        :return:
        '''
        return self.sess.run(self.online_action_outputs,feed_dict={
            self.state_input:state,
            self.is_training: is_training
        })

    def operation_get_action_to_TDtarget(self, state, is_training):
        '''
        Calculate the target actor net action output (just used for the training the Q network)
        :param state: net state batch (sampled from the replay buffer)
        :return:
        '''
        return self.sess.run(self.target_action_outputs, feed_dict={
            self.state_input: state,
            self.is_training: is_training
        })

    def operation_actor_learn(self, gradient, state, is_training):
        '''
        Traning the actor network
        :param gradient: the gradient from Q to actor ouput action(note: the actor ouput is calculated from state batch),
        :param state: state batch (sampled from the replay buffer)
        :return:
        '''
        self.sess.run(self.train,feed_dict={
            self.action_gradient_input:gradient,
            self.state_input:state,
            self.is_training: is_training
        })

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

    def _fully_connected(self, layer_in, weights_shape, biases_shape, activation_fn=None, variable_scope_name="layer", trainable = True):
        if self.use_batch_norm and activation_fn:
            with tf.variable_scope(variable_scope_name):
                weights = self._weight_variable(weights_shape, trainable)
                # Batch normalization uses weights as usual, but does NOT add a bias term. This is because
                # its calculations include gamma and beta variables that make the bias term unnecessary.
                linear_output = tf.matmul(layer_in, weights)
                # Apply batch normalization to the linear combination of the inputs and weights
                batch_normalized_output = tf.layers.batch_normalization(linear_output, training=self.is_training)
                return activation_fn(batch_normalized_output)
        else:
            with tf.variable_scope(variable_scope_name):
                weights = self._weight_variable(weights_shape, trainable)
                biases = self._bias_variable(biases_shape, trainable)
                linear_output = tf.add(tf.matmul(layer_in, weights), biases)
            return linear_output if not activation_fn else activation_fn(linear_output)

    def _weight_variable(self, shape, trainable, name='weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name)
    def _bias_variable(self, shape, trainable, name="bias"):
        return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape), trainable=trainable, name = name)
    def _conv2d(self, x, w, stride = (1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')
    def _max_pooling(self, x, ksize = (1, 2, 2, 1), strides = (1,2,2,1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')