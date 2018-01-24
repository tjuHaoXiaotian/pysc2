# -*- coding: UTF-8 -*-
import tensorflow as tf
from pysc2.agents.ddpg.common import *
DDPG_CFG = tf.app.flags.FLAGS  # alias

class Critic(object):

    def __init__(self, sess, action_dim, state_dim, agent_num, use_batch_norm):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_batch_norm = use_batch_norm
        self.is_training = tf.placeholder(tf.bool, name="is_training")


        self.critic_state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim * 13], name='critic_state_input')
        self.actor_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim * self.agent_num], name='actor_input')

        self.Q_value_label_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Q_value_label_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope("Critic"):
            # online actor
            self.online_q_outputs = self._build_net(self.critic_state_input, self.actor_input, "online_q", trainable=True)
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Critic/online_q')
            self.online_policy_net_vars_by_name = {var.name.strip('Critic/online'): var
                                                   for var in self.online_q_net_vars}
            # target actor
            self.target_q_outputs = self._build_net(self.critic_state_input, self.actor_input, "target_q", trainable=False)
            self.target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Critic/target_q')
            self.target_policy_net_vars_by_name = {var.name.strip('Critic/target'): var
                                                   for var in self.target_q_net_vars}

        # define hard replacement and soft replacement
        self._build_update_graph()
        # define the target label of current Q value
        self._build_td_target_graph()
        # define the cost function
        self._build_cost_graph()
        # define the gradient of Q to actor_output
        self._build_gradient_graph()

    def _build_net(self, critic_state_inputs, actor_input, scope, trainable):
        '''
        :param critic_state_inputs:
        :param actor_input:
        :param scope:
        :param trainable:
        :return:
        '''
        with tf.variable_scope(scope):
            agent_dense = tf.concat([critic_state_inputs, actor_input], axis=1)
            layer1 = self._fully_connected(agent_dense, [self.state_dim * 13 + self.action_dim * self.agent_num, 1024], [1024], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer1", trainable=trainable)
            layer2 = self._fully_connected(layer1, [1024, 256], [256], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer2", trainable=trainable)
            layer3 = self._fully_connected(layer2, [256, 128], [128], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer3", trainable=trainable)
            out = self._fully_connected(layer3, [128, 1], [1], activation_fn=None,
                                        variable_scope_name="out", trainable=trainable)
        return out

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

        # target net soft replacement
        self.soft_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

    def _build_td_target_graph(self):
        self.td_target = tf.where(self.terminal,
                                  self.reward,
                                  self.reward + DDPG_CFG.GAMMA * self.target_q_outputs)

    def _build_cost_graph(self):
        self.cost = tf.reduce_mean(tf.square(self.Q_value_label_input - self.online_q_outputs))

        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='Critic')):
                self.train = tf.train.AdamOptimizer(DDPG_CFG.q_learning_rate).minimize(self.cost)
        else:
            self.train = tf.train.AdamOptimizer(DDPG_CFG.q_learning_rate).minimize(self.cost)
        # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def _build_gradient_graph(self):
        gradient_temp = tf.gradients(self.online_q_outputs, self.actor_input)
        # print("Q 对 actions 的梯度是：", gradient_temp[0].get_shape())
        # output's shape is [1,batch_size,4]????
        # self.gradient = tf.reshape(gradient_temp, (-1, self.action_dim))
        print("梯度导数：", gradient_temp)
        self.gradient = gradient_temp[0]

    def operation_get_TDtarget(self, action_next, state_next, reward, terminal, is_training):
        '''
        Training to get td target
        :param action_next:  target actor output
        :param state_next:
        :param reward:
        :param terminal:
        :return:
        '''
        return self.sess.run(self.td_target, feed_dict={self.actor_input: action_next, # 应该是 target actor output
                                                        self.critic_state_input: state_next,
                                                        self.reward: reward,
                                                        self.terminal: terminal,
                                                        self.is_training: is_training
                                                        })

    def operation_critic_learn(self, TDtarget, state, action,is_training):
        '''
        Training the critic network
        :param TDtarget: the target label (calculated by self.operation_get_TDtarget())
        :param action: the batch action input which is sampled from the replay_buffer
        :param state: the batch sate input which is sampled from the replay_buffer
        :return:
        '''
        _ = self.sess.run(self.train,
                          feed_dict={self.Q_value_label_input: TDtarget,
                                     self.critic_state_input: state,
                                     self.actor_input: action,
                                     self.is_training: is_training
                                     })

    def operation_get_gradient(self, state, action, baseline_action, columns_ids, is_training):
        '''
        Calculate the gradient from Q to actor ouput
        :param state: state: the batch sate input which is sampled from the replay_buffer
        :param action: actor ouput (calculated from batch state: state)
        :return:
        '''

        part1 = self.sess.run(self.gradient, feed_dict={
            self.critic_state_input: state,
            self.actor_input: action,
            self.is_training: is_training
        })
        part2 = self.sess.run(self.gradient, feed_dict={
            self.critic_state_input: state,
            self.actor_input: baseline_action,
            self.is_training: is_training
        })

        result = part1 - part2
        gradients = []
        for row in range(len(columns_ids)):
            gradients.append(result[row][columns_ids[row] * 3:columns_ids[row] * 3+3])
        return gradients

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

    def _fully_connected(self, layer_in, weights_shape, biases_shape, activation_fn=None, variable_scope_name="layer",
                         trainable=True):
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
        return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape), trainable=trainable, name=name)

    def _conv2d(self, x, w, stride=(1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')

    def _max_pooling(self, x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')