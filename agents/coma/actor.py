# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from pysc2.agents.coma.common import soft_update_online_to_target, copy_online_to_target

COMA_CFG = tf.app.flags.FLAGS  # alias

# 每个 unit 动作都一样
class Actor(object):
    def __init__(self, sess, action_dim, state_dim, use_batch_norm):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_batch_norm = use_batch_norm
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim * 13], name="state_input")
        # 实际执行的动作，也就是对应actor要更新的输出 Notice: 这里已经 one-hot了
        self.execute_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name="execute_action")
        # 执行上述动作的 advantage
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantage")

        with tf.variable_scope("Actor"):
            # online actor
            self.softmax_action_outputs = self._build_net(self.state_input,"online_actor",trainable=True)
            eps = 1e-10
            y_clip = tf.clip_by_value(self.softmax_action_outputs, eps, 1.0 - eps)
            self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(y_clip*tf.log(y_clip),axis=1))
            # self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=y_clip,dim=-1))
            self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Actor/online_actor')

            self.online_policy_net_vars_by_name = {var.name.strip('Actor/online'): var
                                                   for var in self.online_policy_net_vars}

            # target actor : 输入的是 S' 输出 a'
            self.target_softmax_action_outputs = self._build_net(self.state_input, "target_actor", trainable=False)
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
            # ==================================== Basic LSTM ==================================
            # 这里暂时把所有单位由远及近 当成时序处理
            state_inputs = tf.reshape(state_inputs, [-1, COMA_CFG.seq_length, COMA_CFG.state_dim])

            lstm  = tf.contrib.rnn.BasicLSTMCell(COMA_CFG.lstm_size, forget_bias=1.0, state_is_tuple=True)
            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
            # Stack up multiple LSTM layers
            cell = tf.contrib.rnn.MultiRNNCell([drop] * COMA_CFG.lstm_layer)
            # Getting an initial state of all zeros
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            # initial_state = self.initial_state
            # outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell, inputs=state_inputs, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell, inputs=state_inputs, initial_state=self.initial_state)

            #直接调用final_state 中的 h_state (final_state[1]) 来进行运算:
            actions_probability = self._fully_connected(outputs[:, -1, :], [COMA_CFG.lstm_size, self.action_dim], [self.action_dim],
                                                        activation_fn=tf.nn.softmax, variable_scope_name="output",
                                                        trainable=trainable)

            # ==================================== linear layers =================================================
            # # activation_func = tf.nn.elu
            # activation_func = tf.nn.relu
            # # activation_func = tf.nn.tanh
            # layer1 = self._fully_connected(state_inputs,[self.state_dim * 13, 1024],[1024],activation_fn=activation_func,variable_scope_name="layer1",trainable=trainable)
            # layer2 = self._fully_connected(layer1,[1024, 256],[256],activation_fn=activation_func,variable_scope_name="layer2",trainable=trainable)
            # layer3 = self._fully_connected(layer2,[256, 128],[128],activation_fn=activation_func,variable_scope_name="layer3",trainable=trainable)
            # # logits = self._fully_connected(layer3,[128,self.action_dim],[self.action_dim],activation_fn=None,variable_scope_name="logits",trainable=trainable)
            # # actions_probability = tf.nn.softmax(logits, dim=-1, name="softmax")
            # actions_probability = self._fully_connected(layer3,[128,self.action_dim],[self.action_dim],activation_fn=tf.nn.softmax,variable_scope_name="logits",trainable=trainable)
            #
            # # 对 logits 归一化就不会越界了
            # # logits_max = tf.reduce_max(logits, axis=1, keep_dims=True)
            # # logits_normalized = logits / logits_max
            # # actions_probability = tf.nn.softmax(logits_normalized, dim=-1, name="softmax")
            #
            # # with tf.name_scope("actor/softmax"):
            # #     tf.summary.histogram('actor/softmax', actions_probability)  # Tensorflow >= 0.12

        return actions_probability

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
            # action 维度上求和（只留下对应执行的动作维度）
            tf.log(tf.reduce_sum(self.softmax_action_outputs * self.execute_action, keep_dims=True, axis=1)) * self.advantage
        )
        
        self.reduce_entropy = COMA_CFG.entropy_regularizer_lambda * self.entropy_loss
        with tf.name_scope("actor/loss"):
            self.total_cost = -(self.cost+self.reduce_entropy)
            tf.summary.scalar('actor_total_loss', self.total_cost)  # tensorflow >= 0.12
            tf.summary.scalar('actor_loss', -self.cost)  # tensorflow >= 0.12

        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Actor')):
                # For RNN do clip
                # Optimizer
                optimizer = tf.train.AdamOptimizer(COMA_CFG.learning_rate)
                # Gradient Clipping
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='Actor/online_actor')
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_cost, tvars), COMA_CFG.grad_clip)
                self.train = optimizer.apply_gradients(zip(grads, tvars))

                # Linear layer
                # self.train = tf.train.AdamOptimizer(COMA_CFG.learning_rate).minimize(self.total_cost

        else:
            # For RNN do clip
            # Optimizer
            optimizer = tf.train.AdamOptimizer(COMA_CFG.learning_rate)
            # Gradient Clipping
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='Actor/online_actor')
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_cost, tvars), COMA_CFG.grad_clip)
            self.train = optimizer.apply_gradients(zip(grads, tvars))

            # Linear layer
            # self.train = tf.train.AdamOptimizer(COMA_CFG.learning_rate).minimize(self.total_cost)



    # def operation_get_action_to_environment(self, state, is_training):
    #     '''
    #     Calculate the online action output
    #     :param state: online state our batch state
    #     :return:
    #     '''
    #     return self.sess.run(self.softmax_action_outputs,feed_dict={
    #         self.state_input:state,
    #         self.is_training: is_training
    #     })
    def operation_cal_softmax_probablility(self, batch_size, state, is_training):
        '''
        :param batch_size: RNN batch size
        :param state:
        :param is_training:
        :return:
        '''
        prob_weights = self.sess.run(self.softmax_action_outputs, feed_dict={
            self.batch_size: batch_size,
            self.state_input: state,
            self.is_training: is_training,
            self.keep_prob: 1.,
            # self.initial_state: new_cell_state
        })
        return prob_weights

    @staticmethod
    def has_nan(x):
        test = x != x
        return np.sum(test) > 0

    # 定义如何选择行为，即状态ｓ处的行为采样.根据当前的行为概率分布进行采样
    def operation_choose_action(self, batch_size, state, is_training):
        '''
        :param batch_size: RNN batch size
        :param state:
        :param is_training:
        :return:
        '''
        prob_weights = self.sess.run(self.softmax_action_outputs,feed_dict={
            self.batch_size: batch_size,
            self.state_input:state,
            self.is_training: is_training,
            self.keep_prob: 1.,
            # self.initial_state: new_cell_state
        })
        # if self.has_nan(prob_weights):
        # print(prob_weights)

        # 按照给定的概率采样
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, prob_weights

    def operation_greedy_action(self, batch_size, state, is_training):
        '''
        :param batch_size: RNN batch size
        :param state:
        :param is_training:
        :return:
        '''
        # observation[np.newaxis, :]
        prob_weights = self.sess.run(self.target_softmax_action_outputs,feed_dict={
            self.batch_size: batch_size,
            self.state_input: state,
            self.is_training: is_training,
            self.keep_prob: 1.,
            # self.initial_state: new_cell_state
        })
        action = np.argmax(prob_weights, axis=1)
        return action



    def operation_actor_learn(self, batch_size, state, execute_action, advantage, is_training):
        '''
        Traning the actor network
        :param batch_size: RNN batch size
        :param state: state batch (sampled from the replay buffer)
        :param execute_action: action batch (sampled from the replay buffer, executed at that timestep)
        :param advantage: calculated advantage
        :param is_training:
        :return:
        '''
        _, cost, final_state = self.sess.run([self.train, self.total_cost, self.final_state],feed_dict={
            self.batch_size: batch_size,
            self.state_input: state,
            self.execute_action: execute_action,
            self.advantage: advantage,
            self.keep_prob: COMA_CFG.keep_prob,
            # self.initial_state: new_cell_state,
            self.is_training: is_training
        })
        # print("cost: ", cost)
        return cost, final_state

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
                biases = self._bias_variable(biases_shape, trainable)
                # Batch normalization uses weights as usual, but does NOT add a bias term. This is because
                # its calculations include gamma and beta variables that make the bias term unnecessary.
                linear_output = tf.add(tf.matmul(layer_in, weights), biases)
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
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name=name)
    def _bias_variable(self, shape, trainable, name="bias"):
        return tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=shape), trainable=trainable, name = name)
    def _conv2d(self, x, w, stride = (1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')
    def _max_pooling(self, x, ksize = (1, 2, 2, 1), strides = (1,2,2,1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')