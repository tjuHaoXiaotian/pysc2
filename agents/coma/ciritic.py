# -*- coding: UTF-8 -*-
import tensorflow as tf
from pysc2.agents.coma.common import soft_update_online_to_target, copy_online_to_target
COMA_CFG = tf.app.flags.FLAGS  # alias

class Critic(object):

    def __init__(self, sess, action_dim, state_dim, agent_num, use_batch_norm):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_batch_norm = use_batch_norm
        self.is_training = tf.placeholder(tf.bool, name="is_training")


        self.critic_state_input = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim * 13], name='critic_state_input')

        # TODO: action 输入的顺序怎么定？ 按距离排序了
        # 其他单位在 s 下选择的动作
        self.other_units_action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim * (self.agent_num-1)], name='other_units_action_input')
        # 自己当时执行的动作
        self.self_action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='self_action_input')
        # actor 输出的执行各个动作概率
        self.actor_output_probability = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actor_output_probability')

        self.Q_value_label_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Q_value_label_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope("Critic"):
            # online actor
            self.online_q_outputs = self._build_net(self.critic_state_input, self.other_units_action_input, "online_q", trainable=True)
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Critic/online_q')
            self.online_policy_net_vars_by_name = {var.name.strip('Critic/online'): var
                                                   for var in self.online_q_net_vars}
            # target actor
            self.target_q_outputs = self._build_net(self.critic_state_input, self.other_units_action_input, "target_q", trainable=False)
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
        # define the advantage function
        self._build_advantage()

    def _build_net(self, critic_state_inputs, other_units_action_input, scope, trainable):
        '''
        :param critic_state_inputs:
        :param other_units_action_input:
        :param scope:
        :param trainable:
        :return:
        '''
        with tf.variable_scope(scope):
            # TODO: 目前是分别计算，求和累加
            layer1_action = self._fully_connected(other_units_action_input, [self.action_dim * (self.agent_num-1), 1024], [1024], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer1_action", trainable=trainable)
            layer1_state = self._fully_connected(critic_state_inputs, [self.state_dim * 13, 1024], [1024], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer1_state", trainable=trainable)
            layer2 = self._fully_connected(tf.add(layer1_action, layer1_state), [1024, 256], [256], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer2", trainable=trainable)
            layer3 = self._fully_connected(layer2, [256, 128], [128], activation_fn=tf.nn.elu,
                                           variable_scope_name="layer3", trainable=trainable)
            out = self._fully_connected(layer3, [128, self.action_dim], [self.action_dim], activation_fn=None,
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
                                  self.reward + COMA_CFG.GAMMA * tf.reduce_max(self.target_q_outputs, keep_dims=True, axis=1))

    def _build_cost_graph(self):
        # 批量计算执行 ai 的 Q(S, a-i, ai)
        online_output_q = tf.reduce_sum(tf.multiply(self.online_q_outputs, self.self_action_input),keep_dims=True,axis = 1)
        self.cost = tf.reduce_mean(tf.square(self.Q_value_label_input - online_output_q))

        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='Critic')):
                self.train = tf.train.AdamOptimizer(COMA_CFG.q_learning_rate).minimize(self.cost)
        else:
            self.train = tf.train.AdamOptimizer(COMA_CFG.q_learning_rate).minimize(self.cost)

    def _build_advantage(self):
        self.advantage = tf.reduce_sum(self.online_q_outputs * self.self_action_input, keep_dims=True, axis=1) - tf.reduce_sum(self.online_q_outputs * self.actor_output_probability, keep_dims=True, axis=1)


    def operation_get_TDtarget(self, state_next, action_next, reward, terminal, is_training):
        '''
        Training to get td target
        :param action_next:  target actor output
        :param state_next:
        :param reward:
        :param terminal:
        :return:
        '''
        return self.sess.run(self.td_target, feed_dict={
                                                        self.critic_state_input: state_next,
                                                        self.other_units_action_input: action_next,  # 应该是 target actor output
                                                        self.reward: reward,
                                                        self.terminal: terminal,
                                                        self.is_training: is_training
                                                        })

    def operation_cal_advantage(self, state_input, action_others, self_action_input, actor_output_probability, is_training):
        batch_advantages = self.sess.run(self.advantage,
                          feed_dict = {
                              self.critic_state_input: state_input,
                              self.other_units_action_input: action_others,
                              self.self_action_input: self_action_input,
                              self.actor_output_probability: actor_output_probability,
                              self.is_training: is_training
                          })
        return batch_advantages

    def operation_critic_learn(self, state, other_unit_actions, self_action, TDtarget, is_training):
        '''
        Training the critic network
        :param TDtarget: the target label (calculated by self.operation_get_TDtarget())
        :param action: the batch action input which is sampled from the replay_buffer
        :param state: the batch sate input which is sampled from the replay_buffer
        :return:
        '''
        _ = self.sess.run(self.train,
                          feed_dict={
                                     self.critic_state_input: state,
                                     self.other_units_action_input: other_unit_actions,
                                     self.self_action_input: self_action,
                                     self.Q_value_label_input: TDtarget,
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