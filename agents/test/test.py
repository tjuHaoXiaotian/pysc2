
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.python.ops.control_flow_ops import switch,merge

class Actor(object):
    # TODO: update 输出（多agent输出）
    learning_rate = 1e-4
    TAU = 0.01 # soft replacement

    def __init__(self, session, agent_num, online_state_input, target_state_input, s_images_dim, action_dim, action_bound, cond_training_q):
        '''
        构造函数
        :param session: tensorflow session
        :param agent_num:  agent 个数
        :param online_state_input:  training 网络输入
        :param target_state_input:  target 网络输入: 这里指的就是 S'
        :param s_images:  state 图片输入
        :param action_dim: 每个agent的动作维度
        :param action_bound: 对应动作的每个维度上的尺度
        :param cond_training_q: 是否是训练 Q 网络
        '''
        self.session = session
        self.agent_num = agent_num
        self.online_state_input = online_state_input
        self.target_state_input = target_state_input
        self.s_images_dim = s_images_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.cond_training_q = cond_training_q

        with tf.variable_scope("Actor"):
            # online actor
            self.online_action_outputs = self._build_net(self.online_state_input,"online_actor",trainable=True)
            self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Actor/online_actor')
            # target actor : 输入的是 S' 输出 a'
            self.target_action_outputs = self._build_net(self.target_state_input,"target_actor",trainable=False)
            self.target_policy_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Actor/target_actor')

        pass

    def choose_action_online(self, s):
        '''
        single state -> single action
        :param s: single state
        :return: actions
        '''
        s = s[np.newaxis,:]
        return self.session.run(self.online_action_outputs, feed_dict={self.online_state_input: s})[0]

    def add_grad_to_graph(self, q_to_a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.online_action_outputs, xs=self.online_policy_net_vars, grad_ys=q_to_a_grads)

        with tf.variable_scope('actor_train'):
            opt = tf.train.RMSPropOptimizer(-self.learning_rate) # (- learning rate) for ascent policy
            grads_and_vars = zip(self.policy_grads, self.online_policy_net_vars)
            self.train_optimizer = opt.apply_gradients(grads_and_vars)

    def training(self, s):
        '''
        batch training update
        :param s: batch state
        :return:
        '''
        self.session.run(self.train_optimizer, feed_dict={
            self.online_state_input: s,
            self.cond_training_q:False
        })

        # target net replacement
        self.soft_replace = [[tf.assign(at, (1 - self.TAU) * at + self.TAU * aon)]
                             for aon, at,in zip(self.online_policy_net_vars, self.target_policy_net_vars)]
        pass

    def _build_net(self, state_inputs, scope, trainable):
        with tf.variable_scope(scope):
            # input: 84 * 84 * 3
            with tf.variable_scope("conv1"):
                # conv 1
                filter1 = self._weight_variable([8,8,self.s_images_dim,16], trainable)
                b1 = self._bias_variable([16], trainable)
                conv1 = tf.nn.relu(self._conv2d(state_inputs, filter1, stride=[1,4,4,1]) + b1)
            # conv1: 20 * 20 * 16
            with tf.variable_scope("conv2"):
                # conv 2
                filter2 = self._weight_variable([4, 4, 16, 32], trainable)
                b2 = self._bias_variable([32], trainable)
                conv2 = tf.nn.relu(self._conv2d(conv1, filter2, stride=[1, 2, 2, 1]) + b2)
                # max pooling
                max_pool2 = self._max_pooling(conv2)
            # conv2: 9 * 9 * 32
            # max_pool2: 5 * 5 * 32 = 800
            with tf.variable_scope("full_con"):
                flat = tf.reshape(max_pool2,[-1,5*5*32])
                w_full = self._weight_variable([5*5*32, 1024], trainable)
                b_full = self._bias_variable([1024], trainable)
                full1 = tf.nn.relu(tf.matmul(flat, w_full) + b_full)
            # full_con: 1024
            with tf.variable_scope("ouput"):
                actions = []
                for agent in range(self.agent_num):
                    with tf.variable_scope("agent_{}".format(agent)):
                        # ouput: 3
                        w_outout = self._weight_variable([1024, self.action_dim], trainable)
                        b_output = self._bias_variable(self.action_dim, trainable)
                        out = tf.nn.sigmoid(tf.matmul(full1, w_outout) + b_output)
                        scaled_a = tf.multiply(out, self.action_bound, name="scaled_action")
                        actions.append(scaled_a)
        return actions

    def _weight_variable(self, shape, trainable):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1),trainable=trainable)

    def _bias_variable(self, shape, trainable):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def _conv2d(self, x, w, stride = (1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')

    def _max_pooling(self, x, ksize = (1, 2, 2, 1), strides = (1,2,2,1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')



class Critic(object):

    gamma = 0.8
    learning_rate = 1e-4
    TAU = 0.01 # soft replacement

    def __init__(self, session, agent_num, online_state_input, batch_action, online_action_outputs, cond_training_q, target_state_input, target_action, batch_reward, batch_label, s_images_dim, action_dim):
        '''
        构造函数
        :param session: tensorflow session
        :param agent_num:  agent 个数
        :param online_state_input:  training 网络输入
        :param batch_action:  online 采样 action （训练 Q）
        :param online_action_outputs:  online actor 输出的(训练 actor：求解对 a 的梯度)
        :param cond_training_q:  switch 是训练 q 还是 actor
        :param target_state_input:  target 网络输入：这里指的就是 S'
        :param target_action:  target actor 网络输出: 输入 S' 输出 a'
        :param s_images_dim:  state 图片厚度
        :param action_dim: 每个agent的动作维度
        '''
        self.session = session
        self.agent_num = agent_num
        self.online_state_input = online_state_input

        self.batch_action = batch_action
        self.online_action_outputs = online_action_outputs
        self.cond_training_q = cond_training_q

        self.target_state_input = target_state_input
        self.target_action = target_action
        self.batch_reward = batch_reward
        # TODO: 多 agent q
        self.batch_label = batch_label  # just a placeholder
        self.s_images_dim = s_images_dim
        self.action_dim = action_dim


        with tf.variable_scope("Critic"):
            # online actor
            self.online_q_outputs = self._build_net(self.online_state_input, self.batch_action, "online_q", trainable=True, online_action_outputs=self.online_action_outputs,cond_training_q=self.cond_training_q)
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='Critic/online_q')
            # target actor
            self.target_q_outputs = self._build_net(self.target_state_input, self.target_action, "target_q", trainable=False)
            self.target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='Critic/target_q')
        # with tf.variable_scope('target_q'):
        #     # TODO: 多 agent q
        #     self.batch_label = batch_label

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.batch_label, tf.concat(self.online_q_outputs,axis=1)),axis=1))

        with tf.variable_scope('critic_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('q_to_a_grads'):
            # TODO: 注意 此处的 a 应该是 online actor 输出的 a
            self.q_to_a_grads = tf.gradients(self.online_q_outputs, self.batch_action)[0]  # tensor of gradients of each sample (None, a_dim)

    pass

    def training(self, s, a, r, s_, end):
        if end:
            y_batch = r
        else:
            # TODO: feeddict中有待计算数值
            target_q = self.session.run(self.target_q_outputs, feed_dict={
                self.target_state_input: s_,
            })
            y_batch = r + self.gamma * target_q

        self.session.run(self.train_op, feed_dict={
            self.online_state_input: s,
            self.batch_action: a,
            self.target_state_input: s_,
            self.batch_label: y_batch,
            self.cond_training_q: True,
        })

        # target net replacement
        self.soft_replace = [[tf.assign(ct, (1 - self.TAU) * ct + self.TAU * con)]
                             for con, ct,in zip(self.online_q_net_vars, self.target_q_net_vars)]

        pass

    def _build_net(self, state_inputs, batch_actions, scope, trainable,
                   online_action_outputs=None, # None for target net. (更新 actor时，用到了，对 a 求导 )
                   cond_training_q=None # bool to control switch. can be None for target net.
                     ):
        '''
        :param state_inputs:
        :param batch_actions:  两个作用：(1) 训练Q，批量抽样 action 的 placeholder (2) target 网络， a' 的引用
        :param scope:
        :param trainable:
        :param online_action_outputs: # None for target net. (更新 actor时，用到了，对 a (action 网络的输出) 求导
        :param cond_training_q:
        :return:
        '''
        with tf.variable_scope(scope):
            # input: 84 * 84 * 3
            with tf.variable_scope("conv1"):
                # conv 1
                filter1 = self._weight_variable([8,8,self.s_images_dim,16], trainable)
                b1 = self._bias_variable([16], trainable)
                conv1 = tf.nn.relu(self._conv2d(state_inputs, filter1, stride=[1,4,4,1]) + b1)
            # conv1: 20 * 20 * 16
            with tf.variable_scope("conv2"):
                # conv 2
                filter2 = self._weight_variable([4, 4, 16, 32], trainable)
                b2 = self._bias_variable([32], trainable)
                conv2 = tf.nn.relu(self._conv2d(conv1, filter2, stride=[1, 2, 2, 1]) + b2)
                # max pooling
                max_pool2 = self._max_pooling(conv2)
            # conv2: 9 * 9 * 32
            # max_pool2: 5 * 5 * 32 = 800
            with tf.variable_scope("full_con"):
                flat = tf.reshape(max_pool2,[-1,5*5*32])
                full_cons_1 = []
                for agent in range(self.agent_num):
                    with tf.variable_scope("agent_{}".format(agent)):
                        if cond_training_q is None:  # target net
                            actions = batch_actions
                        else:
                            # TODO: （后续可以调整）将输出图形dense后 与 每个agent对应输入对应拼接起来
                            # switch return :(output_false, output_true)
                            (_, sw_action_training_q) = switch(data=batch_actions,
                                                               pred=cond_training_q,
                                                               name='switch_actions_training_q')
                            (sw_action_training_policy, _) = switch(data=online_action_outputs,
                                                                    pred=cond_training_q,
                                                                    name='switch_actions_training_policy')
                            (actions, _) = merge([sw_action_training_q, sw_action_training_policy])


                        agent_dense = tf.concat([flat,actions[agent]], axis=1)

                        w_full = self._weight_variable([5*5*32 + self.action_dim, 1024], trainable)
                        b_full = self._bias_variable([1024], trainable)
                        agent_full1 = tf.nn.relu(tf.matmul(agent_dense, w_full) + b_full)
                        full_cons_1.append(agent_full1)

            # full_con: 1024
            with tf.variable_scope("full_con2"):
                full_cons_2 = []
                for agent in range(self.agent_num):
                    with tf.variable_scope("agent_{}".format(agent)):
                        # ouput: 3
                        w_full2 = self._weight_variable([1024, 128], trainable)
                        b_full2 = self._bias_variable(128, trainable)
                        agent_full2 = tf.nn.sigmoid(tf.matmul(full_cons_1[agent], w_full2) + b_full2)
                        full_cons_2.append(agent_full2)

            # full_con2: 128
            with tf.variable_scope("ouput"):
                ouputs = []
                for agent in range(self.agent_num):
                    with tf.variable_scope("agent_{}".format(agent)):
                        # ouput: 3
                        w_outout = self._weight_variable([128, 1], trainable)
                        b_output = self._bias_variable(1, trainable)
                        out = tf.matmul(full_cons_2[agent], w_outout) + b_output
                        ouputs.append(out)
            # ouput: 1
        return ouputs

    def _weight_variable(self, shape, trainable):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1),trainable=trainable)

    def _bias_variable(self, shape, trainable):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def _conv2d(self, x, w, stride = (1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')

    def _max_pooling(self, x, ksize = (1, 2, 2, 1), strides = (1,2,2,1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')

class DDPG(object):

    MAX_EPISODES = 30

    def __init__(self, agent_num, input_images_shape, action_dim, action_bound):
        self.agent_num = agent_num
        self.input_images_shape = input_images_shape
        self.action_dim = action_dim
        self.action_bound = action_bound
        # replay buffer
        self.replay_buffer = deque()
        self.sess = tf.Session()

        with tf.name_scope("S"):
            self.s = tf.placeholder(tf.int32, shape=[None,self.input_images_shape], name="s")

        with tf.name_scope("S_"):
            self.s_ = tf.placeholder(tf.int32, shape=[None,self.input_images_shape], name="s_")

        with tf.name_scope("R"):
            self.r = tf.placeholder(tf.float32, shape=[None,self.agent_num], name="r")

        with tf.name_scope("Y"):
            self.y = tf.placeholder(tf.float32, shape=[None,self.agent_num], name="y")

        with tf.name_scope("batch_action"):
            # 训练q 网络时的action输入
            self.batch_action = tf.placeholder(tf.float32,shape=(None, self.agent_num, self.action_dim),
                                                             name='batch_action'
                                                             )
        # 用于控制q 网络action输入的条件变量：
        # True: training q .
        # False: training policy.
        with tf.name_scope("cond_training_q"):
            self.cond_training_q = tf.placeholder(tf.bool, shape=[], name='cond_training_q')

        actor = Actor(self.sess,self.agent_num,self.s,self.s_,self.input_images_shape[2],self.action_dim,self.action_bound)
        critic = Critic(self.sess,self.agent_num,self.s, self.batch_action, actor.online_action_outputs, self.cond_training_q, self.s_, actor.target_action_outputs,self.r, self.y,self.input_images_shape[2],self.action_dim)
        actor.add_grad_to_graph(critic.q_to_a_grads)

        # saver
        self.saver = tf.train.Saver()
        path = "./test"

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        pass