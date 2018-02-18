#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np
import time
import tensorflow as tf

# ==========================================features =================================================
# feature 玩家 0, 1
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
# feature 队友
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
# feature 单位种类
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
# feature Which units are selected.
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
# feature hit points
_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
# feature energy
_ENERGY = features.SCREEN_FEATURES.unit_energy.index
# feature TODO: effects 是啥
_EFFECTS = features.SCREEN_FEATURES.effects.index
# =========================================features end ==============================================

# ==========================================actions ==================================================
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_STOP_QUICK = actions.FUNCTIONS.Stop_quick.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
# _ATTACK_ATTACK_SCREEN = actions.FUNCTIONS.Attack_Attack_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# =========================================actions end ===============================================

# ==========================================parameters ===============================================
_NOT_QUEUED = [0]
_SELECT = [0]  # Select,Toggle,SelectAllOfType,AddAllType
_SCREEN_SIZE = [83, 83]
# =========================================parameters end ============================================


# ========================================= global FLAGS =============================================
flags = tf.app.flags
FLAGS = flags.FLAGS  # alias

# Random 种子
flags.DEFINE_integer('random_seed', 7, '')
np.random.seed(FLAGS.random_seed)

# Game parameter
flags.DEFINE_integer('agent_num', 9, 'my units number')
flags.DEFINE_integer('enemy_num', 4, 'enemy number')
flags.DEFINE_integer('action_dim', 10, 'one-hot action dimension')
flags.DEFINE_integer('state_dim', 11, 'one-hot state dimension')

# Replay buffer config
flags.DEFINE_string('replay_buffer_file_name', 'replay_buffer', 'replay buffer file name')
flags.DEFINE_integer('replay_buff_size', 16000, 'replay buffer size')  # 1M 10**6
flags.DEFINE_integer('replay_buff_save_segment_size', 8000, 'replay buffer save segment size')  # every 180,000 Transition data.  30*3000
flags.DEFINE_string('replay_buffer_file_path', "{}/replay_buffers".format(FLAGS.map), 'replay buffer file storage path')

# Training parameters
flags.DEFINE_integer('batch_size', 64, 'sample batch size')
flags.DEFINE_float('actor_learning_rate', 0.0005, 'actor optimizer learning rate')
flags.DEFINE_float('critic_learning_rate', 0.0005, 'critic optimizer learning rate')
flags.DEFINE_float('tau', 0.01, 'soft replacement rate')
flags.DEFINE_float('gamma', 0.95, 'discount rate of the previous reward')
flags.DEFINE_integer('lstm_layer', 1, 'LSTM layer number')
flags.DEFINE_integer('lstm_size', 32, 'LSTM hidden state size')
flags.DEFINE_integer('grad_clip', 5, 'gradient clip boundary')
flags.DEFINE_integer('keep_prob', 1, 'keep probability')
flags.DEFINE_float('entropy_regularizer_lambda', 0.05, 'percent of the entropy regularization loss (actor network)')

# Training log
flags.DEFINE_integer('cal_win_rate_every_episodes', 100, '')
flags.DEFINE_integer('log_every_steps', 100, '')
flags.DEFINE_integer('training_every_steps', 8, '')
flags.DEFINE_integer('print_softmax_every_steps', 1000, '')
flags.DEFINE_integer('verify_every_episodes', 1005, '')
# ========================================= global FLAGS end =========================================


from pysc2.agents.drnn.env_until import cal_local_observation_for_unit, convert_discrete_action_2_sc2_action, one_hot_action
from pysc2.agents.drnn.replay_buffer import construct_transition, ReplayBuffer
from pysc2.agents.drnn.actor import Actor
from pysc2.agents.drnn.ciritic import Critic

'''
    1: 确定通按照 tag 排序后（从小到大），tag 的顺序 对应 数组index的顺序 (能用)
    2: 确定按照转换后的位置选择每个单位是可用的

    python -m pysc2.bin.agent --agent pysc2.agents.coma2.coma.Coma --map DefeatRoaches
'''


class DRNN(base_agent.BaseAgent):

    def __init__(self):
        super(DRNN, self).__init__(FLAGS.enemy_num)
        # 训练模式
        self.is_training = True
        self.reload_from_model = False
        self.use_batch_norm = True

        self.sess = tf.Session()
        self.actor = Actor(self.sess, FLAGS.action_dim, FLAGS.state_dim, self.use_batch_norm)
        self.critic = Critic(self.sess, FLAGS.action_dim, FLAGS.state_dim, FLAGS.agent_num,
                             self.use_batch_norm)

        # saver
        self.saver = tf.train.Saver()

        # ============== analysis config ===============
        self.has_recorded_win = False
        # 最近 100 次胜利次数记录
        self.recent_100_episodes_win_cumulative = 0
        self.pre_100_episodes_win = 0
        # 模型的胜利次数
        self.model_win_episodes = 0
        self.model_total_episodes = 0
        # 每一轮 reward
        self.return_of_each_episode = 0
        self.pre_return_of_each_episode = 0
        # 累计胜利次数
        self.cumulative_win_times = 0

        # ===================================  可视化  ====================================
        # 记录 cumulative reward 曲线
        with tf.name_scope("cumulative_reward"):
            self.cumulative_reward_tensor = tf.placeholder(dtype=tf.float32, shape=[], name="reward")
            tf.summary.scalar('cumulative_reward', self.cumulative_reward_tensor)  # tensorflow >= 0.12

        with tf.name_scope("cumulative_win_times"):
            self.cumulative_win_times_tensor = tf.placeholder(dtype=tf.float32, shape=[], name="win_times")
            tf.summary.scalar('cumulative_win_times', self.cumulative_win_times_tensor)  # tensorflow >= 0.12

        with tf.name_scope("return_of_each_episode"):
            self.return_of_each_episode_tensor = tf.placeholder(dtype=tf.float32, shape=[],
                                                                name="return_of_each_episode")
            tf.summary.scalar('return_of_each_episode', self.return_of_each_episode_tensor)  # tensorflow >= 0.12
        # ================================== 可视化 END ====================================

        if self.is_training:
            # initialize replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size=FLAGS.replay_buff_size,
                                              save_segment_size=FLAGS.replay_buff_save_segment_size,
                                              save_path=FLAGS.replay_buffer_file_path,
                                              seed=FLAGS.random_seed
                                              )
            self.pre_save_time = time.time()
            self.model_id = 0

            # add summary
            self.merged = tf.summary.merge_all()  # tensorflow >= 0.12
            # writer
            self.writer = tf.summary.FileWriter("{}/logs/".format(FLAGS.map), self.sess.graph)  # tensorflow >=0.12

            # initialize the variables
            self.sess.run(tf.global_variables_initializer())

            # copy all the values of the online model to the target model
            self.actor.operation_update_TDnet_compeletely()
            self.critic.operation_update_TDnet_compeletely()
        elif (not self.is_training) or self.reload_from_model:
            self.saver.restore(self.sess, "{}/checkpoint_{}/model.ckpt".format(FLAGS.map, 0))

    def reset(self):
        super(DRNN, self).reset()
        # reset return
        self.pre_return_of_each_episode = self.return_of_each_episode
        self.return_of_each_episode = 0
        self.has_recorded_win = False

    def record_recent_win_rate(self):
        # 每隔 xxx 次，统计胜利比例
        if self.episodes % FLAGS.cal_win_rate_every_episodes == 0:
            # log to file
            content = "episodes {}, steps {}, recent 100 times' win rate {}/100".format(self.episodes,
                                                                                        self.steps,
                                                                                        self.recent_100_episodes_win_cumulative)
            self.append_log_to_file("{}/recent/win_rate.txt".format(FLAGS.map), content)
            self.recent_100_episodes_win_cumulative = 0

    def step(self, obs):
        super(DRNN, self).step(obs)

        # 1: update pre step hit points
        if self.is_first(obs):
            self.friends_pre_health = {tag: health for tag, health in self.get_raw_friends_data(obs)[:, [0, 6]]}
            self.enemies_pre_health = {tag: health for tag, health in self.get_raw_opponents_data(obs)[:, [0, 6]]}
        # 2: 不是开始，计算上一step 动作的 reward
        elif not self.has_recorded_win: # 保证胜利时，也只记录一次
            reward = self._calculated_reward(obs)
            # 统计每局的累计 Return
            self.return_of_each_episode += reward

        # 3: 胜利或者失败
        if self.is_finished(obs):
            # 如果是胜利（必须保证，只记录一次）
            if self.is_win(obs) and not self.has_recorded_win:
                # 统计总局数：
                self.model_total_episodes += 1
                # 统计模型赢了次数：
                self.model_win_episodes += 1
                self.recent_100_episodes_win_cumulative += 1
                self.cumulative_win_times += 1
                print("win: episode {}, total step {}, return: {}, cumulative win times: {}, recent {}/100".format(
                            self.episodes, self.steps, self.return_of_each_episode, self.cumulative_win_times,
                            self.recent_100_episodes_win_cumulative))

                if self.is_training:
                    # 存储 (s, a, r, s_, is_finished) (结局状态不重要)
                    self.save_transition(self.pre_all_alive_agents, reward, None)

                self.record_recent_win_rate()

                self.has_recorded_win = True # 保证只记录一次
            elif self.is_loss(obs) and not self.has_recorded_win: # 输了
                # 统计总局数：
                self.model_total_episodes += 1
                print("loss: episode {}, total step {}, return: {}, cumulative win times: {}, recent {}/100".format(
                    self.episodes, self.steps, self.return_of_each_episode, self.cumulative_win_times,
                    self.recent_100_episodes_win_cumulative))

                if self.is_training:
                    # 存储 (s, a, r, s_, is_finished) (结局状态不重要)
                    self.save_transition(self.pre_all_alive_agents, reward, None)

                self.record_recent_win_rate()
            # convert to sc2 actions
            actions_original, actions_queue = [], []
            actions_queue.append(actions.FunctionCall(_NO_OP, []))

        # 4: 开始或者游戏中
        else:
            # 对于己方的每个单位，计算局部观察信息，并执行动作
            actions_queue = []
            alive_friends = self.get_raw_friends_data(obs)
            alive_enemies = self.get_raw_opponents_data(obs)
            all_alive_agents = {}
            select_actions_prob = []
            for friend in alive_friends:
                agent_tuple = {}
                local_observation, sequence_len, alive_friends_order = cal_local_observation_for_unit(friend, alive_friends, alive_enemies, self.friends_tag_2_id)
                if self.is_training:
                    selected_action_id, _ = self.actor.operation_choose_action(1,
                                                                               [local_observation[0]],
                                                                               [local_observation[1]],
                                                                               [sequence_len[0]],
                                                                               [sequence_len[1]],
                                                                               is_training=False)
                    select_actions_prob.append(_)
                else:
                    selected_action_id = self.actor.operation_greedy_action(1,
                                                                            [local_observation[0]],
                                                                            [local_observation[1]],
                                                                            [sequence_len[0]],
                                                                            [sequence_len[1]],
                                                                            is_training=False)
                # 此处存储的相当于是 r, s', finished, actions of s'
                agent_tuple['state_friend'] = local_observation[0]
                agent_tuple['state_enemy'] = local_observation[1]
                agent_tuple['sequence_friend'] = sequence_len[0]
                agent_tuple['sequence_enemy'] = sequence_len[1]
                agent_tuple['terminated'] = False
                agent_tuple['action'] = one_hot_action(selected_action_id)
                agent_tuple['action_other_order'] = alive_friends_order[:-1] # 站在自己角度，其他存活单位 id(not tag) 顺序（不包括自己）
                assert len(alive_friends_order) > 0
                if alive_friends_order[-1] != self.friends_tag_2_id[friend[0]]:
                    print('我观察到的顺序',alive_friends_order)
                    print('friend 2 tag',self.friends_tag_2_id)
                    print('我的信息',friend)
                    print('所有存活单位',alive_friends)
                    print('我的观察',local_observation)
                    print('观察 sequence length',sequence_len)
                    return None
                all_alive_agents[self.friends_tag_2_id[friend[0]]] = agent_tuple

                # convert to sc2 actions
                action_sc2 = convert_discrete_action_2_sc2_action(friend, selected_action_id, alive_enemies,
                                                                  self.enemies_id_2_tag)
                actions_queue.extend(action_sc2)

            if self.steps % FLAGS.print_softmax_every_steps == 0:
                print("select_actions", select_actions_prob)
                self.append_log_to_file("{}/actions/select_actions.txt".format(FLAGS.map),
                                         "episodes {}, steps {}, select actions: {}".format(self.episodes,
                                                                                            self.steps,
                                                                                            select_actions_prob))

            if not self.is_first(obs) and self.is_training:
                self.save_transition(self.pre_all_alive_agents, reward, all_alive_agents)

            # update prev properties
            self.pre_all_alive_agents = all_alive_agents

            # TODO: batch training
            if self.is_training and self.replay_buffer.length >= FLAGS.batch_size * 2 and self.steps % FLAGS.training_every_steps == 0:
                (fr_states, em_state, fr_seq_len, em_seq_len, ac_others,
                ac, reward,
                nxt_fr_states, nxt_em_states, nxt_fr_sequence_len, nxt_em_sequence_len,
                nxt_oth_fr_states, nxt_oth_em_states, nxt_oth_fr_seq_len, nxt_oth_em_seq_len,
                terminated_batch) = self.replay_buffer.sample_batch(FLAGS.batch_size)
                # training critic:
                # 1: 准备 batch 数据
                action_others_batch_s_ = []
                # TODO: 可能会只有一个（当前单位已经死亡）
                for nxt_fr_s, nxt_fr_seq, nxt_em_s, nxt_em_seq in zip(nxt_oth_fr_states, nxt_oth_fr_seq_len,
                                                                      nxt_oth_em_states, nxt_oth_em_seq_len):  # 对于每一个单位：其他所有单位的观察

                    if nxt_fr_s is None:
                        action_others_s_ = None
                    else:
                        # 还没有 one-hot
                        action_others_s_ = self.actor.operation_greedy_action(len(nxt_fr_s), nxt_fr_s, nxt_em_s, nxt_fr_seq,
                                                                              nxt_em_seq, is_training=False)
                    action_per = []
                    if action_others_s_ is not None:
                        for action_id in action_others_s_:
                            one_hot_a = one_hot_action(action_id)
                            action_per.append(one_hot_a)
                    action_others_batch_s_.append(self._flatten_others_actions(action_per))

                # 2: cal td target
                batch_td_target = self.critic.operation_get_TDtarget(
                    len(nxt_fr_states),
                    nxt_fr_states,
                    nxt_em_states,
                    nxt_fr_sequence_len,
                    nxt_em_sequence_len,
                    action_others_batch_s_,# 已经对齐了，11 * 8
                    reward,
                    terminated_batch,
                    is_training=True
                )
                # 3: training critic
                self.critic.operation_critic_learn(len(fr_states),
                                                   fr_states,
                                                   em_state,
                                                   fr_seq_len,
                                                   em_seq_len,
                                                   ac_others,
                                                   ac,batch_td_target,
                                                   is_training=True)

                # training actor
                # 3: calculate advantage
                actor_output_probability = self.actor.operation_cal_softmax_probablility(len(fr_states),
                                                                                         fr_states,
                                                                                         em_state,
                                                                                         fr_seq_len,
                                                                                         em_seq_len,
                                                                                         is_training=True)
                batch_advantages = self.critic.operation_cal_advantage(len(fr_states),
                                                                       fr_states,
                                                                       em_state,
                                                                       fr_seq_len,
                                                                       em_seq_len,
                                                                       ac_others,
                                                                       ac,
                                                                       actor_output_probability,
                                                                       is_training=True)
                # update actor
                cost = self.actor.operation_actor_learn(len(fr_states),
                                                        fr_states,
                                                        em_state,
                                                        fr_seq_len,
                                                        em_seq_len,
                                                        ac,
                                                        batch_advantages, is_training=True)
                # self.new_state = final_state

                # ===================================  可视化  ====================================
                # add summary
                if self.steps % FLAGS.log_every_steps == 0:
                    feed_dict = {
                        self.actor.state_inputs_friends: fr_states,
                        self.actor.state_inputs_enemies: em_state,
                        self.actor.sequence_length_friends: fr_seq_len,
                        self.actor.sequence_length_enemies: em_seq_len,
                        self.actor.execute_action: ac,
                        self.actor.advantage: batch_advantages,
                        self.actor.is_training: True,
                        self.actor.keep_prob: 1.,
                        self.actor.batch_size: len(fr_states),

                        self.critic.state_input_friends: fr_states,
                        self.critic.state_input_enemies: em_state,
                        self.critic.sequence_length_friends: fr_seq_len,
                        self.critic.sequence_length_enemies: em_seq_len,
                        self.critic.other_units_action_input: ac_others,
                        self.critic.self_action_input: ac,
                        self.critic.Q_value_label_input: batch_td_target,
                        self.critic.is_training: True,
                        self.critic.keep_prob: 1.,
                        self.critic.batch_size: len(fr_states),

                        self.cumulative_reward_tensor: self.reward,
                        self.cumulative_win_times_tensor: self.cumulative_win_times,
                        self.return_of_each_episode_tensor: self.pre_return_of_each_episode
                    }
                    rs = self.sess.run(self.merged, feed_dict=feed_dict)
                    self.writer.add_summary(rs, self.steps)
                # ================================== 可视化 END ====================================

                # soft update the parameters of the two model
                # print("soft update parameters: episode {}, step {}, reward： {}".format(self.episodes, self.steps, reward))
                self.actor.operation_soft_update_TDnet()
                self.critic.operation_soft_update_TDnet()

        # TODO 每隔半个小时保存一次模型
        # if self.is_training and (time.time() - self.pre_save_time) > 1800:
        if self.is_training and self.model_total_episodes == FLAGS.verify_every_episodes and (time.time() - self.pre_save_time) > 30: # (防重复)
            content = "model {}: episodes {}, steps {}, win rate {}/{}".format(self.model_id, self.episodes, self.steps,
                                                                               self.model_win_episodes,
                                                                               self.model_total_episodes)
            self.append_log_to_file("{}/model/model.txt".format(FLAGS.map), content)

            self.model_win_episodes = 0
            self.model_total_episodes = 0

            self.saver.save(self.sess, "{}/checkpoint_{}/model.ckpt".format(FLAGS.map, self.model_id))
            self.pre_save_time = time.time()
            self.model_id += 1


        # base update
        self.update_of_each_step(obs)

        if self.episodes == FLAGS.verify_every_episodes * 10:
            return None

        return actions_queue

    def _flatten_others_actions(self, others_actions):
        while len(others_actions) < (FLAGS.agent_num - 1):
            others_actions.append(np.zeros([FLAGS.action_dim], dtype=np.float32))
        temp = np.array(others_actions, dtype=np.float32)
        return temp.ravel()

    def save_transition(self,
                        pre_all_alive_agents,
                        reward,
                        all_alive_agents):
        '''
        :return:
        '''
        for unit_id, agent in pre_all_alive_agents.items():
            # state
            s = [pre_all_alive_agents[unit_id]['state_friend'], pre_all_alive_agents[unit_id]['state_enemy']]
            sequence_length = [pre_all_alive_agents[unit_id]['sequence_friend'], pre_all_alive_agents[unit_id]['sequence_enemy']]
            a_others = [pre_all_alive_agents[id]['action'] for id in reversed(pre_all_alive_agents[unit_id]['action_other_order'])]
            # 其他 agent actions
            a_others = self._flatten_others_actions(a_others)
            # 自己的 action
            a = pre_all_alive_agents[unit_id]['action']
            # total reward
            r = reward
            # 自己的下一个状态 s'
            terminated = all_alive_agents is None or all_alive_agents.get(unit_id, None) is None
            only_self_alive = False
            if not terminated:
                # s_
                s_ = [all_alive_agents[unit_id]['state_friend'],
                      all_alive_agents[unit_id]['state_enemy']]

                # sequence length_
                sequence_length_ = [all_alive_agents[unit_id]['sequence_friend'],
                                   all_alive_agents[unit_id]['sequence_enemy']]

                # 还有其他队友单位存活
                if len(all_alive_agents[unit_id]['action_other_order']) > 0:
                    others_s_ = [[all_alive_agents[id]['state_friend'], all_alive_agents[id]['state_enemy']] for id in reversed(all_alive_agents[unit_id]['action_other_order'])]
                    others_sequence_length_ = [[all_alive_agents[id]['sequence_friend'], all_alive_agents[id]['sequence_enemy']] for id in reversed(all_alive_agents[unit_id]['action_other_order'])]
                else:
                    others_s_ = None
                    others_sequence_length_ = [[0, 0]]
                    only_self_alive = True

            else:  # 当下单位已经死亡（下一个状态是什么没有影响）
                # s_
                s_ = [pre_all_alive_agents[unit_id]['state_friend'],
                      pre_all_alive_agents[unit_id]['state_enemy']]
                sequence_length_ = [0, 0]
                others_s_ = None
                others_sequence_length_ = [[0, 0]]
            transition = construct_transition(s, sequence_length, a_others, a, r, s_, sequence_length_, others_s_,
                                              others_sequence_length_, terminated, only_self_alive)
            # print(transition)
            self.replay_buffer.store(transition)

    def _calculated_reward(self, obs):
        '''
        计算己方每个单位的 reward
        :param obs: 环境观察
        :return: reward
        '''
        # current tag and health
        friends_current_health = {tag: health for tag, health in self.get_raw_friends_data(obs)[:, [0, 6]]}
        enemies_current_health = {tag: health for tag, health in self.get_raw_opponents_data(obs)[:, [0, 6]]}
        friends_sum_delta_health, enemies_sum_delta_health, friends_sum_health = 0., 0., 0.
        for tag in self.all_friends_tag:
            cur = friends_current_health.get(tag, -1)
            # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
            if cur == -1:
                friends_current_health[tag] = 0
                friends_sum_delta_health += (self.friends_pre_health[tag] - 0)
            else:
                friends_sum_delta_health += (self.friends_pre_health[tag] - cur)
                friends_sum_health += cur

        for tag in self.all_enemies_tag:
            cur = enemies_current_health.get(tag, -1)
            # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
            if enemies_current_health.get(tag, -1) == -1:
                enemies_current_health[tag] = 0
                enemies_sum_delta_health += (self.enemies_pre_health[tag] - 0)
            else:
                enemies_sum_delta_health += (self.enemies_pre_health[tag] - cur)

        # update prev
        self.friends_pre_health = friends_current_health
        self.enemies_pre_health = enemies_current_health

        # 计算上一轮动作执行所产生的 reward
        reward_part1 = enemies_sum_delta_health - friends_sum_delta_health * 0.5
        # 杀掉一个 +10
        reward_part2 = self.pre_action_kill_enemies_num(obs) * 10
        # 赢一局 +200 + alive_friends_sum_health
        reward_part3 = 0
        if self.is_win(obs):
            reward_part3 = 200 + friends_sum_health
        reward = reward_part1 + reward_part2 + reward_part3
        return reward