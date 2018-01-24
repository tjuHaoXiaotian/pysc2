#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np
import time
import random
import tensorflow as tf
from pysc2.agents.common.env_until import cal_local_observation_for_unit,convert_discrete_action_2_sc2_action,one_hot_action
from pysc2.agents.coma.replay_buffer import construct_transition, ReplayBuffer
from pysc2.agents.coma.actor import Actor
from pysc2.agents.coma.ciritic import Critic
import os
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
_SELECT = [0] # Select,Toggle,SelectAllOfType,AddAllType
_SCREEN_SIZE = [83, 83]
# =========================================parameters end ============================================


COMA_CFG = tf.app.flags.FLAGS  # alias
COMA_CFG.map = "DefeatRoaches"

COMA_CFG.replay_buff_size = 1
COMA_CFG.batch_size = 16
COMA_CFG.replay_buff_size = 10**6  # 1M
COMA_CFG.replay_buff_save_segment_size = 30*3000  # every 180,000 Transition data.
# COMA_CFG.replay_buff_save_segment_size = 30  # every 180,000 Transition data.
COMA_CFG.replay_buffer_file_path = "{}/replay_buffers".format(COMA_CFG.map)
COMA_CFG.random_seed = 1


COMA_CFG.agent_num = 9
COMA_CFG.enemy_num = 4
COMA_CFG.action_dim = 10 # one-hot 以后
COMA_CFG.state_dim = 11
COMA_CFG.tau = 0.01 # soft replacement
COMA_CFG.learning_rate = 1e-4
COMA_CFG.q_learning_rate = 1e-4
COMA_CFG.GAMMA = 0.95
COMA_CFG.ENTROPY_REGULARIZER_LAMBDA = 0.1

COMA_CFG.CAL_WIN_RATE_EVERY_EPISODES = 100


'''
    1: 确定通按照 tag 排序后（从小到大），tag 的顺序 对应 数组index的顺序 (能用)
    2: 确定按照转换后的位置选择每个单位是可用的
'''
class Coma(base_agent.BaseAgent):
    print_infos = False

    def __init__(self):
        super(Coma, self).__init__()

        # 训练模式
        self.is_training = True
        self.use_batch_norm = True

        # previous step alived roaches
        self.pre_alived_roaches = 4
        self.win_times = 0
        self.cumulative_win_times = 0
        self.cumulative_reward = 0

        self.sess = tf.Session()
        self.actor = Actor(self.sess, COMA_CFG.action_dim, COMA_CFG.state_dim, self.use_batch_norm)
        self.critic = Critic(self.sess, COMA_CFG.action_dim, COMA_CFG.state_dim, COMA_CFG.agent_num,
                             self.use_batch_norm)

        # saver 存储 variable parameter
        self.saver = tf.train.Saver()

        # 最近 100 次胜利次数记录
        self.recent_100_episodes_win_cumulative = 0
        self.pre_100_episodes_win = 0

        # 记录 cumulative reward 曲线
        with tf.name_scope("cumulative_reward"):
            self.cumulative_reward_tensor = tf.placeholder(dtype=tf.float32, shape=None, name="reward")
            tf.summary.scalar('cumulative_reward', self.cumulative_reward_tensor)  # tensorflow >= 0.12

        if self.is_training:
            # 实例化 replay buffer，指定是否将buffer数据保存到文件
            self.replay_buffer = ReplayBuffer(buffer_size=COMA_CFG.replay_buff_size,
                                              save_segment_size=COMA_CFG.replay_buff_save_segment_size,
                                              save_path=COMA_CFG.replay_buffer_file_path,
                                              seed=COMA_CFG.random_seed
                                              )
            self.pre_save_time = time.time()
            self.model_id = 0
            self.model_win_episodes = 0
            self.model_total_episodes = 0

            # add summary
            # merged= tf.merge_all_summaries()    # tensorflow < 0.12
            self.merged = tf.summary.merge_all()  # tensorflow >= 0.12

            # writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)  # tensorflow >=0.12

            # initalize the variables
            self.sess.run(tf.global_variables_initializer())
            # copy all the values of the online model to the target model
            self.actor.operation_update_TDnet_compeletely()
            self.critic.operation_update_TDnet_compeletely()
        else:
            self.saver.restore(self.sess, "checkpoint_1/model.ckpt")

    def reset(self):
        super(Coma, self).reset()
        self.pre_alived_roaches = 4
        self.win_times = 0
        self.cumulative_reward = 0

    def step(self, obs):
        super(Coma, self).step(obs)
        # time.sleep(5)  # 休眠 1s 方便观察

        # 1： 每局开始记录所有存活单位的信息
        if self.isFirst(obs):
            self.all_friends_tag = self._get_raw_friends_data(obs)[:,0]
            self.all_enemies_tag = self._get_raw_opponents_data(obs)[:,0]
            self.friends_tag_2_id = {tag: id for id, tag in enumerate(self.all_friends_tag)}
            self.friends_id_2_tag = dict(enumerate(self.all_friends_tag))
            self.enemies_tag_2_id = {tag: id for id, tag in enumerate(self.all_enemies_tag)}
            self.enemies_id_2_tag = dict(enumerate(self.all_enemies_tag))

        # 2: 计算上一轮的 reward （当然，第一轮没有用）
        reward = self._calculated_reward(obs)
        self.cumulative_reward += reward

        # 3: 如果还没有结束(这里是胜利)
        if not self.is_roach_defeated(obs):
            if self.isLast(obs): # 我输了
                # 统计模型赢了次数：
                self.model_total_episodes += 1
                if self.episodes % COMA_CFG.CAL_WIN_RATE_EVERY_EPISODES == 0:
                    self.pre_100_episodes_win = self.recent_100_episodes_win_cumulative
                    self.recent_100_episodes_win_cumulative = 0
                    # log to file
                    content = "episodes {}, steps {}, recent 100 times' win rate {}/100".format(self.episodes,self.steps,self.pre_100_episodes_win)
                    self._append_log_to_file("{}/recent/win_rate.txt".format(COMA_CFG.map), content)

                actions_original,actions_queue = [], []
                actions_queue.append(actions.FunctionCall(_NO_OP, []))
                # 存储 (结局状态不重要)
                self.save_transition(self.pre_alive_order, self.pre_observation, self.pre_actions, reward, self.pre_alive_order, self.pre_observation, True)
                print("loss: episode {}, total step {}, reward: {}, cumulative win times: {}, recent {}/100".format(self.episodes, self.steps, self.cumulative_reward, self.cumulative_win_times, self.pre_100_episodes_win))

            else:  # 游戏中
                # 4: 对于己方的每个单位，计算局部观察信息，并执行动作
                actions_queue = []
                alive_friends = self._get_raw_friends_data(obs)
                alive_enemies = self._get_raw_opponents_data(obs)
                states_to_save = {}
                actions_to_save = {}
                alive_order_to_save = {}
                # select_actions = []
                for friend in alive_friends:
                    local_observation, alive_friends_order = cal_local_observation_for_unit(friend, alive_friends, alive_enemies, self.friends_tag_2_id)
                    selected_action_id,_ = self.actor.operation_choose_action([local_observation], is_training=False)
                    # select_actions.append(selected_action_id)
                    # select_actions.append(_)
                    action_sc2 = convert_discrete_action_2_sc2_action(friend, selected_action_id, alive_enemies, self.enemies_id_2_tag)
                    actions_queue.extend(action_sc2)
                    # 保存 state 和 action
                    states_to_save[self.friends_tag_2_id[friend[0]]] = local_observation
                    actions_to_save[self.friends_tag_2_id[friend[0]]] = one_hot_action(selected_action_id)
                    # 每个单位角度，存活单位顺序（包括自己）
                    alive_order_to_save[self.friends_tag_2_id[friend[0]]] = alive_friends_order
                # print(select_actions)
                # 5：如果不是开始和结束，则存储 (s,a,r,s_,done,model alived ids)
                if not self.isFirst(obs) and self.is_training:
                    # 存储
                    self.save_transition(self.pre_alive_order, self.pre_observation, self.pre_actions, reward, alive_order_to_save, states_to_save, False)

                # 6: update prev properties
                self.pre_observation = states_to_save
                self.pre_actions = actions_to_save
                self.pre_alive_order = alive_order_to_save

        else:
            # TODO: 星际2 mini game 每局打赢后还会额外增5个新兵继续
            # If goes here, current episode is ended and our team win
            # 执行动作 no_op
            actions_queue = []
            actions_queue.append(actions.FunctionCall(_NO_OP, []))

            if self.win_times == 2 and self.is_training:  # 我赢了, 只记录一次
                # 统计模型赢了次数：
                self.model_total_episodes += 1
                self.model_win_episodes += 1
                self.recent_100_episodes_win_cumulative += 1
                self.cumulative_win_times += 1

                if self.episodes % COMA_CFG.CAL_WIN_RATE_EVERY_EPISODES == 0:
                    self.pre_100_episodes_win = self.recent_100_episodes_win_cumulative
                    self.recent_100_episodes_win_cumulative = 0
                    # log to file
                    content = "episodes {}, steps {}, recent 100 times' win rate {}/100".format(self.episodes,self.steps, self.pre_100_episodes_win)
                    self._append_log_to_file("{}/recent/win_rate.txt".format(COMA_CFG.map), content)

                # 存储 (结局状态不重要)
                self.save_transition(self.pre_alive_order, self.pre_observation, self.pre_actions, reward, self.pre_alive_order, self.pre_observation, True)
                print("win: episode {}, total step {}, reward: {}, cumulative win times: {}, recent {}/100".format(self.episodes, self.steps, self.cumulative_reward, self.cumulative_win_times, self.pre_100_episodes_win))


        # TODO: batch training  and self.episodes % 2 == 0
        if self.replay_buffer.length >= COMA_CFG.batch_size * 2 and self.is_training and self.win_times <= 2 and self.steps % 2 == 0:
            state_batch, action_others_batch, action_batch, reward_batch, next_state_batch, next_state_others_batch, terminated_batch = self.replay_buffer.sample_batch(COMA_CFG.batch_size)
            # training critic:
            # 1: 准备 batch 数据
            action_others_batch_s_ = []
            for next_state_others in next_state_others_batch:
                # 还没有 one-hot
                action_others_s_ = self.actor.operation_greedy_action(next_state_others, is_training=True)
                action_per = []
                for action_id in action_others_s_:
                    one_hot_a = one_hot_action(action_id)
                    action_per.append(one_hot_a)
                action_others_batch_s_.append(self._flatten_others_actions(action_per))

            # 2: cal td target
            batch_td_target = self.critic.operation_get_TDtarget(
                next_state_batch,
                action_others_batch_s_,
                reward_batch,
                terminated_batch,
                is_training=True
            )
            self.critic.operation_critic_learn(state_batch, action_others_batch, action_batch, batch_td_target, is_training=True)

            # training actor
            # 3: calculate advantage
            actor_output_probability = self.actor.operation_cal_softmax_probablility(state_batch, is_training=True)
            batch_advantages = self.critic.operation_cal_advantage(state_batch, action_others_batch, action_batch, actor_output_probability, is_training=True)
            # update actor
            self.actor.operation_actor_learn(state_batch, action_batch, batch_advantages, is_training=True)

            # add summary
            if self.steps % 50 == 0:
                rs = self.sess.run(self.merged, feed_dict={
                    self.actor.state_input:state_batch,
                    self.actor.execute_action:action_batch,
                    self.actor.advantage:batch_advantages,
                    self.actor.is_training: True,

                    self.critic.critic_state_input: state_batch,
                    self.critic.other_units_action_input: action_others_batch,
                    self.critic.self_action_input: action_batch,
                    self.critic.Q_value_label_input: batch_td_target,
                    self.critic.is_training: True,

                    self.cumulative_reward_tensor: self.reward
                })
                self.writer.add_summary(rs, self.steps)

            # soft update the parameters of the two model
            # print("soft update parameters: episode {}, step {}, reward： {}".format(self.episodes, self.steps, reward))
            self.actor.operation_soft_update_TDnet()
            self.critic.operation_soft_update_TDnet()

        # 没一个小时保存一次模型
        if self.is_training and (time.time() - self.pre_save_time) > 1800:
            content = "model {}: episodes {}, steps {}, win rate {}/{}".format(self.model_id, self.episodes, self.steps, self.model_win_episodes, self.model_total_episodes)
            self._append_log_to_file("{}/model/model.txt".format(COMA_CFG.map), content)

            self.model_win_episodes = 0
            self.model_total_episodes = 0

            self.saver.save(self.sess, "{}/checkpoint_{}/model.ckpt".format(COMA_CFG.map,self.model_id))
            self.pre_save_time = time.time()
            self.model_id += 1

        return actions_queue

    def _append_log_to_file(self, file_name, content):
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        f = open(file_name, "a")  # 打开文件以便写入
        # np.set_printoptions(threshold=np.inf)  # 全部输出
        # f.write(str(self._obs))
        print(content, file=f)
        f.flush()
        f.close()

    def _flatten_others_actions(self, others_actions):
        while len(others_actions) < (COMA_CFG.agent_num - 1):
            others_actions.append(np.zeros([COMA_CFG.action_dim], dtype=np.float32))
        temp = np.array(others_actions, dtype=np.float32)
        return temp.ravel()

    def save_transition(self, all_alive_order_s, all_alive_s, all_alive_a, all_r, all_alive_order_s_, all_alive_s_, all_done):
        for unit_id, local_state in all_alive_s.items():
            s = local_state
            alive_order_s = all_alive_order_s.get(unit_id)
            a_others = [all_alive_a[id] for id in alive_order_s if id != unit_id]
            a_others = self._flatten_others_actions(a_others)
            a = all_alive_a.get(unit_id)
            r = all_r
            s_ = all_alive_s_.get(unit_id, None)
            alive_order_s_ = all_alive_order_s_.get(unit_id, None)
            done = True if all_done else s_ is None
            if alive_order_s_ is not None and len(alive_order_s_) > 1: # 还有其他单位存活
                s__others = [all_alive_s_[id] for id in alive_order_s_ if id != unit_id]  # 为了求其他单位在 s' 下的动作，需要存储其他单位在 s'的局部观察
            else: # 当下单位已经死亡
                s__others = [np.zeros([11 * (COMA_CFG.agent_num+COMA_CFG.enemy_num)], dtype=np.float32)]  # Done = True, 状态不重要，但是不能为空，所以设置为上一个状态


            if s_ is None: # 当下单位已经死亡
                s_ = s # Done = True, 状态不重要，但是不能为空，所以设置为上一个状态

            transition = construct_transition(s, a_others, a, r, s_, s__others, done)
            # print(transition)
            self.replay_buffer.store(transition)

    def isFirst(self, obs):
        return obs.first()

    def isMid(self, obs):
        return obs.mid()

    def isLast(self, obs):
        return obs.last()

    def _get_raw_friends_data(self, obs):
        '''
        获取己方单位原始信息
        :param obs:
        :return:
        '''
        return obs.observation['my_units']

    def _get_raw_opponents_data(self, obs):
        '''
        获取敌方单位原始信息
        :param obs:
        :return:
        '''
        return obs.observation['my_opponents']

    def _calculated_reward(self, obs):
        '''
        计算己方每个单位的 reward
        :param obs: 环境观察
        :return: training_needed_model_ids, reward
        '''
        # 为后续 存储 prev_alives
        friends = self._get_raw_friends_data(obs)
        enemies = self._get_raw_opponents_data(obs)
        # tag and health
        friends_current_health = friends[:, [0,6]]
        enemies_current_health = enemies[:, [0,6]]

        if self.isFirst(obs):
            # first time： reward is 0
            # 存储 pre step hit_points
            self.friends_pre_health = {tag: health for tag, health in friends_current_health}
            self.enemies_pre_health = {tag: health for tag, health in enemies_current_health}
            return 0
        else:
            self.friends_current_health = {tag: health for tag, health in friends_current_health}
            self.enemies_current_health = {tag: health for tag, health in enemies_current_health}
            friends_sum_delta_health, enemies_sum_delta_health, friends_sum_health = 0., 0., 0.
            for tag in self.all_friends_tag:
                cur = self.friends_current_health.get(tag, -1)
                # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
                if cur == -1:
                    self.friends_current_health[tag] = 0
                    friends_sum_delta_health += (self.friends_pre_health[tag] - 0)
                else:
                    friends_sum_delta_health += (self.friends_pre_health[tag] - cur)
                    friends_sum_health += cur

            for tag in self.all_enemies_tag:
                cur = self.enemies_current_health.get(tag, -1)
                # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
                if self.enemies_current_health.get(tag,-1) == -1:
                    self.enemies_current_health[tag] = 0
                    enemies_sum_delta_health += (self.enemies_pre_health[tag] - 0)
                else:
                    enemies_sum_delta_health += (self.enemies_pre_health[tag] - cur)

            # 计算上一轮动作执行所产生的 reward
            # Note: 此处的reward 是返回给上一轮执行的动作（上一轮动作执行的后果）
            # avg_friends_delta_health = friends_sum_delta_health / COMA_CFG.agent_num
            # avg_friends_delta_health = friends_sum_delta_health / len(self.prev_alives)
            # avg_enemies_delta_health = enemies_sum_delta_health / COMA_CFG.enemy_num
            # avg_enemies_delta_health = enemies_sum_delta_health / len(self.prev_enemies_alives)
            # reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*COMA_CFG.agent_num, dtype=np.float32)
            # reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*len(self.prev_alives), dtype=np.float32)
            reward_part1 = enemies_sum_delta_health * 10 - friends_sum_delta_health
            # 杀掉一个 +10
            reward_part2 = self._kill_enemies(obs) * 100
            reward_part3 = 0
            if self.is_roach_defeated(obs):
                reward_part3 = 1000 + friends_sum_health
            reward = reward_part1 + reward_part2 + reward_part3

            self.friends_pre_health = self.friends_current_health
            self.enemies_pre_health = self.enemies_current_health
            return reward

    def _kill_enemies(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            # 全部杀光光
            current_enemies = 0
        else:
            current_enemies = len(self._get_raw_opponents_data(obs))
        return self.pre_alived_roaches - current_enemies

    def is_roach_defeated(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            self.pre_alived_roaches = 0
            self.win_times += 1
            return True
        else:
            self.pre_alived_roaches = len(self._get_raw_opponents_data(obs))
            return False
