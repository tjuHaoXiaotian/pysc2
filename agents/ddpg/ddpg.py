#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import time
import random
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import tensorflow as tf
from pysc2.agents.ddpg.replay_buffer import *
from pysc2.agents.ddpg.actor import Actor
from pysc2.agents.ddpg.ciritic import Critic
from sklearn.preprocessing import OneHotEncoder

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
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_ATTACK_SCREEN  = actions.FUNCTIONS.Attack_Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# =========================================actions end ===============================================

# ==========================================parameters ===============================================
_NOT_QUEUED = [0]
_SELECT_ALL = [0] # (select vs select_add)
# =========================================parameters end ============================================

DDPG_CFG = tf.app.flags.FLAGS  # alias

DDPG_CFG.replay_buff_size = 1
DDPG_CFG.batch_size = 8
DDPG_CFG.replay_buff_size = 10**6  # 1M
DDPG_CFG.replay_buff_save_segment_size = 30*3000  # every 180,000 Transition data.
# DDPG_CFG.replay_buff_save_segment_size = 30  # every 180,000 Transition data.
DDPG_CFG.replay_buffer_file_path = "replay_buffers"
DDPG_CFG.random_seed = 1

DDPG_CFG.agent_num = 9
DDPG_CFG.enemy_num = 4
DDPG_CFG.action_dim = 3
DDPG_CFG.state_dim = 8
DDPG_CFG.action_bound = np.array([1, 84, 84], dtype=np.float32)
DDPG_CFG.action_sigmoid_vars = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DDPG_CFG.action_sigmoid_vars_min = np.array([0.05, 0.01, 0.01], dtype=np.float32)
DDPG_CFG.action_vars_delta = (DDPG_CFG.action_sigmoid_vars - DDPG_CFG.action_sigmoid_vars_min) / (3600 * 6)

DDPG_CFG.tau = 0.01 # soft replacement
DDPG_CFG.learning_rate = 1e-4
DDPG_CFG.q_learning_rate = 1e-3
DDPG_CFG.GAMMA = 0.95

class ControllUnits(base_agent.BaseAgent):

    print_infos = False

    def __init__(self):
        super(ControllUnits,self).__init__()
        # 实例化 replay buffer，指定是否将buffer数据保存到文件
        self.replay_buffer = ReplayBuffer(buffer_size=DDPG_CFG.replay_buff_size,
                               save_segment_size= DDPG_CFG.replay_buff_save_segment_size,
                               save_path=DDPG_CFG.replay_buffer_file_path,
                               seed=DDPG_CFG.random_seed
                               )
        
        # 训练模式
        self.is_training = True
        self.use_batch_norm = True

        # previous step alived roaches
        self.pre_alived_roaches = 4
        self.win_times = 0
        self.cumulative_win_times = 0
        self.cumulative_reward = 0

        self.sess = tf.Session()
        self.actor = Actor(self.sess, DDPG_CFG.action_dim, DDPG_CFG.state_dim, self.use_batch_norm)
        self.critic = Critic(self.sess, DDPG_CFG.action_dim, DDPG_CFG.state_dim, DDPG_CFG.agent_num, self.use_batch_norm)
        # saver 存储 variable parameter
        self.saver = tf.train.Saver()

        if self.is_training:
            self.pre_save_time = time.time()
            # for 更新探索率
            self.pre_update_exploration_time = time.time()
            self.model_id = 0
            # initalize the variables
            self.sess.run(tf.global_variables_initializer())
            # copy all the values of the online model to the target model
            self.actor.operation_update_TDnet_compeletely()
            self.critic.operation_update_TDnet_compeletely()
        else:
            self.saver.restore(self.sess, "checkpoint_1/model.ckpt")

    def update_exploration(self, elapse_time):
        if np.min(DDPG_CFG.action_sigmoid_vars) > 0.01:
            DDPG_CFG.action_sigmoid_vars -= DDPG_CFG.action_vars_delta * elapse_time

    def reset(self):
        super(ControllUnits, self).reset()
        # reset previous step alived roaches
        self.pre_alived_roaches = 4
        self.win_times = 0
        self.cumulative_reward = 0
        # for 更新探索率
        self.pre_update_exploration_time = time.time()

    # TODO: try to understand the random algorithm
    def _add_noise_and_reshape(self, action, var, lower_bound, upper_bound):
        '''
        :param action: 均值
        :param var: 方差
        :return:
        '''
        return np.clip(np.random.normal(loc=action, scale=var), lower_bound, upper_bound)
            # .reshape(DDPG_CFG.action_dim)

    def build_dead(self, tag, unit_type, owner):
        return np.array((tag, unit_type, owner, 0, 0, 0, 0),dtype=np.int64)

    def _input_feature_pre_processing(self,obs):
        '''
        预处理图片输入，转换为所需 feature
        :param obs: 环境 observation
        :return: [one_hot_location[self], one_hot_location[enemy], hit_points]
        '''
        # numpy style
        temp_state = []
        alive_friends = self._get_raw_friends_data(obs)
        alive_enemies = self._get_raw_opponents_data(obs)
        alive_friends_origin_ids = []
        alive_enemies_origin_ids = []
        alive_friends_locations = []
        # 按照顺序重新排列
        prev_id = -1
        for alive_idx in range(len(alive_friends)):
            alive_tag = alive_friends[alive_idx][0]
            alive_origin_id = self.friends_tag_2_id[alive_tag]
            alive_friends_origin_ids.append(alive_origin_id)
            alive_friends_locations.append([alive_friends[alive_idx][3], alive_friends[alive_idx][4]])
            if prev_id == alive_origin_id - 1: # 上一个已经存入
                temp_state.append(alive_friends[alive_idx])
            else:
                for loss_id in range(prev_id + 1, alive_origin_id):
                    temp_state.append(self.build_dead(self.all_friends_tag[loss_id], 48, 1))
                temp_state.append(alive_friends[alive_idx])
            prev_id = alive_origin_id

        for loss_id in range(prev_id + 1, len(self.all_friends_tag)):
            temp_state.append(self.build_dead(self.all_friends_tag[loss_id], 48, 1))

        prev_id = -1
        for alive_idx in range(len(alive_enemies)):
            alive_tag = alive_enemies[alive_idx][0]
            alive_origin_id = self.enemies_tag_2_id[alive_tag]
            alive_enemies_origin_ids.append(alive_origin_id)
            if prev_id == alive_origin_id - 1: # 上一个已经存入
                temp_state.append(alive_enemies[alive_idx])
            else:
                for loss_id in range(prev_id + 1, alive_origin_id):
                    temp_state.append(self.build_dead(self.all_enemies_tag[loss_id], 110, 2))
                temp_state.append(alive_enemies[alive_idx])
            prev_id = alive_origin_id
        for loss_id in range(prev_id + 1, len(self.all_enemies_tag)):
            temp_state.append(self.build_dead(self.all_enemies_tag[loss_id], 110, 2))
        temp_state = np.array(temp_state, dtype=np.int64)


        # one-hot encoding
        enc = OneHotEncoder()
        enc.fit(temp_state[:,[1,2]])
        part1 = enc.transform(temp_state[:,[1,2]]).toarray()
        part2_1_division = [83, 83, 360, 45]
        part2_2_division = [83, 83, 360, 145]
        # normalizing
        part2_1 = temp_state[:9,3:] / part2_1_division
        part2_2 = temp_state[9:,3:] / part2_2_division
        part2 = np.vstack([part2_1,part2_2])
        features = np.hstack([part1,part2])
        return np.hstack(features), [alive_friends_origin_ids,alive_enemies_origin_ids], alive_friends_locations

    def _kill_enemies(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            # 全部杀光光
            current_enemies = 0
        else:
            current_enemies = len(self._get_raw_opponents_data(obs))
        return self.pre_alived_roaches - current_enemies

    def _calculated_reward(self, obs):
        '''
        计算己方每个单位的 reward
        :param obs: 环境观察
        :return: training_needed_model_ids, reward
        '''
        friends = self._get_raw_friends_data(obs)
        # 为后续 存储 prev_alives
        # self.current_alives = friends[:, 0]
        enemies = self._get_raw_opponents_data(obs)
        # self.current_enemies_alives = enemies[:, 0]
        # tag and health
        friends_current_health = friends[:, [0,6]]  
        enemies_current_health = enemies[:, [0,6]]

        if self.isFirst(obs):
            # first time： reward is 0
            # 存储 pre step hit_points
            self.friends_pre_health = {tag: health for tag, health in friends_current_health}
            self.enemies_pre_health = {tag: health for tag, health in enemies_current_health}
            # 存储 pre step alive units
            # self.prev_alives = self.current_alives
            # self.prev_enemies_alives = self.current_enemies_alives
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
            # print("friends health: ",self.friends_current_health)
            # print("enemies health: ",self.enemies_current_health)

            # 计算上一轮动作执行所产生的 reward
            # Note: 此处的reward 是返回给上一轮执行的动作（上一轮动作执行的后果）
            # avg_friends_delta_health = friends_sum_delta_health / DDPG_CFG.agent_num
            # avg_friends_delta_health = friends_sum_delta_health / len(self.prev_alives)
            # avg_enemies_delta_health = enemies_sum_delta_health / DDPG_CFG.enemy_num
            # avg_enemies_delta_health = enemies_sum_delta_health / len(self.prev_enemies_alives)
            # reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*DDPG_CFG.agent_num, dtype=np.float32)
            # reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*len(self.prev_alives), dtype=np.float32)
            reward_part1 = enemies_sum_delta_health * 10 - friends_sum_delta_health
            # 杀掉一个 +10
            reward_part2 = self._kill_enemies(obs) * 100
            reward_part3 = 0
            if self.is_roach_defeated(obs):
                reward_part3 = 2000 + friends_sum_health
            reward = reward_part1 + reward_part2 + reward_part3

            self.friends_pre_health = self.friends_current_health
            self.enemies_pre_health = self.enemies_current_health
            # 存储 pre step alive units
            # self.prev_alives = self.current_alives
            # self.prev_enemies_alives = self.current_enemies_alives
            return reward

    def _convert_actions_2_sc2(self, model_output_actions, alive_friends_locations):
        '''
        将模型输出的动作，转换为 pysc2 可执行的动作
        :param model_output_actions: 所有模型输出的动作（二维数组） [0, 64, 64] 0: move 1: attack
        :return:
        '''
        actions_queue = []
        actions_original = []
        arg_select_point_act = [0]  # Select,Toggle,SelectAllOfType,AddAllType
        for location, action in zip(alive_friends_locations, model_output_actions):
            # select position 选中当前单位
            arg_select_position = location
            actions_queue.append(actions.FunctionCall(_SELECT_POINT, [arg_select_point_act, arg_select_position]))
            # 存活 unit 对应模型输出对应的动作
            # 针对存活 unit 动作的每一维度添加噪声 add random noise to sigmoid actions
            if self.is_training:
                # print("before: ",action)
                action[0] = self._add_noise_and_reshape(action[0], DDPG_CFG.action_sigmoid_vars[0], 0, 1)
                action[1] = self._add_noise_and_reshape(action[1], DDPG_CFG.action_sigmoid_vars[1], 0, 1)
                action[2] = self._add_noise_and_reshape(action[2], DDPG_CFG.action_sigmoid_vars[2], 0, 1)
                # print("after: ",action)

            actions_original.append(action)
            coordination = [math.floor(action[1] * DDPG_CFG.action_bound[1]), math.floor(action[2] * DDPG_CFG.action_bound[2])]
            # print("after: ", coordination)
            if action[0] >= 0.5: # attack
                actions_queue.append(actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, coordination]))
            else:  # move
                actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coordination]))
        return actions_original, actions_queue

    def is_roach_defeated(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            self.pre_alived_roaches = 0
            self.win_times += 1
            return True
        else:
            self.pre_alived_roaches = len(self._get_raw_opponents_data(obs))
            return False

    def random_action_by_location(self, obs):
        # 要执行的动作队列
        actions_queue = []
        actions_original = []
        friends = self._get_raw_friends_data(obs)
        enemies = self._get_raw_opponents_data(obs)
        # 选中一个
        arg_select_point_act = [0]  # Select,Toggle,SelectAllOfType,AddAllType
        for id in range(len(friends)):
            cur_friend = friends[id]
            arg_select_position = [cur_friend[3], cur_friend[4]]
            actions_queue.append(actions.FunctionCall(_SELECT_POINT, [arg_select_point_act, arg_select_position]))
            if random.random() < 10:
                # 随机挑选一个敌人攻击
                select = enemies[0]
                target = [select[3], select[4]]
                # actions_queue.append(actions.FunctionCall(_ATTACK_ATTACK_SCREEN, [_NOT_QUEUED, [roaches_x[select], roaches_y[select]]]))
                actions_queue.append(
                    actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target]))
                actions_original.append([1, target[0], target[1]])
            else:
                # 地图上随机选一个点逃跑
                # 移动一个
                # target = [random.randint(0, 83), random.randint(0, 83)]
                select = enemies[0]
                target = [select[3], select[4]]
                actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]))
                actions_original.append([0, target[0], target[1]])
        return actions_original, actions_queue

    def _convert_2_all_actions(self, original_actions, model_ids):
        all_actions = [[0., 0., 0.]] * DDPG_CFG.agent_num
        for id, model_id in enumerate(model_ids):
            all_actions[model_id] = original_actions[id]
        all_actions = np.array(all_actions, dtype=np.float32)
        all_actions = np.hstack(all_actions)
        return all_actions

    # TODO: 只训练对应时刻活着单位对应的 model
    def step(self, obs):
        super(ControllUnits, self).step(obs)
        # time.sleep(10)  # 休眠 1s 方便观察

        # 1：开局记录本局所有 unit 的 tag 与 model 对应关系
        if self.isFirst(obs):
            # 记录所有存活单位的信息
            self.all_friends_tag = self._get_raw_friends_data(obs)[:,0]
            self.all_enemies_tag = self._get_raw_opponents_data(obs)[:,0]
            self.friends_tag_2_id = {tag: id for id, tag in enumerate(self.all_friends_tag)}
            self.enemies_tag_2_id = {tag: id for id, tag in enumerate(self.all_enemies_tag)}

        # 2: 计算上一轮的 reward （当然，第一轮没有用）
        reward = self._calculated_reward(obs)
        self.cumulative_reward += reward

        # 如果还没有结束(这里是胜利)
        if not self.is_roach_defeated(obs):
            # 3：拿到本次观察，执行动作 [1,2,3, 4,5,6, 7,8,9]
            features, alive_model_ids, alive_friends_locations = self._input_feature_pre_processing(obs)
            if self.isLast(obs):
                # 我输了
                actions_original,actions_queue = [], []
                actions_queue.append(actions.FunctionCall(_NO_OP, []))

                self.update_exploration(time.time()-self.pre_update_exploration_time)
                print("loss: episode {}, total step {}, reward: {}, cumulative win times: {}".format(self.episodes, self.steps,
                                                                                               self.cumulative_reward,
                                                                                              self.cumulative_win_times))
            else:
                # TODO: 执行动作，活着的执行就可以了
                actions_to_convert = []
                for model_id in alive_model_ids[0]:
                    current_unit_obs = np.copy(features[model_id * DDPG_CFG.state_dim:(model_id+1) * DDPG_CFG.state_dim])
                    current_unit_obs = np.concatenate([current_unit_obs, features],axis=0)
                    # actor 模型输出
                    model_output_action = self.actor.operation_get_action_to_environment([current_unit_obs], is_training=False)
                    # 单 agent 输入，压缩维度
                    actions_to_convert.append(np.squeeze(model_output_action))
                actions_original, actions_queue = self._convert_actions_2_sc2(actions_to_convert, alive_friends_locations)
                # actions_original, actions_queue = self.random_action_by_location(obs)

            # 4：如果不是开始则存储 (s,a,r,s_,done,model alived ids)
            if not self.isFirst(obs) and self.is_training:
                model_ids = self.pre_model_ids[0]
                s = self.pre_observation
                a = self._convert_2_all_actions(self.pre_actions,model_ids)
                r = reward
                s_ = features
                done = self.isLast(obs) # 可能有结束的情况
                transition = construct_transition(a, r, s_, done, model_ids)
                # print(transition)
                self.replay_buffer.store(transition)

            # 5: update prev properties
            self.pre_observation = features
            self.pre_actions = actions_original
            self.pre_model_ids = alive_model_ids
        else:
            # TODO: 星际2 mini game 每局打赢后还会额外增5个新兵继续
            # If goes here, current episode is ended and our team win
            # 执行动作 no_op
            actions_queue = []
            actions_queue.append(actions.FunctionCall(_NO_OP, []))

            if self.win_times == 2 and self.is_training:  # 只记录一次
                # 我赢了
                self.update_exploration(time.time() - self.pre_update_exploration_time)
                self.cumulative_win_times += 1
                # 6 :记录最后 tuple
                model_ids = self.pre_model_ids[0]
                s = self.pre_observation
                a = self._convert_2_all_actions(self.pre_actions, model_ids)
                r = reward
                # s_ = None
                s_ = s
                done = True
                transition = construct_transition(a, r, s_, done, model_ids)
                # print(transition)
                self.replay_buffer.store(transition)
                # win
                print("win: episode {}, total step {}, reward: {}, cumulative win times: {}".format(self.episodes,
                                                                                                     self.steps,
                                                                                                     self.cumulative_reward,
                                                                                                     self.cumulative_win_times))


        # TODO: batch training
        if self.replay_buffer.length >= DDPG_CFG.batch_size * 2 and self.is_training and self.win_times <= 2:
            batch_s, batch_a, batch_r, batch_s_, batch_done, batch_training_model_ids = self.replay_buffer.sample_batch(DDPG_CFG.batch_size)
            batch_action_s_ = []
            batch_actor_units_all_states = []
            batch_critic_units_all_states = []
            batch_actor_units_all_actions = []
            batch_actor_units_all_baseline_actions = []
            batch_actor_units_model_ids = []

            for batch_idx in range(len(batch_s)):
                # 对一次数据里的每一个 存活的unit
                s, s_, model_ids = batch_s[batch_idx], batch_s_[batch_idx],batch_training_model_ids[batch_idx]
                temp_batch_s, temp_batch_s_, temp_batch_model_ids = [], [], []
                for model_id in model_ids:
                    # 为 actor 组装 batch state
                    current_unit_obs = np.copy(s[model_id * DDPG_CFG.state_dim:(model_id + 1) * DDPG_CFG.state_dim])
                    current_unit_obs = np.concatenate([current_unit_obs, s], axis=0)
                    temp_batch_s.append(current_unit_obs)

                    current_unit_obs_ = np.copy(s_[model_id * DDPG_CFG.state_dim:(model_id + 1) * DDPG_CFG.state_dim])
                    current_unit_obs_ = np.concatenate([current_unit_obs_, s_], axis=0)
                    temp_batch_s_.append(current_unit_obs_)


                # get the target action for s_
                alive_action_actor_next = self.actor.operation_get_action_to_TDtarget(temp_batch_s_, is_training=True)
                all_action_next = self._convert_2_all_actions(alive_action_actor_next, model_ids)
                batch_action_s_.append(all_action_next)

                # 存活单位的输出
                units_actor_outputs = self.actor.operation_get_action_to_environment(temp_batch_s, is_training=True)
                all_actions = self._convert_2_all_actions(units_actor_outputs, model_ids)
                units_all_baseline_actions = []
                units_all_critic_states = []
                units_all_actions = []
                for model_id in model_ids:
                    baseline_all_actions = np.copy(all_actions)
                    # 对应 模型的输入设为 default
                    baseline_all_actions[model_id * DDPG_CFG.action_dim: (model_id+1) * DDPG_CFG.action_dim] = 0.
                    units_all_baseline_actions.append(baseline_all_actions)

                    units_all_actions.append(all_actions)
                    units_all_critic_states.append(s)
                    batch_actor_units_model_ids.append(model_id)

                batch_critic_units_all_states.extend(units_all_critic_states)
                batch_actor_units_all_actions.extend(units_all_actions)
                batch_actor_units_all_baseline_actions.extend(units_all_baseline_actions)
                batch_actor_units_all_states.extend(temp_batch_s)

            # training critic:
            batch_td_target = self.critic.operation_get_TDtarget(batch_action_s_, batch_s_, batch_r, batch_done, is_training=True)
            self.critic.operation_critic_learn(batch_td_target, batch_s, batch_a, is_training=True)

            # training actor
            q_to_a_gradients = self.critic.operation_get_gradient(
                np.array(batch_critic_units_all_states),
                np.array(batch_actor_units_all_actions),
                np.array(batch_actor_units_all_baseline_actions),
                batch_actor_units_model_ids,
                is_training=True
            )

            self.actor.operation_actor_learn(q_to_a_gradients, batch_actor_units_all_states, is_training=True)

            # soft update the parameters of the two model
            # print("soft update parameters: episode {}, step {}, reward： {}".format(self.episodes, self.steps, reward))
            self.actor.operation_soft_update_TDnet()
            self.critic.operation_soft_update_TDnet()


        if self.is_training and (time.time() - self.pre_save_time) > 3600:
            # print()
            self.saver.save(self.sess, "checkpoint_{}/model.ckpt".format(self.model_id))
            self.pre_save_time = time.time()
            self.model_id += 1
        # print("执行动作：",actions_queue)
        return actions_queue

    def isFirst(self, obs):
        return obs.first()

    def isMid(self, obs):
        return obs.mid()

    def isLast(self, obs):
        return obs.last()

    # 自己每死一个单位：-1，Roach defeated: +10

    # 获取己方单位原始信息
    def _get_raw_friends_data(self, obs):
        # print("_get_raw_friends_data: \n", obs.observation['my_units'])
        return obs.observation['my_units']

    # 获取敌方单位原始信息
    def _get_raw_opponents_data(self, obs):
        # print("_get_raw_opponents_data: \n", obs.observation['my_opponents'])
        return obs.observation['my_opponents']