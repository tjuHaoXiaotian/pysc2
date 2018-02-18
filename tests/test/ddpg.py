#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import random
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents.test.common.replay_buffer import *


import tensorflow as tf
from pysc2.agents.test.common.actor import Actor
from pysc2.agents.test.common.ciritic import Critic

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
# =========================================actions end ===============================================

# ==========================================parameters ===============================================
_NOT_QUEUED = [0]
_SELECT_ALL = [0] # (select vs select_add)
# =========================================parameters end ============================================

DDPG = tf.app.flags.FLAGS  # alias
DDPG.replay_buff_size = 1
DDPG.batch_size = 32
DDPG.replay_buff_size = 10**6  # 1M
# DDPG.replay_buff_save_segment_size = 30*3000  # every 180,000 Transition data.
DDPG.replay_buff_save_segment_size = 30  # every 180,000 Transition data.
DDPG.replay_buffer_file_path = "replay_buffer"
DDPG.random_seed = 1

DDPG.agent_num = 9
DDPG.enemy_num = 4
DDPG.action_dim = 3
DDPG.state_dim = [84, 84, 3]
DDPG.action_bound = [1, 83, 83]
DDPG.action_vars = [0.5, 40, 40]

class ControllUnits(base_agent.BaseAgent):

    print_infos = False

    def __init__(self):
        super(ControllUnits,self).__init__()
        # 实例化 replay buffer，指定是否将buffer数据保存到文件
        self.replay_buffer = ReplayBuffer(buffer_size=DDPG.replay_buff_size,
                               save_segment_size= DDPG.replay_buff_save_segment_size,
                               save_path=DDPG.replay_buffer_file_path,
                               seed=DDPG.random_seed
                               )

        self.sess = tf.Session()
        self.actor = Actor(self.sess, DDPG.action_dim, DDPG.state_dim, DDPG.action_bound)
        self.critic = Critic(self.sess, DDPG.action_dim, DDPG.state_dim)
        # initalize the variables
        self.sess.run(tf.global_variables_initializer())
        # copy all the values of the online model to the target model
        self.actor.operation_update_TDnet_compeletely()
        self.critic.operation_update_TDnet_compeletely()

        # saver 存储 variable parameter
        # self.saver = tf.train.Saver()

    def reset(self):
        super(ControllUnits, self).reset()

    # TODO: try to understand the random algorithm
    def _add_noise_and_reshape(self, action, var, lower_bound, upper_bound):
        '''
        :param action: 均值
        :param var: 方差
        :return:
        '''
        return np.clip(np.random.normal(action, var), lower_bound, upper_bound)
            # .reshape(DDPG.action_dim)

    def _input_feature_pre_processing(self,obs):
        '''
        预处理图片输入，转换为所需 feature
        :param obs: 环境 observation
        :return: [one_hot_location[self], one_hot_location[enemy], hit_points]
        '''
        # tensorflow style
        # one_hot_player_id = tf.one_hot(self._get_player_id(obs),depth=3, on_value=1, off_value=0, dtype=tf.int32)
        # channels = tf.unstack(one_hot_player_id, axis=2)
        # hit_points = self._get_hit_points(obs)
        # images = tf.stack([channels[1], channels[2], hit_points])

        # numpy style
        player_id = self._get_player_id(obs)
        hit_points = self._get_hit_points(obs)
        layer1 = np.copy(player_id)
        layer2 = player_id
        # 己方单位
        layer1[layer1 == 2] = 0
        # 对方单位
        layer2[layer2 == 1] = 0
        layer2[layer2 == 2] = 1
        layer3 = hit_points
        images = np.stack([layer1, layer2, layer3], axis=2)
        return images

    def _get_training_needed_model_and_calculated_reward(self, obs):
        '''
        计算己方每个单位的 reward
        :param obs: 环境观察
        :return: training_needed_model_ids, reward
        '''
        friends = self._get_raw_friends_data(obs)
        # 为后续 存储 prev_alives 及 为当前存活的 agent 选择动作
        self.current_alives = friends[:, 0]
        enemies = self._get_raw_opponents_data(obs)
        self.current_enemies_alives = enemies[:, 0]
        friends_current_health = friends[:, [0,6]]
        enemies_current_health = enemies[:, [0,6]]

        if self.isFirst(obs):
            # first time： reward is 0
            # 存储 pre step hit_points
            self.pre_friends_current_health = {tag: health for tag, health in friends_current_health}
            self.pre_enemies_current_health = {tag: health for tag, health in enemies_current_health}
            # 存储 pre step alive units
            self.prev_alives = self.current_alives
            self.prev_enemies_alives = self.current_enemies_alives

            # self.unit_order = [friend[0] for friend in friends_current_health]
            self.unit_order = friends[:, 0]
            self.tag_to_idx = {tag:idx for idx,tag in enumerate(self.unit_order)}
            # self.enemy_order = [enemy[0] for enemy in enemies_current_health]
            self.enemy_order = enemies[:, 0]
            return None, np.zeros([DDPG.agent_num,1],dtype=np.float32)
        else:
            self.cur_friends_current_health = {tag: health for tag, health in friends_current_health}
            self.cur_enemies_current_health = {tag: health for tag, health in enemies_current_health}
            friends_sum_delta_health, enemies_sum_delta_health = 0., 0.
            for tag in self.unit_order:
                cur = self.cur_friends_current_health.get(tag, -1)
                # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
                if cur == -1:
                    self.cur_friends_current_health[tag] = 0
                    friends_sum_delta_health += (self.pre_friends_current_health[tag] - 0)
                else:
                    friends_sum_delta_health += (self.pre_friends_current_health[tag] - cur)
            for tag in self.enemy_order:
                cur = self.cur_enemies_current_health.get(tag, -1)
                # 当前返回列表中没有对应 unit，则对应unit已死亡，设置其生命值为 0
                if self.cur_enemies_current_health.get(tag,-1) == -1:
                    self.cur_enemies_current_health[tag] = 0
                    enemies_sum_delta_health += (self.pre_enemies_current_health[tag] - 0)
                else:
                    enemies_sum_delta_health += (self.pre_enemies_current_health[tag] - cur)
            # print("friends health: ",self.cur_friends_current_health)
            # print("enemies health: ",self.cur_enemies_current_health)

            # 计算上一轮动作执行所产生的 reward
            # Note: 此处的reward 是返回给上一轮执行的动作（上一轮动作执行的后果）
            # avg_friends_delta_health = friends_sum_delta_health / DDPG.agent_num
            avg_friends_delta_health = friends_sum_delta_health / len(self.prev_alives)
            # avg_enemies_delta_health = enemies_sum_delta_health / DDPG.enemy_num
            avg_enemies_delta_health = enemies_sum_delta_health / len(self.prev_enemies_alives)
            # reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*DDPG.agent_num, dtype=np.float32)
            reward = np.array([avg_enemies_delta_health-avg_friends_delta_health]*len(self.prev_alives), dtype=np.float32)
            training_needed_model_ids = [self.tag_to_idx[tag] for tag in self.prev_alives]

            self.pre_friends_current_health = self.cur_friends_current_health
            self.pre_enemies_current_health = self.cur_enemies_current_health
            # 存储 pre step alive units
            self.prev_alives = self.current_alives
            self.prev_enemies_alives = self.current_enemies_alives
            return training_needed_model_ids, reward

    def _convert_actions(self, model_output_actions):
        '''
        将模型输出的动作，转换为 pysc2 可执行的动作
        :param model_output_actions: 所有模型输出的动作（二维数组） [0, 64, 64] 0: move 1: attack
        :return:
        '''
        actions_queue = []
        actions_original = []
        actions_model_ids = []
        # current_alives 就是当前发动作的顺序
        for select_point, agent_tag in enumerate(self.current_alives):
            # 模型 id
            idx = self.tag_to_idx[agent_tag]
            actions_model_ids.append(idx)
            
            # select point 选中当前单位
            arg_select_unit = [select_point]
            arg_select_unit_act = [0]  # SingleSelect,DeselectUnit,SelectAllOfType,DeselectAllOfType
            actions_queue.append(actions.FunctionCall(_SELECT_UNIT, [arg_select_unit_act, arg_select_unit]))
            # 存活 unit 对应模型输出对应的动作
            action = model_output_actions[idx]
            # 针对存活 unit 动作的每一维度添加噪声
            action[0] = self._add_noise_and_reshape(action[0], DDPG.action_vars[0], 0, DDPG.action_bound[0])
            action[1] = self._add_noise_and_reshape(action[1], DDPG.action_vars[1], 0, DDPG.action_bound[1])
            action[2] = self._add_noise_and_reshape(action[2], DDPG.action_vars[2], 0, DDPG.action_bound[2])

            actions_original.append(action)

            if action[0] >= 0.5: # attack
                actions_queue.append(actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [round(action[1]), round(action[2])] ]))
            else:  # move
                actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [round(action[1]), round(action[2])]]))
        # 选中所有队员
        actions_queue.append(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]))
        
        return actions_original, actions_queue, actions_model_ids

    # TODO: 只训练对应时刻活着单位对应的 model
    
    def random_action(self, obs):
        time.sleep(0.2)  # 休眠 1s 方便观察
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        roaches_y, roaches_x = (player_relative == _PLAYER_HOSTILE).nonzero()  # 1
        # 要执行的动作队列
        actions_queue = []
        actions_original = []
        if self.isFirst(obs):  # 第一次：为所有队员编队
            pass
            # # 选中所有队员
            # actions_queue.append(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]))
            # 编队
            # self._build_group(actions_queue)

        # 选中一个
        arg_select_unit_act = [0]  # SingleSelect,DeselectUnit,SelectAllOfType,DeselectAllOfType
        for id in range(len(self.current_alives)):
            arg_select_unit = [id]
            # arg_select_unit = [self.current % 9]
            actions_queue.append(actions.FunctionCall(_SELECT_UNIT, [arg_select_unit_act, arg_select_unit]))
            if random.random() < 0:
                # 随机挑选一个敌人攻击
                select = random.randint(0, len(roaches_x) - 1)
                # target = [random.randint(0, 83), random.randint(0, 83)]
                # target = [83, 0]
                # actions_queue.append(actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target]))
                # actions_queue.append(actions.FunctionCall(_ATTACK_ATTACK_SCREEN, [_NOT_QUEUED, [roaches_x[select], roaches_y[select]]]))
                actions_queue.append(
                    actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [roaches_x[select], roaches_y[select]]]))
                actions_original.append([1, roaches_x[select], roaches_y[select]])
            else:
                # 地图上随机选一个点逃跑
                # 移动一个
                target = [random.randint(0, 83), random.randint(0, 83)]
                actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]))
                actions_original.append([0, target[0], target[1]])
        # 选中所有队员
        actions_queue.append(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]))
        actions_model_ids = [self.tag_to_idx[tag] for tag in self.current_alives]
        return actions_original, actions_queue, actions_model_ids
    
    def step(self, obs):
        super(ControllUnits, self).step(obs)
        # time.sleep(10)  # 休眠 1s 方便观察

        model_ids, reward = self._get_training_needed_model_and_calculated_reward(obs)

        # 执行动作
        actions_original,actions_queue,actions_model_ids = self.random_action(obs)
        # model_output_action = self.actor.operation_get_action_to_environment([self._input_feature_pre_processing(obs)])
        # model_output_actions = [model_output_action] * DDPG.agent_num
        # actions_original, actions_queue, actions_model_ids = self._convert_actions(model_output_actions)

        if not self.isFirst(obs):
            # 不是开局，存储(s,a,r,s_,done,tag_initial_index)
            s = self.pre_observation
            a = self.pre_actions  # batch action 只用于训练 Critic 网络
            r = reward
            s_ = self._input_feature_pre_processing(obs)
            done = self.isLast(obs)
            training_needed_idx = model_ids # 对应着哪个需要训练哪个 model
            # training_needed_idx.append(" : ")
            # training_needed_idx.extend(self.actions_model_ids)
            transition = construct_transition(a, r, s_, done, training_needed_idx)
            self.replay_buffer.store(transition)
            # print(transition)
            

            # TODO: batch training
            if self.replay_buffer.length >= DDPG.batch_size:
                batch_s, batch_a, batch_r, batch_s_, batch_done, batch_training_needed_idx = self.replay_buffer.sample_batch(DDPG.batch_size)
                # training critic:
                batch_action_next = self.actor.operation_get_action_to_TDtarget(batch_s_)
                batch_td_target = self.critic.operation_get_TDtarget(batch_action_next,batch_s_,batch_r,batch_done)
                self.critic.operation_critic_learn(batch_td_target, batch_s, batch_a)

                # training actor
                actor_outputs = self.actor.operation_get_action_to_environment(batch_s)
                q_to_a_gradients = self.critic.operation_get_gradient(batch_s, actor_outputs)
                self.actor.operation_actor_learn(q_to_a_gradients, batch_s)

                # soft update the parameters of the two model
                print("soft update parameters: episode {}, step {}".format(self.episodes, self.steps))
                self.actor.operation_soft_update_TDnet()
                self.critic.operation_soft_update_TDnet()

        # update prev properties
        if self.isFirst(obs):
            self.pre_observation = self._input_feature_pre_processing(obs)
        else:
            self.pre_observation = s_
        self.pre_actions = actions_original
        self.actions_model_ids = actions_model_ids
        # actions_queue.append(actions.FunctionCall(_NO_OP, []))
        # print("执行动作：",actions_queue)
        return actions_queue


    # 编队
    def _build_group(self, actions_queue):
        # 为所有队员编队
        arg_select_unit_act = [0]  # SingleSelect,DeselectUnit,SelectAllOfType,DeselectAllOfType
        for group in range(9):
            arg_select_unit = [group]
            actions_queue.append(actions.FunctionCall(_SELECT_UNIT, [arg_select_unit_act, arg_select_unit]))
            # 把选中的 unit 添加到编队中
            actions_queue.append(actions.FunctionCall(_SELECT_CONTROL_GROUP, [[1], [group]]))  # 1:set, 0:recall

    # 获取所有单位位置信息 Takes values in [0, 4], denoting [background, self, ally, neutral, enemy]

    # 获取玩家分布信息
    def _get_player_id(self, obs):
        player_id = obs.observation["screen"][_PLAYER_ID]
        # f = open("obs/player_id.txt", "a")
        # np.set_printoptions(threshold=np.inf)
        # for row in range(len(player_id)):
        #     row_str = ""
        #     for col in range(len(player_id[row])):
        #         row_str += (str(player_id[row][col]) + " ")
        #     print(row_str, file=f)
        # f.flush()
        # f.close()
        return player_id

    def _get_player_relative(self, obs):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        marines_y, marines_x = (player_relative == _PLAYER_FRIENDLY).nonzero()  # 1
        print("队友坐标位置信息", '\n', marines_x, "\n", marines_y)
        roaches_y, roaches_x = (player_relative == _PLAYER_HOSTILE).nonzero()  # 1
        print("敌人坐标位置信息", '\n', roaches_x, "\n", roaches_y)

        if self.print_infos:
            f = open("obs/player_relative.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(player_relative,'\n', file=f)
            f.flush()
            f.close()

        return player_relative

    # (单位种类)暂时没用
    def _get_unit_type(self, obs):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        if self.print_infos:
            f = open("obs/unit_type.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(unit_type,'\n', file=f)
            f.flush()
            f.close()
        return unit_type

    def _get_hit_points(self, obs):
        hit_points = obs.observation["screen"][_HIT_POINTS]
        # if self.print_infos:
        # f = open("obs/hit_points.txt", "a")
        # np.set_printoptions(threshold=np.inf)
        # for row in range(len(hit_points)):
        #     row_str = ""
        #     for col in range(len(hit_points[row])):
        #         row_str += (str(hit_points[row][col]) + " ")
        #     print(row_str, file=f)
        # f.flush()
        # f.close()
        return hit_points

    # 暂时没用，都为 0
    def _get_energy(self, obs):
        energy = obs.observation["screen"][_ENERGY]
        if self.print_infos:
            f = open("obs/energy.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(energy,'\n', file=f)
            f.flush()
            f.close()

        return energy

    def _get_effects(self, obs):
        effects = obs.observation["screen"][_EFFECTS]
        if self.print_infos:
            f = open("obs/effects.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(effects,'\n', file=f)
            f.flush()
            f.close()

        return effects

    # 获取队友（生命值信息）信息
    def _get_multi_select(self, obs):
        print("multi_select: ", obs.observation['multi_select'])
        return obs.observation['multi_select']

    def isFirst(self, obs):
        return obs.first()

    def isMid(self, obs):
        return obs.mid()

    def isLast(self, obs):
        return obs.last()

    # 自己每死一个单位：-1，Roach defeated: +10
    def get_reward(self, obs):
        print("奖励：", obs.reward)
        return obs.reward

    # 累积分数（暂时用不到）
    def get_score_cumulative(self, obs):
        print("score_cumulative: ", obs.observation['score_cumulative'])
    # 游戏 step（暂时用不到）
    def get_game_loop(self, obs):
        print("game_loop: ", obs.observation['game_loop'])

    # 获取己方单位原始信息
    def _get_raw_friends_data(self, obs):
        # f = open("play_observation/trace/raw_units.txt".format(time.time()), "a")  # 打开文件以便写入
        # np.set_printoptions(threshold=np.inf)  # 全部输出
        # # f.write(str(self._obs))
        # print(obs.observation['my_units'][0][7], file=f)
        # f.flush()
        # f.close()
        # print("_get_raw_friends_data: \n", obs.observation['my_units'])
        return obs.observation['my_units']

    # 获取敌方单位原始信息
    def _get_raw_opponents_data(self, obs):
        # print("_get_raw_opponents_data: \n", obs.observation['my_opponents'])
        return obs.observation['my_opponents']