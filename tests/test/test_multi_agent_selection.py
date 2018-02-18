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
_BEHAVIOR_HOLDFIREON_QUICK = actions.FUNCTIONS.Behavior_HoldFireOn_quick.id
_HOLDPOSITION_QUICK = actions.FUNCTIONS.HoldPosition_quick.id
_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
# _MOVE_SCREEN = actions.FUNCTIONS.Scan_Move_screen.id
_STOP_QUICK = actions.FUNCTIONS.Stop_quick.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_ATTACK_SCREEN = actions.FUNCTIONS.Attack_Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
# =========================================actions end ===============================================

# ==========================================parameters ===============================================
_NOT_QUEUED = [0]
_SELECT_ALL = [0]  # (select vs select_add)
# =========================================parameters end ============================================

'''
    1: 确定通按照 tag 排序后（从小到大），tag 的顺序 对应 数组index的顺序 (能用)
    2: 确定按照转换后的位置选择每个单位是可用的
'''

class ControllUnits(base_agent.BaseAgent):
    print_infos = False

    def __init__(self):
        super(ControllUnits, self).__init__()
        self.pre_alived_roaches = 4

        # saver 存储 variable parameter
        # self.saver = tf.train.Saver()

    def reset(self):
        super(ControllUnits, self).reset()
        self.pre_alived_roaches = 4

    # TODO: 只训练对应时刻活着单位对应的 model

    def random_action(self, obs):
        # time.sleep(0.2)  # 休眠 1s 方便观察
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
        for id in range(len(self.friends)):
            if(id == 2):
                arg_select_unit = [id]
                # arg_select_unit = [self.current % 9]
                actions_queue.append(actions.FunctionCall(_SELECT_UNIT, [arg_select_unit_act, arg_select_unit]))
                if random.random() < 10:
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
            else:
                actions_queue.append(actions.FunctionCall(_NO_OP, []))
            # 选中所有队员
            actions_queue.append(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]))
        return actions_original, actions_queue

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
        # 按照顺序重新排列
        prev_id = -1
        for alive_idx in range(len(alive_friends)):
            alive_tag = alive_friends[alive_idx][0]
            alive_origin_id = self.friends_tag_2_id[alive_tag]
            alive_friends_origin_ids.append(alive_origin_id)
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
        result = np.hstack([part1,part2])
        return result, [alive_friends_origin_ids,alive_enemies_origin_ids]

    def random_action_by_location(self):
        # 要执行的动作队列
        actions_queue = []
        actions_original = []

        # if self.steps == 0:
            # actions_queue.append(actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED]))
            # actions_queue.append(actions.FunctionCall(_BEHAVIOR_HOLDFIREON_QUICK, [_NOT_QUEUED]))
        # 选中一个
        arg_select_point_act = [0]  # Select,Toggle,SelectAllOfType,AddAllType
        for id in range(len(self.friends)):
            # if id == 0:
                cur_friend = self.friends[id]
                arg_select_position = [cur_friend[3], cur_friend[4]]
                # arg_select_unit = [self.current % 9]
                actions_queue.append(actions.FunctionCall(_SELECT_POINT, [arg_select_point_act, arg_select_position]))
                if random.random() < 10:
                    # 随机挑选一个敌人攻击
                    select = self.enemies[0]
                    # target = [random.randint(0, 83), random.randint(0, 83)]
                    # target = [83, 0]
                    # actions_queue.append(actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target]))
                    # actions_queue.append(actions.FunctionCall(_ATTACK_ATTACK_SCREEN, [_NOT_QUEUED, [roaches_x[select], roaches_y[select]]]))
                    actions_queue.append(
                        actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [select[3], select[4]]]))
                    actions_original.append([1, select[3], select[4]])
                else:
                    # 地图上随机选一个点逃跑
                    # 移动一个
                    # target = [random.randint(0, 83), random.randint(0, 83)]
                    select = self.enemies[0]
                    target = [select[3], select[4]]
                    # actions_queue.append(actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, target]))
                    actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]))
                    actions_original.append([0, target[0], target[1]])
                    # if self.steps % 3 == 0 and self.steps != 0:
                    #     actions_queue.append(actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED]))
                    #     actions_queue.append(actions.FunctionCall(_STOP_QUICK, [_NOT_QUEUED]))
                        # actions_queue.append(actions.FunctionCall(_NO_OP, []))

                # actions_queue.append(actions.FunctionCall(_NO_OP, []))
            # 选中所有队员
            # actions_queue.append(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]))
        return actions_original, actions_queue

    def _kill_enemies(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            # 全部杀光光
            current_enemies = 0
        else:
            current_enemies = len(self._get_raw_opponents_data(obs))
        return self.pre_alived_roaches - current_enemies

    def step(self, obs):
        super(ControllUnits, self).step(obs)

        if self.isFirst(obs):
            # 记录所有存活单位的信息
            self.all_friends_tag = self._get_raw_friends_data(obs)[:,0]
            self.all_enemies_tag = self._get_raw_opponents_data(obs)[:,0]
            self.friends_tag_2_id = {tag: id for id, tag in enumerate(self.all_friends_tag)}
            self.enemies_tag_2_id = {tag: id for id, tag in enumerate(self.all_enemies_tag)}

        self.get_reward(obs)
        print("killed enemies: ", self._kill_enemies(obs))
        # print("base agent reward: ", self.reward)

        # print("soft update parameters: episode {}, step {}".format(self.episodes, self.steps))
        # print("游戏进度是否结束：", self.isLast(obs))
        # print("available_actions: ", obs.observation['available_actions'])
        self.friends = self._get_raw_friends_data(obs)
        # 为后续 存储 prev_alives 及 为当前存活的 agent 选择动作
        self.enemies = self._get_raw_opponents_data(obs)

        self._get_player_relative(obs)
        actions_queue = []
        if self.is_roach_defeated(obs):
            # s' is no need
            # time.sleep(5)  # 休眠 1s 方便观察
            actions_queue.append(actions.FunctionCall(_NO_OP, []))
        else:
            # print(self.all_friends_tag)
            # print(self.all_enemies_tag)
            # print(self._input_feature_pre_processing(obs))

            actions_original, actions_queue = self.random_action_by_location()
        time.sleep(3)  # 休眠 1s 方便观察
        return actions_queue

    def is_roach_defeated(self, obs):
        if self.pre_alived_roaches < 4 and len(self._get_raw_opponents_data(obs)) == 4:
            self.pre_alived_roaches = 0
            return True
        else:
            self.pre_alived_roaches = len(self._get_raw_opponents_data(obs))
            return False

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
        # print("队友坐标位置信息", '\n', marines_x, "\n", marines_y)
        roaches_y, roaches_x = (player_relative == _PLAYER_HOSTILE).nonzero()  # 1
        # print("敌人坐标位置信息", '\n', roaches_x, "\n", roaches_y)

        if self.print_infos:
            f = open("obs/player_relative.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(player_relative, '\n', file=f)
            f.flush()
            f.close()

        return player_relative

    # (单位种类)暂时没用
    def _get_unit_type(self, obs):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        if self.print_infos:
            f = open("obs/unit_type.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(unit_type, '\n', file=f)
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
            print(energy, '\n', file=f)
            f.flush()
            f.close()

        return energy

    def _get_effects(self, obs):
        effects = obs.observation["screen"][_EFFECTS]
        if self.print_infos:
            f = open("obs/effects.txt", "a")
            np.set_printoptions(threshold=np.inf)
            print(effects, '\n', file=f)
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