#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import OneHotEncoder

from pysc2.lib import actions

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

_max_health_marines = 45.
_max_health_roaches = 145.
_max_dx = 83.
_max_dy = 62.
_max_distance = math.sqrt(_max_dx ** 2 + _max_dy ** 2)
_max_facing = 360.

def cal_local_observation_for_unit(current_unit, current_alive_friends, current_alive_enemies, units_tag_2_id):
    '''
    为 current_unit 准备局部观察信息
    :param current_unit:  当前 unit 信息
    :param current_alive_friends:  当前所有存活的己方单位
    :param current_alive_enemies:  当前所有存活的敌方单位
    :return: [ 己方单位由近及远（存活—> 死亡）, 敌方单位由近及远（存活—> 死亡）] 每个单位 11 维信息 * 13 个单位
              [alive, owner, 类型, relative_distance, dx, dy, facing, health]
    '''
    alive_friends_order = []
    alive_friends = []
    alive_enemies = []
    all_alives = []
    # 拼接存活 己方单位
    for friend in current_alive_friends:
        alive = 1
        tag = friend[0]
        unit_type = friend[1]
        unit_owner = friend[2]
        dx = friend[3] - current_unit[3]
        dy = friend[4] - current_unit[4]
        relative_distance = math.sqrt(dx ** 2 + dy ** 2) / _max_distance
        dx  /= _max_dx
        dy  /= _max_dy
        facing = friend[5] / _max_facing
        health = friend[6] / _max_health_marines
        relative_friend = [alive, unit_owner, unit_type, relative_distance, dx, dy, facing, health]
        alive_friends_order.append(units_tag_2_id[tag])
        alive_friends.append(relative_friend)

    # 拼接存活 敌方单位
    for enemy in current_alive_enemies:
        alive = 1
        unit_type = enemy[1]
        unit_owner = enemy[2]
        dx = enemy[3] - current_unit[3]
        dy = enemy[4] - current_unit[4]
        relative_distance = math.sqrt(dx ** 2 + dy ** 2) / _max_distance
        dx /= _max_dx
        dy /= _max_dy
        facing = enemy[5] / _max_facing
        health = enemy[6] / _max_health_roaches
        relative_enemy = [alive, unit_owner, unit_type, relative_distance, dx, dy, facing, health]
        alive_enemies.append(relative_enemy)

    all_alives.extend(alive_friends)
    all_alives.extend(alive_enemies)
    # 按照相对距离从远到近排序
    all_alives = sorted(all_alives, key=lambda x: (x[3]), reverse=True)

    # print(all_alives)

    # 按照相对距离从远到近排序
    alive_enemies = sorted(alive_enemies, key=lambda x: (x[3]), reverse = True)

    # 按照相对距离从远到近排序
    alive_friends_and_order = sorted(zip(alive_friends_order,alive_friends), key=lambda x: (x[1][3]), reverse = True)
    alive_friends_order = [item[0] for item in alive_friends_and_order]
    alive_friends = [item[1] for item in alive_friends_and_order]

    for friend_dead in range(9 - len(current_alive_friends)):
        alive = 0
        unit_type = 48
        unit_owner = 1
        dx = 0
        dy = 0
        relative_distance = 0
        facing = 0
        health = 0
        relative_friend_dead = [alive, unit_owner, unit_type, relative_distance, dx, dy, facing, health]
        all_alives.insert(0,relative_friend_dead)


    for enemy_dead in range(4 - len(current_alive_enemies)):
        alive = 0
        unit_type = 110
        unit_owner = 2
        dx = 0
        dy = 0
        relative_distance = 0
        facing = 0
        health = 0
        relative_enemy_dead = [alive, unit_owner, unit_type, relative_distance, dx, dy, facing, health]
        all_alives.insert(0,relative_enemy_dead)

    tmp = np.array(all_alives, dtype=np.float32)

    # one-hot encoding
    enc = OneHotEncoder()
    enc.fit([[1,1,48], [0,1,48], [1,2,110], [0,2,110]])
    part1 = enc.transform(tmp[:, [0, 1, 2]]).toarray()
    part2 = tmp[:, 3:]
    features = np.hstack([part1,part2])
    # print(features)
    # return features
    return np.hstack(features), None if len(alive_friends_order) == 0 else alive_friends_order

Action = namedtuple('Action', ['stop', 'noop', 'move_up', 'move_right', 'move_down', 'move_left'
                               'attack_0', 'attack_1', 'attack_2', 'attack_3'])


def convert_discrete_action_2_sc2_action(current_unit, action_id, alive_enemies, all_enemies_id_2_tag):
    actions_queue = []
    # select position 选中当前单位
    location = [current_unit[3], current_unit[4]]
    actions_queue.append(actions.FunctionCall(_SELECT_POINT, [_SELECT, location]))
    if action_id == 0: # stop
        actions_queue.append(actions_queue.append(actions.FunctionCall(_STOP_QUICK, [_NOT_QUEUED])))
    elif action_id == 1:  # noop
        actions_queue.append(actions.FunctionCall(_NO_OP, []))
    elif action_id <= 5:  # move
        if action_id == 2: # move_up
            target_location = [current_unit[3], 0]
        elif action_id == 3: # move_right
            target_location = [_SCREEN_SIZE[0], current_unit[4]]
        elif action_id == 4: # move_down
            target_location = [current_unit[3], _SCREEN_SIZE[1]]
        else:  # move_left
            target_location = [0, current_unit[4]]
        actions_queue.append(actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target_location]))
    else:
        alive_enemies = {enemy[0]: enemy for enemy in alive_enemies}
        if action_id == 6: # attack_0
            target = alive_enemies.get(all_enemies_id_2_tag[0], None)
        elif action_id == 7: # attack_1
            target = alive_enemies.get(all_enemies_id_2_tag[1], None)
        elif action_id == 8: # attack_2
            target = alive_enemies.get(all_enemies_id_2_tag[2], None)
        else: # attack_3
            target = alive_enemies.get(all_enemies_id_2_tag[3], None)
        if target is not None:
            actions_queue.append(actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [target[3], target[4]]]))
        else: # 目标单位已死亡
            actions_queue.append(actions.FunctionCall(_NO_OP, []))
    return actions_queue

def one_hot_action(action_id):
    action = np.zeros([10,],dtype=np.int32)
    action[action_id] = 1
    return action
