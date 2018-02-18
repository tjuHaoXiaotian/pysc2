# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A base agent to write custom scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
import os

class BaseAgent(object):
    """A base agent to write custom scripted agents."""

    def __init__(self, enemies_num):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        # 用于判断输赢
        self.enemies_num = enemies_num
        self.pre_alive_enemies = enemies_num
        self.has_winned = False
        #

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1
        self.pre_alive_enemies = self.enemies_num
        self.has_winned = False

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        # 1： 每局开始记录所有存活单位的信息
        if self.is_first(obs):
            self.all_friends_tag = self.get_raw_friends_data(obs)[:, 0]
            self.all_enemies_tag = self.get_raw_opponents_data(obs)[:, 0]
            self.friends_tag_2_id = {tag: id for id, tag in enumerate(self.all_friends_tag)}
            self.friends_id_2_tag = dict(enumerate(self.all_friends_tag))
            self.enemies_tag_2_id = {tag: id for id, tag in enumerate(self.all_enemies_tag)}
            self.enemies_id_2_tag = dict(enumerate(self.all_enemies_tag))


        return actions.FunctionCall(0, [])

    def is_first(self, obs):
        return obs.first()

    def is_mid(self, obs):
        return obs.mid()

    def is_last(self, obs):
        return obs.last()

    def get_raw_friends_data(self, obs):
        '''
    获取己方单位原始信息
    :param obs:
    :return:
    '''
        return obs.observation['my_units']

    def get_raw_opponents_data(self, obs):
        '''
    获取敌方单位原始信息
    :param obs:
    :return:
    '''
        return obs.observation['my_opponents']

    # TODO: 只针对自带的两个地图中打赢随即敌人复活情况
    def is_win(self, obs):
        # 当前时刻存活敌人单位数量 > 上一时刻数量
        if len(self.get_raw_opponents_data(obs)) > self.pre_alive_enemies:
            self.has_winned = True

        return self.has_winned

    def is_loss(self, obs):
        return self.is_last(obs)

    def is_finished(self, obs):
        return self.is_win(obs) or self.is_loss(obs)

    def pre_action_kill_enemies_num(self, obs):
        if self.is_win(obs):
            current_enemies = 0
        else:
            current_enemies = len(self.get_raw_opponents_data(obs))
        return self.pre_alive_enemies - current_enemies

    def update_of_each_step(self, obs):
        '''
        last operation of the step() function
        :param obs:
        :return:
        '''
        # 1: update pre_alive_enemies
        if self.is_win(obs):
            self.pre_alive_enemies = 0
        else:
            self.pre_alive_enemies = len(self.get_raw_opponents_data(obs))


    def append_log_to_file(self, file_name, content):
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        f = open(file_name, "a")  # 打开文件以便写入
        # np.set_printoptions(threshold=np.inf)  # 全部输出
        # f.write(str(self._obs))
        print(content, file=f)
        f.flush()
        f.close()