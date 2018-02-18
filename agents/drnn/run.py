#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
python -m pysc2.bin.agent --agent pysc2.agents.coma.coma.Coma --map DefeatRoaches
'''

from pysc2.bin import agent

if __name__ == "__main__":
    argv = [
        'drnn',
        '--agent',
        'pysc2.agents.drnn.drnn.DRNN',
        '--map',
        'DefeatRoaches'
    ]

    agent.run(argv)