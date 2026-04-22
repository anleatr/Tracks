# 例4.2 杰克租车问题
from dataclasses import dataclass
import random
import numpy as np

@dataclass(frozen=True) # 需要添加frozen, 否则不可哈希无法作为字典key
class State:
    place1: int # 第一个地点车辆
    place2: int

class Agent:
    
    def __init__(self):

        # 状态空间， 动作空间
        self.state_space = [State(i+1, j+1) for i in range(20) for j in range(20)]
        self.action_space = np.arange(-5, 6) # 正数代表从1移到2，负数表示从2移到1
        
        # 初始化策略，状态， 
        self.policy = self.init_policy()
        print(self.policy)


    def init_policy(self):
        # 构造一个从状态空间到动作空间的映射， 目前为止都是确定性策略
        policy = {}
        for state in self.state_space:
            policy[state] = random.choice(np.arange(-state.place2, state.place1+1))  # 移动数目需小于当前车辆数
        return policy

if __name__ == "__main__":
    agent = Agent()