# import torch 
import numpy as np
import matplotlib.pyplot as plt
import time

class KArmedPundit():
    def __init__(self, k:int = 10):
        # 生成一个长度为k， 服从正态分布数组，作为每个臂的平均奖励值
        self.arm_nums = k
        self.scale = 5
        self.scale_on_same_arm = 2

        self.ave_reward = np.random.normal(loc=0, scale=self.scale, size=self.arm_nums)

    def choess_arm(self, index:int):
        if(index >= self.arm_nums):
            raise ValueError("index overange len")

        return np.random.normal(loc=self.ave_reward[index], scale=self.scale_on_same_arm)
    
    def get_gt_reword(self):
        return self.ave_reward

class model():
    def __init__(self):

        # 决策的模型应该如何设置？ 多层神经网络吗，但是输入是什么

        self.time_step = 0
        self.ave_reward = 0
    
    

def test(karmpuchdit:KArmedPundit, i=2000):
    # 测试函数， 我们测试两千次抽取的平均结果
    gt = karmpuchdit.get_gt_reword()
    _len = karmpuchdit.arm_nums
    result = [[] for _ in range(_len)]  # 用一个二维数组存储结果

    for _ in range(i):
        # 随机抽取其中一个臂
        choice = np.random.randint(_len)  # [0, _len-1)
        result[choice].append(karmpuchdit.choess_arm(choice))
 
    # 可视化抽取结果
    # plt.figure(figsize=(10, 6))
    # for i, y_vals in enumerate(result):
    #     for y in y_vals:
    #         plt.scatter(i, y, color='black')
    # 收集所有点的坐标, 快了十几倍
    all_x = [i for i, y_vals in enumerate(result) for _ in y_vals]
    all_y = [y for y_vals in result for y in y_vals]

    plt.figure(figsize=(10, 6))
    plt.scatter(all_x, all_y, color='black')
    # 标记gt
    plt.plot(range(len(gt)), gt, 'r-o', linewidth=2, markersize=8, 
         label='Ground Truth', zorder=5)
    plt.show()

start = time.time()
arm = KArmedPundit(k=10)
test(arm)

print(f"use_time:{time.time() - start}")