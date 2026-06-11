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

    def get_reward_from_action(self, action:int):
        if(action >= self.arm_nums):
            raise ValueError("index overange len")

        return np.random.normal(loc=self.ave_reward[action], scale=self.scale_on_same_arm)
    
    def get_gt_reword(self):
        return self.ave_reward

class EpsilonGreedModel():
    def __init__(self, k, epsilon=0.01):

        self.k = k # 动作长度
        self.epsilon = epsilon

        # 决策的模型应该如何设置？ 多层神经网络吗，但是输入是什么， 使用epsilon-贪婪模型
        # k臂赌博机问题是非上下文的，也就是state-action-reward循环中不需要考虑state因素
        
        self.action_value_func = [0 for i in range(k)] # 衡量每个动作的价值
        self.action_times = [0 for i in range(k)]   # 记录选择每个动作的次数
        self.time_step = 0
        self.total_reward = 0

    def select_action(self):
        # 根据动作价值函数选择动作
        
        if np.random.rand() > self.epsilon:
            # 如果同时有多个最大值，随机返回一个
            arr = np.array(self.action_value_func)
            max_val = np.max(arr)
            max_indexs = np.where(arr == max_val)[0]
            return np.random.choice(max_indexs)
        else:
            return np.random.randint(self.k)

    
    def update(self, action, reward):
        # 更新state-action value
        old_times = self.action_times[action] 
        old_value = self.action_value_func[action] 
        
        new_value = (old_value * old_times + reward ) / (old_times + 1)

        self.action_times[action] += 1
        self.action_value_func[action] = new_value 
        self.time_step += 1
        self.total_reward += reward


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

def main():
    k = 10
    epsilon = 0.1
    epochs = 2000

    env = KArmedPundit(k = k)
    model = EpsilonGreedModel(k=env.arm_nums, epsilon=epsilon)

    for epoch in range(epochs):
        action = model.select_action()
        reward = env.get_reward_from_action(action=action)
        model.update(action, reward)
        print(f"{epoch} / {epochs}, Total_reward: {model.total_reward:.2f}")

    # 可视化真值
    gt = env.get_gt_reword()
    print(f"gt: {gt}")
    print(f"value:{model.action_value_func}")
    
    # 可视化每个动作选择次数
    x = np.arange(k)
    y = model.action_times
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.show()
if __name__ == "__main__":
    main()

# start = time.time()
# arm = KArmedPundit(k=10)
# test(arm)


# print(f"use_time:{time.time() - start}")

# epsilon越小，可能会导致模型找不到最优动作