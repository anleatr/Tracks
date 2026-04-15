一、你的代码里有 4 个致命错误
1. 损失函数写错了！你写的不是 MSE，是 MAE，且求导不稳定
你写的：
python
运行

loss = torch.sqrt((label - y)**2)  # 这是绝对值 |error|

这是 MAE，在 0 点不可导，非常容易导致梯度震荡 / 爆炸！
正确 MSE 应该是：
python
运行

loss = torch.mean((label - y) ** 2)

2. label 加了 requires_grad=True，完全错误！
python
运行

label = torch.tensor(label, dtype=torch.float32, requires_grad=True)

标签 永远不需要梯度，会破坏计算图，导致梯度异常、NaN。
3. forward 里把 x 转 tensor，破坏计算图
python
运行

x = torch.tensor(x, dtype=torch.float32)

x 本来就是 tensor，你重新包装 → 梯度断了，计算异常。
4. 学习率或初始化导致梯度爆炸
你用了：
python
运行

self.W = torch.rand(...)

torch.rand 是 0~1 均匀分布，非常容易梯度爆炸。深度学习都用 正态分布小随机数。

### 优化器
