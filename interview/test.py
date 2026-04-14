import torch

# ===================== 1. 生成数据 =====================
# 多特征：n 个样本，d 个特征
n_samples = 1000
n_features = 5

# 真实权重 w 和 b
true_w = torch.tensor([2.5, -1.3, 0.7, 3.2, -0.5]).reshape(-1, 1)
true_b = torch.tensor([1.8])

# 生成 X, y
X = torch.randn(n_samples, n_features)  # (1000,5)
y = X @ true_w + true_b                # 线性关系
y += 0.1 * torch.randn_like(y)         # 加噪声

# ===================== 2. 初始化参数（需要梯度） =====================
w = torch.randn(n_features, 1, requires_grad=True)  # (5,1)
b = torch.zeros(1, requires_grad=True)             # 标量

# ===================== 3. 超参数 =====================
lr = 0.03
epochs = 100

# ===================== 4. 训练循环（从零实现） =====================
for epoch in range(epochs):
    # 1. 前向传播：y_hat = Xw + b
    y_hat = X @ w + b

    # 2. 计算 MSE 损失（手动）
    loss = torch.mean((y_hat - y) ** 2)

    # 3. 反向传播（自动求梯度）
    loss.backward()

    # 4. 手动更新参数（禁用梯度跟踪）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # 梯度清零！非常重要
        w.grad.zero_()
        b.grad.zero_()

    # 打印
    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss={loss.item():.4f}")

# ===================== 5. 查看结果 =====================
print("\n真实 w：", true_w.data.numpy().flatten())
print("学到 w：", w.data.numpy().flatten())
print("真实 b：", true_b.item())
print("学到 b：", b.item())
