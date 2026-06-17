import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 设置中文字体（避免中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

def compute_normalized_laplacian(adj_matrix):
    """
    计算 对称归一化图拉普拉斯 L = I - D^(-1/2) A D^(-1/2)
    :param adj_matrix: 邻接矩阵 (n x n)
    :return: L, 度矩阵D
    """
    n = adj_matrix.shape[0]
    # 计算每个节点的度
    degree = np.sum(adj_matrix, axis=1)
    # 构造度矩阵 D
    D = np.diag(degree)
    # D^(-1/2)，防止除0
    D_sqrt_inv = np.zeros_like(D, dtype=float)
    for i in range(n):
        if degree[i] > 1e-8:
            D_sqrt_inv[i, i] = 1.0 / np.sqrt(degree[i])
    # 对称归一化拉普拉斯
    L = np.eye(n) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    return L, D

def laplacian_embedding(L, dim=2):
    """
    拉普拉斯嵌入：对拉普拉斯矩阵谱分解，取前 dim 个非零特征值对应的特征向量
    :param L: 归一化拉普拉斯矩阵
    :param dim: 嵌入维度（默认2维，方便可视化）
    :return: embedding (n x dim)
    """
    # 特征值、特征向量 (eigh 专用于实对称矩阵)
    eigvals, eigvecs = eigh(L)
    # 排序：特征值从小到大，跳过接近0的零特征值（连通分量）
    eps = 1e-8
    valid_idx = np.where(eigvals > eps)[0]
    # 取前 dim 个特征向量作为嵌入
    selected_vecs = eigvecs[:, valid_idx[:dim]]
    return selected_vecs

def plot_graph_and_embedding(G, embedding, title1="原始图拓扑", title2="2维拉普拉斯嵌入"):
    """可视化原始图 & 拉普拉斯嵌入结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 绘制原始图拓扑
    pos = nx.spring_layout(G, seed=42)  # 固定布局
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color="lightblue", 
            node_size=600, font_size=12)
    ax1.set_title(title1, fontsize=14)
    
    # 2. 绘制2维拉普拉斯嵌入空间
    ax2.scatter(embedding[:, 0], embedding[:, 1], s=200, c="orange")
    for i in range(embedding.shape[0]):
        ax2.annotate(str(i), (embedding[i, 0], embedding[i, 1]), 
                     fontsize=12, ha="center", va="center")
    ax2.set_title(title2, fontsize=14)
    ax2.set_xlabel("维度 1")
    ax2.set_ylabel("维度 2")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ===================== 案例1：简单路径图（基础演示） =====================
print("=" * 50)
print("案例1：简单路径图 1-2-3-4-5")
print("=" * 50)

# 1. 建图
G1 = nx.path_graph(5)  # 节点0-1-2-3-4 连成一条链
adj1 = nx.to_numpy_array(G1)
print("邻接矩阵：")
print(adj1)

# 2. 计算归一化拉普拉斯 & 嵌入
L1, D1 = compute_normalized_laplacian(adj1)
emb1 = laplacian_embedding(L1, dim=2)
print("\n2维拉普拉斯嵌入矩阵（每行对应一个节点）：")
print(emb1)

# 3. 可视化
plot_graph_and_embedding(G1, emb1, 
                         title1="路径图: 0-1-2-3-4",
                         title2="路径图 2维拉普拉斯嵌入")

# ===================== 案例2：LLaGA 模板树（核心场景） =====================
print("\n" + "=" * 50)
print("案例2：LLaGA 固定结构模板树（两跳、固定采样，含pad节点）")
print("=" * 50)

# 模拟 LLaGA 邻域细节模板：根节点0，一阶邻居 1,2,3；二阶邻居用pad(4,5,6,7,8,9)补齐
# 构造固定拓扑的模板树（对应论文里固定形状计算树）
G2 = nx.Graph()
# 节点：0(根),1,2,3(一跳),4,5,6,7,8,9(二跳/pad)
nodes = list(range(10))
G2.add_nodes_from(nodes)
# 连边：根节点连接三个一跳节点
edges = [(0,1), (0,2), (0,3)]
# 一跳节点各自连二跳/pad节点（模拟固定采样+pad补齐）
edges += [(1,4), (1,5), (1,6)]
edges += [(2,7), (2,8), (2,9)]
G2.add_edges_from(edges)

adj2 = nx.to_numpy_array(G2)
print("LLaGA模板树 邻接矩阵形状:", adj2.shape)

# 计算拉普拉斯 + 嵌入
L2, D2 = compute_normalized_laplacian(adj2)
emb2 = laplacian_embedding(L2, dim=2)
print("\nLLaGA模板树 2维拉普拉斯嵌入（每行=模板中一个位置的编码）：")
print(emb2)

# 可视化 LLaGA 模板树及其嵌入
plot_graph_and_embedding(G2, emb2,
                         title1="LLaGA 固定结构模板树",
                         title2="模板树 2维拉普拉斯嵌入 (位置编码)")

# ===================== 补充：关键结论打印 =====================
print("\n" + "=" * 50)
print("核心现象总结：")
print("1. 原始图中相邻节点，在拉普拉斯嵌入空间坐标距离更近")
print("2. LLaGA 使用【固定拓扑模板树】，因此拉普拉斯嵌入只需计算一次，全局复用")
print("3. pad节点和普通节点共享同一套结构嵌入，保证维度与结构位置对齐")
print("=" * 50)
