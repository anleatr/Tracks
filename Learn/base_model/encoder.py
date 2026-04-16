import torch
import torch.nn as nn
import math

class PositionEmbeddingLayer(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        self.position_matrix = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.position_matrix[pos][i] += torch.sin(torch.tensor(pos / math.pow(10000, i / d_model))) # 偶数
                self.position_matrix[pos][i+1] += torch.cos(torch.tensor(pos / math.pow(10000, i / d_model))) # 奇数

        self.register_buffer("pe", self.position_matrix)
        
    def forward(self, x: torch.tensor):
        """
        x: torch.tensor [batch, seq_length, dim]
        """
        assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
        
        batch = x.shape[0]

        # expand batch
        position_matrix = self.pe.unsqueeze(0).expand(batch, -1, -1)
        return x + position_matrix
    
class MultiHeadAttentionLayer(nn.Module):
    # 多头注意力
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, f"got d_model{d_model} and self.num_heads{num_heads}, can't exact division"

        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = d_model // self.num_heads
 
        # self.w_q = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]
        # self.w_k = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]
        # self.w_v = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]
        self.qkv_proj = nn.Linear(d_model, d_model * 3) # 同时存储qkv权重，后续分割成[d_model , d_model]
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x: torch.tensor):
        assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
        batchs, seq_len, d_model = x.shape
        
        # 怎么用向量化操作加速计算 x: batchs, seq_len ,d_model ,w:d_model, hidden_dim
        qkv = self.qkv_proj(x) # [batch ,seq_len , d_model * 3]
        qkv = qkv.reshape(batchs, seq_len, 3 ,self.num_heads, self.hidden_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # 为什么把batchs放在第二位？
        q, k, v = qkv[0], qkv[1], qkv[2] # [batchs ,self.num_heads, seq_len , self.hidden_dim]

        # 计算注意力, 这样我们就实现了多个batch和head的向量化计算
        attention_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim) # [batchs ,num_heads, seq_len ,seq_len]
        attn = torch.softmax(attention_score, dim=-1)

        # 最终的注意力值
        attn_value = attn @ v #[batch, num_heads, seq_len ,self.hidden_dim]

        # 接下来再把所有head的值拼接起来
        attn_value = attn_value.transpose(1, 2).reshape(batchs, seq_len, -1) # [batch ,nums_heads, d_model]

        return self.out_proj(attn_value)
        # result = [] 
        # for batch in range(batchs):
        #     x_batch = x[batch]
        #     attention_value_list = []
        #     for head in range(self.num_heads):  # 计算每个头
        #         Q = x_batch @ self.w_q[head]
        #         K = x_batch @ self.w_k[head]
        #         V = x_batch @ self.w_v[head]
        #         # 如果是多头，在哪里把每个头融合起来 ？
        #         attention_score = torch.softmax( ((Q @ K.T) / math.sqrt(self.hidden_dim)), dim=1)
        #         attention_value_per_head = attention_score @ V
        #         attention_value_list.append(attention_value_per_head)
            
        #     # concat
        #     attention_value = torch.cat(attention_value_list, dim=1)
        #     result.append(attention_value)
        
        # # print(f"d_model: {self.d_model}, num_heads: {self.num_heads}, dim_per_head:{attention_value_per_head.shape[1]}")
        # # 接下来怎么把结果按照batch拼接起来
        # return torch.stack(result, dim=0)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=2048):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x:torch.tensor):
        return self.model(x)

class LayerNormalization(nn.Module):
    def __init__():
        super().__init__()

    def forward(self,x: torch.tensor):
        """
        (batch, seq_len, d_model), 如何进行层归一化? -- 在d_model维度
        """

class EncoderBlock(nn.Module):
    def __init__(self, test_data, num_heads=4, num_block=6):
        super().__init__()
        # 输入x, 输出相同shape的x
        self.num_block = num_block
        batch, seq_len, d_model = test_data.shape

        self.attention = MultiHeadAttentionLayer(d_model, num_heads=num_heads)
        self.feedforward = FeedForwardLayer(d_model)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x1 = self.attention(x)
        x1 = x1 + x
        x1 = self.ln(x1)

        x2 = self.feedforward(x1)
        x2 = x2 + x1
        x2 = self.ln(x2)

        return x2
        
class Encoder(nn.Module):
    def __init__(self, test_data, num_heads=4, num_block=6):
        super().__init__()

        self.num_block = num_block
        batch, seq_len, d_model = test_data.shape

        self.pos_emb = PositionEmbeddingLayer(seq_len, d_model)

        # 不能用列表，否则追踪不到这些层
        self.model = nn.ModuleList([EncoderBlock(test_data, num_heads=num_heads, num_block=num_block) for _ in range(num_block)])
    def forward(self, x:torch.tensor):
        x = self.pos_emb(x)

        for layer in self.model:
            x = layer(x)
        return x

if __name__ == "__main__":
    device = "cuda"

    x = torch.randn(1, 3, 512)
    encoder = Encoder(test_data=x, num_block=6)

    # x = x.to(device)
    # encoder = encoder.to(device)

    # import time
    # start = time.time()
    # for _ in range(1000):
    #     y = encoder(x)
    # print(f"cpu use:{time.time() - start}") # 4.0
    # print(f"gpu use:{time.time() - start}") # 0.945 0.91
    # print(x.shape)
    # print(y.shape)
    # encoder = Encoder(test_data=x, num_block=6).to(torch.device("cuda"))
    # y = encoder(x)
    # print(f"gpu use:{time.time() - start}") 
    # 第一个问题，使用nn.Parameter, 这样才会随模型移动，并且model.parameter和stact_dict{}才会包含这个参数，
    # 第二个问题，向量化操作
    # 第三个问题，position_embedding的参数不需要梯度，但是也需要移动到gpu上，

    print(dir(encoder))