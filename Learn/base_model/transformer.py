import torch
import torch.nn as nn
import math

class PositionEmbeddingLayer(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        self.position_matrix = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.position_matrix[pos][i] = torch.sin(torch.tensor(pos / math.pow(10000, i / d_model)))
                self.position_matrix[pos][i+1] = torch.cos(torch.tensor(pos / math.pow(10000, i / d_model)))

    def forward(self, x: torch.tensor):
        """
        x: torch.tensor [batch, seq_length, dim]
        """
        assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
        
        batch = x.shape[0]

        # expand batch
        position_matrix = self.position_matrix.unsqueeze(0).expand(batch, -1, -1)
        return x + position_matrix
    
class MultiHeadAttentionLayer(nn.Module):
    # 多头注意力
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, f"got d_model{d_model} and self.num_heads{num_heads}, can't exact division"

        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = d_model // self.num_heads
 
        self.w_q = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]
        self.w_k = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]
        self.w_v = [torch.randn(d_model, self.hidden_dim) for _ in range(self.num_heads)]

    def forward(self, x: torch.tensor):
        assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
        batchs, seq_len, d_model = x.shape

        result = []
        for batch in range(batchs):
            x_batch = x[batch]
            attention_value_list = []
            for head in range(self.num_heads):  # 计算每个头
                Q = x_batch @ self.w_q[head]
                K = x_batch @ self.w_k[head]
                V = x_batch @ self.w_v[head]
                # 如果是多头，在哪里把每个头融合起来 ？
                attention_score = torch.softmax( ((Q @ K.T) / math.sqrt(self.hidden_dim)), dim=1)
                attention_value_per_head = attention_score @ V
                attention_value_list.append(attention_value_per_head)
            
            # concat
            attention_value = torch.cat(attention_value_list, dim=1)
            result.append(attention_value)
        
        # print(f"d_model: {self.d_model}, num_heads: {self.num_heads}, dim_per_head:{attention_value_per_head.shape[1]}")
        # 接下来怎么把结果按照batch拼接起来
        return torch.stack(result, dim=0)

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
        
class Encoder(nn.Module):
    def __init__(self, test_data, num_heads=4, num_block=6):
        super().__init__()

        self.num_block = num_block
        batch, seq_len, d_model = test_data.shape
        self.pos_emb = PositionEmbeddingLayer(seq_len, d_model)
        self.attention = MultiHeadAttentionLayer(d_model, num_heads=num_heads)
        self.feedforward = FeedForwardLayer(d_model)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x:torch.tensor):
        x = self.pos_emb(x)

        for _ in range(self.num_block):
            x1 = self.attention(x)
            x1 = x1 + x
            x1 = self.ln(x1)

            x2 = self.feedforward(x1)
            x2 = x2 + x1
            x2 = self.ln(x2)

            x = x2
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 512)
    # batch, seq_len, d_model = x.shape
    # embedding = PositionEmbeddingLayer(x.shape[1], x.shape[2])
    # y = embedding(x)
    # print(x.shape)
    # print(y.shape)
    
    # attentionlayer = MultiHeadAttentionLayer(d_model=d_model, num_heads=4)
    # z = attentionlayer(y)
    # print(z.shape)
    encoder = Encoder(test_data=x, num_block=6)
    y = encoder(x)

    print(x.shape)
    print(y.shape)