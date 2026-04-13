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
    
class AttentionLayer(nn.Module):
    # 单头注意力
    def __init__(self, d_model):
        super().__init__()
        self.hidden_dim = d_model
        self.w_q = torch.randn(d_model, self.hidden_dim)
        self.w_k = torch.randn(d_model, self.hidden_dim)
        self.w_v = torch.randn(d_model, self.hidden_dim)

    def forward(self, x: torch.tensor):
        assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
        batchs, seq_len, d_model = x.shape

        result = []
        for batch in range(batchs):
            x_batch = x[batch]
            Q = x_batch @ self.w_q
            K = x_batch @ self.w_k
            V = x_batch @ self.w_v

            attention_score = torch.softmax( ((Q @ K.T) / math.sqrt(self.hidden_dim)), dim=1)
            attention_value = attention_score @ V
            result.append(attention_value)
        
        # 接下来怎么把结果按照batch拼接起来
        return torch.stack(result, dim=0)

class FeedForwardLayer(nn.Module):
    def __init__():
        super().__init__()


if __name__ == "__main__":
    x = torch.randn(1, 3, 12)
    batch, seq_len, d_model = x.shape
    embedding = PositionEmbeddingLayer(x.shape[1], x.shape[2])
    y = embedding(x)
    print(x.shape)
    print(y.shape)
    
    attentionlayer = AttentionLayer(x.shape[1], x.shape[2])
    z = attentionlayer(y)
    print(z.shape)