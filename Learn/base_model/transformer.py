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
    

if __name__ == "__main__":
    x = torch.randn(1, 3, 12)
    batch, seq_len, d_model = x.shape
    embedding = PositionEmbeddingLayer(x.shape[1], x.shape[2])
    y = embedding(x)
    print(x.shape)
    print(y.shape)
    