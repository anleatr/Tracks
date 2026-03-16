import torch
import math

def position_embedding(x: torch.tensor):
    """
    x: torch.tensor [batch, seq_length, dim]
    """
    assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
    
    position_matrix = torch.zeros_like(x[0])

    batch, seq_len, dim = x.shape

    for pos in range(seq_len):
        for i in range(0, dim, 2):
            position_matrix[pos][i] = torch.sin(torch.tensor(pos / math.pow(10000, i / dim)))
            position_matrix[pos][i+1] = torch.cos(torch.tensor(pos / math.pow(10000, i / dim)))

    # expand batch
    position_matrix = position_matrix.unsqueeze(0).expand(batch, -1, -1)

    return x + position_matrix

if __name__ == "__main__":
    x = torch.randn(1, 3, 28)
    print(x)
    print(x.shape)
    x_with_position = position_embedding(x)
    print(x_with_position)
    print(x_with_position.shape)

    print(x_with_position - x)