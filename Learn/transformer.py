import torch

def position_embedding(x: torch.tensor):
    """
    x: torch.tensor [batch, seq_length, dim]
    """
    assert x.dim() == 3, f"input length should be 3 but get{x.dim()}"
    
    position_matrix = torch.zeros_like(x[0])

    batch, seq_len, dim = x.shape

    for pos in range(seq_len):
        for i in range(0, dim, 2):
            position_matrix[pos][i] = torch.sin(pos / torch.pow(10000, i / dim))
            position_matrix[pos][i+1] = torch.cos(pos / torch.pow(10000, i / dim)) 

    # expand batch
    position_matrix = position_matrix.unsqueeze(0).expand(batch, -1, -1)

    return x + position_matrix