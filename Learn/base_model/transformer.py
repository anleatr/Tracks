import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self, 
            seq_len_en, 
            seq_len_de, 
            d_model, 
            nums_head=4,
            nums_block=6
        ):
        super().__init__()

        self.nums_head =nums_head
        self.nums_block = nums_block

        self.encoder = Encoder(seq_len=seq_len_en, d_model=d_model, 
                               num_heads=self.nums_head, num_block=self.nums_block)
        
        self.decoder = Decoder(seq_len=seq_len_de, d_model=d_model,
                               nums_block=self.nums_block)
        
    def forward(self, source, target):
        memory = self.encoder(source)
        ouput = self.decoder(memory, target)

        # 最后做一层分类头
        return ouput

model = Transformer(3, 5, 512)

source = torch.randn(5, 3, 512)
target = torch.randn(5, 5, 512)

y = model(source, target)
print(target.shape)
print(y.shape)