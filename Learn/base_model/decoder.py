import torchvision
import torch
import torch.nn as nn
import math

from encoder import PositionEmbeddingLayer, FeedForwardLayer

# 输入: [batch, seq_len ,d_model]

class MaskMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nums_head=4):
        super().__init__()
        assert d_model % nums_head ==0, f"got d_model{d_model}, nums_head{nums_head}, but should be exact division"

        self.nums_head = nums_head
        self.hidden_dim = d_model // self.nums_head

        # 如果是单头，每个应该是[d_model ,d_model], 如果是多头， [d_model , d_model // nums_head], nums_head, 3
        self.qkv_proj = nn.Linear(d_model, d_model*3)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len , d_model = x.shape
        # 首先生成QKV
        qkv = self.qkv_proj(x) # [batch, seq_len, d_model * 3]
        qkv = qkv.reshape(batch, seq_len, self.nums_head, 3, self.hidden_dim)
        qkv = qkv.permute(3, 0, 2, 1, 4) # [num, batch, nums_head, seq_len , hidden_dim]
        q, k , v = qkv[0], qkv[1], qkv[2] #  [batch, nums_head, seq_len , hidden_dim]

        # 计算注意力分数
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim) # [batch, nums_head, seq_len ,seq_len]

        # 做掩码， 生成一个下三角矩阵
        # mask = torch.ones_like(score).tril()
        # mask[torch.where(mask==0)] = - torch.inf
        mask = torch.ones(size=(1, 1, seq_len, seq_len), device = x.device)
        mask = mask.masked_fill(mask==0, float('-inf'))
        
        # 计算注意力值
        attn = torch.softmax(score + mask, dim=-1) @ v # [batch, nums_head, seq_len , hidden_dim]

        # 把各头堆叠起来
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        return self.out_proj(attn)

class CrossAttention(nn.Module):
    def __init__(self, d_model, nums_head=4):
        super().__init__()
        assert d_model % nums_head ==0, f"got d_model{d_model}, nums_head{nums_head}, but should be exact division"
        
        self.nums_head = nums_head
        self.hidden_dim = d_model // nums_head

        self.q_proj = nn.Linear(d_model ,d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, encoder_x, decoder_x):
        # 唯一的区别是Q是decoder自己的，k和v是encoder的
        # assert encoder_x.shape == decoder_x.shape, "encoder and decode shape should be same"
        # encoder和decoer的序列长度可以不一样

        batch, seq_len_decoder , d_model = decoder_x.shape
        _, seq_len_encoder , _ = encoder_x.shape

        q = self.q_proj(decoder_x).reshape(batch, seq_len_decoder, self.nums_head, -1) # [batch, seq_len_de, nums_head, hidden_dim]  
        kv = self.kv_proj(encoder_x).reshape(batch, seq_len_encoder, self.nums_head, 2, -1)
        kv = kv.permute(3, 0, 2, 1, 4) 

        q = q.transpose(1, 2) # [batch, nums_head, seq_len_de, hidden_dim]
        k, v = kv[0], kv[1] # [batch, nums_head, seq_len_en, hidden_dim]

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim) # [batch, nums_head, seq_len_decoder ,seq_len_encoder]

        attn = torch.softmax(score, dim=-1) @ v # [batch, nums_head, seq_len_decoder, hidden_dim]

        # 把各头堆叠起来
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len_decoder, -1)

        return self.out_proj(attn)

class DecoderBlock(nn.Module):
    def __init__(self, seq_len, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.seq_len = seq_len
        self.d_model = d_model

       
        self.mask_attn = MaskMultiHeadAttention(d_model=self.d_model)
        self.cross_attn = CrossAttention(d_model=self.d_model)
        self.fc = FeedForwardLayer(d_model=self.d_model)
   
        self.ln1 = nn.LayerNorm( d_model)
        self.ln2 = nn.LayerNorm( d_model)
        self.ln3 = nn.LayerNorm( d_model)

    def forward(self, encoder_x, x):
       
        x1 = self.mask_attn(x)
        x1 = self.ln1(x1)
        x1 = x1 + x

        x2 = self.cross_attn(encoder_x, x1)
        x2 = self.ln2(x2)
        x2 = x2 + x1

        x3 = self.fc(x2)
        x3 = self.ln3(x3)
        
        return x3 + x2

class Decoder(nn.Module):
    def __init__(self, seq_len, d_model, nums_block=6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.nums_block = nums_block

        self.pe = PositionEmbeddingLayer(seq_len=self.seq_len, d_model=self.d_model)

        self.model = nn.ModuleList([
            DecoderBlock(seq_len=self.seq_len, d_model=self.d_model) 
            for _ in range(self.nums_block)
        ])

    def forward(self, encoder_x, x):
        x = self.pe(x)

        for layer in self.model:
            x = layer(encoder_x, x)
        
        return x

if __name__ == "__main__":
    source = torch.randn(5, 3, 512)
    target = torch.randn(5, 5, 512)
    model = Decoder(5, 512)
    y = model(source, target)
    print(target.shape)
    print(y.shape)