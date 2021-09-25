import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
import math

class TansformerConfig:
    vocab_size = 1000
    sequence_len = 128
    n_block = 8
    n_head = 8
    embed_dim = 100
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embed_pdrop = 0.1
    causal = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
#         embed_dim embed_dim
        assert config.embed_dim % config.n_head == 0
        embed_dim = config.embed_dim
        

        # self attention is followed by layernorm so bias is not required
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        # regularization in the form of dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # concatenate n_head into one output and project
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.sequence_len, config.sequence_len))
                                     .view(1, 1, config.sequence_len, config.sequence_len))
        self.n_head = config.n_head
        self.causal = config.causal

    def forward(self, q, k, v, pad_mask=None, causal=None):
        '''
        split input dimension into n_head for query, key, value. calculate (q.Tk/sqrt(dk)).v

        '''

        batch_size, sequence_len, embed_dim = v.shape 
        n_head = self.n_head
        # last dimension after splitting
        dk = embed_dim//n_head
        
        q = self.query(q).view(batch_size, sequence_len, n_head, dk).transpose(1,2) # B,nh,T,dk
        k = self.key(k).view(batch_size, sequence_len, n_head, dk).transpose(1,2)
        v = self.value(v).view(batch_size, sequence_len, n_head, dk).transpose(1,2)
        
        
        # scale q and k
        
        att = (q @ k.transpose(-2,-1))/(dk**0.5) # B,nh,T,dk x B,nh,dk,T -> B,nh,T,T
        if pad_mask is not None:
            # print(pad_mask.shape)
            att = att.masked_fill(pad_mask==0, float('-1e10'))
        if causal is not None:
            att = att.masked_fill(self.causal_mask == 0, float('-1e10'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # B,nh,T,T x B,Nh,T,dk -> B,nh,T,dk
        # swap n_head back to last dimension then re-assemble side by side to embed_dim
        y = y.transpose(1,2).contiguous().view(batch_size, sequence_len, n_head*dk) 
        # output projection
        y = self.resid_drop(self.proj(y)) # output is batch_size, sequence_len, embed_dim
        
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim, n_head = config.embed_dim, config.n_head
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(config)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
                            nn.Linear(embed_dim,4*embed_dim),
                            nn.GELU(),
                            nn.Linear(4*embed_dim,embed_dim),
                            nn.Dropout(config.resid_pdrop),
                            )
        
    def forward(self, x, pad_mask=None, causal=None):
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, pad_mask, causal)
        x_norm = self.layernorm2(x)
        x = x + self.feedforward(x_norm)
        
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Linear(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.embed_pdrop)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_len, config.embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(config) for i in range(config.n_block)])
        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.embed_dim = config.embed_dim
        
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.device = config.device
        self.causal = config.causal


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, idx, targets=None):
        batch_size, sequence_len, vocab_size = idx.shape 
        embed_dim = self.embed_dim
        idx = idx.float()

        # pad_mask_x = (idx>0).view(self.batch_size, 1, 1, self.sequence_len) # 0 is padding idx
        embed = self.embed(idx) # each token map to learnable vector
        position_embedding = self.position_embedding[:,:sequence_len,:]

        x = self.drop(embed + position_embedding)
        for block in self.blocks:
            x = block(x, causal=True)

        x = self.layernorm(x)
        
        logits = self.linear(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    def generate_output(self, sample, dataset, temperature=1, top_k=None, steps = 1000):
        '''
        Generate n samples characters given x prompt
        '''
        import numpy as np

        self.eval()
        with torch.no_grad():
            
            device = self.device
            vocab_size = dataset.vocab_size

            idx = [dataset.ch2i[k] for k in sample]
            if len(idx) > dataset.sequence_len:
                idx = idx[:sequence_len]
            else:
                idx = [0]*(dataset.sequence_len-len(idx)) + idx


            x = dataset.vectorization(idx)
            x = torch.tensor(x).to('cuda').unsqueeze(0).float()
           
            for i in range(steps - len(x)):
                logits, loss = self.forward(x)
                logits = logits[:,-1,:]
                logits = logits / temperature
                v, ix = logits.topk(k=top_k, dim=-1)
                logits[logits < v[:,-1]] = -float('inf')
                probs = torch.softmax(logits, dim=-1).view(-1)

                next_index = torch.multinomial(probs, num_samples=1).item()

                idx += [next_index]

                char = torch.zeros((1, 1, vocab_size)).to(device)
                char[0,0,next_index]=1
                x = torch.cat((x, char), dim=1)
                x = x[:,1:,:]
                
            out = ''.join([dataset.i2ch[i] for i in idx])
        
        self.train()
        
        return out

