import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

class ModelConfig:
    vocab_size = 1000
    sequence_len = 128
    hidden_size = 256
    batch_size = 128
    device = 'cuda'

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Initialize parameters with small random values
        
        Returns:
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        """
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        batch_size = config.batch_size
        device = config.device

        Wf = torch.randn(hidden_size, vocab_size + hidden_size, device=device)*0.01 
        bf = torch.ones((hidden_size, 1), requires_grad=True, device=device)
        Wu = torch.randn(hidden_size, vocab_size + hidden_size, device=device)*0.01 
        bu = torch.zeros((hidden_size, 1), requires_grad=True, device=device)
        Wcc = torch.randn(hidden_size, vocab_size + hidden_size, device=device)*0.01 
        bcc = torch.zeros((hidden_size, 1), requires_grad=True, device=device)
        Wo = torch.randn(hidden_size, vocab_size + hidden_size, device=device)*0.01
        bo = torch.zeros((hidden_size, 1), requires_grad=True, device=device)
        Wy = torch.randn(vocab_size, hidden_size, device=device)*0.01
        by = torch.zeros((vocab_size, 1), requires_grad=True, device=device)
        
        Wf.requires_grad = True
        Wu.requires_grad = True
        Wcc.requires_grad = True
        Wo.requires_grad = True
        Wy.requires_grad = True

        self.a_prev = torch.randn((hidden_size,batch_size), device=config.device)
        self.c_prev = torch.randn((hidden_size,batch_size), device=config.device)
        
        self.params = [Wf, bf, Wu, bu, Wcc, bcc, Wo, bo, Wy, by]
        self.device = config.device
        self.hidden_size = config.hidden_size

    def parameters(self):
        return self.params

    def lstm_step_forward(self, x, a_prev, c_prev):
        # x [H,B]
        Wf, bf, Wu, bu, Wcc, bcc, Wo, bo, Wy, by = self.params
        concat = torch.cat((a_prev, x),axis=0)

        f = torch.sigmoid(Wf @ concat + bf) # forget gate
        u = torch.sigmoid(Wu @ concat + bu) # update gate
        cc = torch.tanh(Wcc @ concat + bcc) # candidate
        o = torch.sigmoid(Wo @ concat + bo) # output gate

        c = f*c_prev + u*cc # cell state
        a = o*torch.tanh(c) # hidden state

        y = Wy @ a + by # [H,B]
        return y, a, c


    def lstm_forward(self, batch_X, a_prev, c_prev):

        logits = [] # [L,H,B]

        for t in range(batch_X.shape[0]):
            y, a_prev, c_prev = self.lstm_step_forward(batch_X[t], a_prev, c_prev)
            logits.append(y)

        logits = torch.stack(logits, dim=0)

        return logits, a_prev, c_prev

    def forward(self, batch_X, batch_Y):

        batch_X = batch_X.permute(1,2,0) # [B,L,H] -> [L,H,B]
        logits, self.a_prev, self.c_prev =  self.lstm_forward(batch_X, self.a_prev.detach(),
                                                             self.c_prev.detach())
        logits = logits.permute(2,0,1)  # [L,H,B] - > [B,L,H] 
        logits = logits.contiguous().view(-1, logits.shape[-1])
        loss = F.cross_entropy(logits, batch_Y.view(-1))

        return logits, loss

    def generate_output(self, sample, dataset, temperature=1, top_k=None, steps = 1000):
        '''
        Generate n samples characters given x prompt
        '''
        import numpy as np
        self.eval()
        device = self.device
        vocab_size = dataset.vocab_size
        a_prev = torch.randn((self.hidden_size,1), device=device)
        c_prev = torch.randn((self.hidden_size,1), device=device)

        idx = [dataset.ch2i[k] for k in sample]
        x = dataset.vectorization(idx)
        x = torch.tensor(x).to('cuda').unsqueeze(-1).float()
       

        # prime the network with input
        for item in x:
            logits, a_prev, c_prev = self.lstm_step_forward(item, a_prev, c_prev)

        for i in range(steps - len(x)):
            logits = logits / temperature
            v, ix = logits.topk(k=top_k, dim=0)
            logits[logits < v[-1, :]] = -float('inf')
            probs = torch.softmax(logits, dim=0).view(-1)

            next_index = torch.multinomial(probs, num_samples=1).item()

            x = torch.zeros((vocab_size,1)).to(device)
            x[next_index,0]=1
            idx += [next_index]
            logits, a_prev, c_prev = self.lstm_step_forward(x, a_prev, c_prev)
        out = ''.join([dataset.i2ch[i] for i in idx])
        
        return out