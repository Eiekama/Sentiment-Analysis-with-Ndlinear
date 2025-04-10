"""
Adapted from Andrej Karpathy's implementation of a minimal GPT network at https://github.com/karpathy/minGPT
"""

import math, time

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    """

    def __init__(
            self,
            n_embd: int,
            n_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        # key, query, value projections
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.out_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):

        q, k ,v  = self.qkv_proj(x).split(self.n_embd, dim=-1)

        # b = batch size, t = sequence length, h = number of heads, d = n_embd / number of heads
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)

        att = einsum(q, k, 'b h q d, b h k d -> b h q k') / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = einsum(att, v, 'b h q t, b h t d -> b h q d')
        y = rearrange(y, 'b h q d -> b q (h d)')

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(
            self,
            n_embd: int,
            n_query_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
        ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_query_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        return x, end_time-start_time, end_memory-start_memory

class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(
            self,
            vocab_size: int,
            block_size: int, # max sequence length
            n_embd: int,
            output_dim: int,
            n_head=4,
            n_layer=6,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        ):
        super().__init__()
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # token embedding
            wpe = nn.Embedding(block_size, n_embd), # positional embedding
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(block_size*n_embd, output_dim, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(
            self,
            weight_decay=.1,
            learning_rate=.001,
            betas: tuple[float,float] = (.9, .999),
        ):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx, targets=None):
        block_times = []
        block_mem_consumed = []
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, time, mem = block(x)
            block_times.append(time)
            block_mem_consumed.append(mem)
        x = self.transformer.ln_f(x)
        x = rearrange(x, "b t e -> b (t e)")
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return logits, loss, sum(block_times)/len(block_times), sum(block_mem_consumed)/len(block_mem_consumed)
