import torch
import torch.nn as nn
import torch.nn.functional as F
from config import n_embed, block_size, n_heads, n_layers, dropout


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1)
        wei = wei * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))

        wei = self.dropout(F.softmax(wei, dim=-1))

        v = self.value(x)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embed // n_heads

        self.sa = MultiHeadAttention(n_heads, head_size)

        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)

        self.position_embeddings = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(n_embed)

        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token = self.token_embeddings(idx)

        pos = self.position_embeddings(torch.arange(T, device=idx.device))

        x = token + pos

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.head(x)

        if targets is None:
            return logits

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
