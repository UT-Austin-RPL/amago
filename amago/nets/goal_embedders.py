import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import gin


@gin.configurable(denylist=["goal_dim"])
class FFGoalEmb(nn.Module):
    def __init__(
        self,
        goal_length: int,
        goal_dim: int,
        goal_emb_dim: int = 32,
        zero_embedding: bool = False,
    ):
        super().__init__()
        self.ff1 = nn.Linear(goal_length * goal_dim, 2 * goal_dim)
        self.ff2 = nn.Linear(2 * goal_dim, goal_emb_dim)
        self.goal_norm = nn.LayerNorm(goal_emb_dim)
        self.goal_emb_dim = goal_emb_dim
        self.zero_embedding = zero_embedding

    def forward(self, goals):
        B, L, GL, GD = goals.shape
        if self.zero_embedding:
            return torch.zeros(
                (B, L, self.goal_emb_dim), dtype=torch.float32, device=goals.device
            )
        goals = rearrange(goals, "b l gl gd -> b l (gl gd)")
        goal_emb = F.leaky_relu(self.ff1(goals))
        goal_emb = self.goal_norm(self.ff2(goal_emb))
        return goal_emb


@gin.configurable(denylist=["goal_dim"])
class TokenGoalEmb(nn.Module):
    def __init__(
        self,
        goal_length: int,
        goal_dim: int,
        min_token: int = -128,
        max_token: int = 128,
        goal_emb_dim: int = 32,
        embedding_dim: int = 16,
        hidden_size: int = 64,
        zero_embedding: bool = False,
    ):
        super().__init__()
        assert goal_emb_dim > 1 or zero_embedding
        self._min_token = min_token
        self._max_token = max_token - min_token + 1
        self.goal_emb = nn.Embedding(
            num_embeddings=self._max_token, embedding_dim=embedding_dim
        )
        self.goal_rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.goal_ff = nn.Linear(hidden_size, goal_emb_dim)
        self.goal_norm = nn.LayerNorm(goal_emb_dim)
        self.goal_emb_dim = goal_emb_dim
        self.zero_embedding = zero_embedding

    def forward(self, goals):
        B, L, GL, GD = goals.shape
        if self.zero_embedding:
            return torch.zeros(
                (B, L, self.goal_emb_dim), dtype=torch.float32, device=goals.device
            )

        goals = (goals - self._min_token).clamp(0.0, self._max_token)
        goals = rearrange(goals, "b l gl gd -> (b l) (gl gd)").long()
        goal_emb = self.goal_emb(goals)
        _, goal_emb = self.goal_rnn(goal_emb)
        goal_emb = self.goal_norm(self.goal_ff(goal_emb.squeeze(0)))
        goal_emb = rearrange(goal_emb, "(b l) gd -> b l gd", b=B, l=L)
        return goal_emb
