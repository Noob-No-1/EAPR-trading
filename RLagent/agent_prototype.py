# agent_prototype.py
# ---------------------------------------------
# RL agent prototype with state = ONLY embedding
# Actions: 0=hold, 1=buy, 2=sell (fixed USD amount per step)
# Reward (critic): unrealized_profit + realized_profit
# ---------------------------------------------
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============== Environment ==============
@dataclass
class EnvConfig:
    embeddings_path: str                  # npy: [N, D]
    prices_csv_path: str                  # CSV with 'close' column, aligned per 5-min bar
    trade_amount_usd: float = 5.0         # fixed $ amount to buy/sell each step
    start_cash_usd: float = 1000.0        # initial cash
    reward_mode: str = "level"            # "level" -> unrealized+realized (as requested), "delta" -> change in equity
    seed: int = 42

class TradingEnvEmbeddingOnly:
    """
    Minimal trading environment where the observation is ONLY the embedding vector.
    Internals (not part of state) track: price, cash (spot dollars), position (shares),
    mean_entry_price, realized PnL; unrealized is computed from current price.
    Actions: 0=hold, 1=buy, 2=sell with a fixed USD amount per step.

    Info dict returns:
      - balance (equity = cash + position_value)
      - spot_dollars (cash)
      - unrealized_profit
      - realized_profit
      - mean_entry_price
      - last_entry_price
      - position_shares
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Load data
        self.embeddings = np.load(cfg.embeddings_path)  # [N, D]
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D [N,D], got {self.embeddings.shape}")
        df = pd.read_csv(cfg.prices_csv_path)
        if "close" not in df.columns:
            raise ValueError(f"'close' column not found in {cfg.prices_csv_path}. Columns: {list(df.columns)}")
        self.prices = df["close"].astype(float).to_numpy()

        # Align lengths (safeguard)
        N = min(len(self.embeddings), len(self.prices))
        self.embeddings = self.embeddings[:N]
        self.prices = self.prices[:N]
        self.N = N
        self.D = self.embeddings.shape[1]

        # Episode state
        self.t = 0
        self.reset()

    def reset(self, start_index: Optional[int] = None) -> np.ndarray:
        """Reset environment. Optionally start at a random valid index."""
        if start_index is None:
            # start so that we can at least step once
            self.t = 0
        else:
            self.t = int(max(0, min(self.N - 1, start_index)))

        self.cash = float(self.cfg.start_cash_usd)    # spot dollars
        self.position_shares = 0.0
        self.mean_entry_price = 0.0
        self.last_entry_price = 0.0
        self.realized_profit = 0.0
        self.prev_equity = self._equity(self.t)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        # State is ONLY the embedding vector at time t
        return self.embeddings[self.t].astype(np.float32, copy=False)

    def _equity(self, t_idx: int) -> float:
        price = self.prices[t_idx]
        pos_val = self.position_shares * price
        unreal = self._unrealized_profit(price)
        # Equity = cash + position value (same as cash + unreal + cost basis),
        # but we keep realized separately per spec.
        return self.cash + pos_val

    def _unrealized_profit(self, price: float) -> float:
        if self.position_shares <= 0.0:
            return 0.0
        return (price - self.mean_entry_price) * self.position_shares

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: 0=hold, 1=buy, 2=sell
        Always uses fixed USD size self.cfg.trade_amount_usd.
        If trying to sell more than current position value, sell everything.
        """
        if self.t >= self.N - 1:
            # No more steps possible
            obs = self._get_obs()
            reward = 0.0
            done = True
            return obs, reward, done, self._info()

        price = float(self.prices[self.t])
        trade_usd = float(self.cfg.trade_amount_usd)

        # Execute action
        if action == 1:  # BUY
            if self.cash > 0.0:
                spend = min(trade_usd, self.cash)
                if spend > 0.0:
                    shares = spend / price
                    # Update mean entry price (weighted average)
                    total_shares = self.position_shares + shares
                    if total_shares > 0:
                        self.mean_entry_price = (
                            (self.mean_entry_price * self.position_shares) + (price * shares)
                        ) / total_shares
                        self.last_entry_price = price
                    self.position_shares = total_shares
                    self.cash -= spend

        elif action == 2:  # SELL
            position_value = self.position_shares * price
            if position_value > 0.0:
                sell_value = min(trade_usd, position_value)
                shares_to_sell = sell_value / price
                shares_to_sell = min(shares_to_sell, self.position_shares)  # safeguard

                # Realized PnL from selling shares
                realized_this = (price - self.mean_entry_price) * shares_to_sell
                self.realized_profit += realized_this

                self.position_shares -= shares_to_sell
                self.cash += sell_value

                # If we closed the position entirely, reset mean entry price
                if self.position_shares <= 1e-12:
                    self.position_shares = 0.0
                    self.mean_entry_price = 0.0
                    self.last_entry_price = 0.0

        # Compute reward per spec
        price_now = float(self.prices[self.t])
        unreal = self._unrealized_profit(price_now)
        equity_now = self._equity(self.t)

        if self.cfg.reward_mode == "level":
            reward = unreal + self.realized_profit
        elif self.cfg.reward_mode == "delta":
            # More typical in RL: reward is change in equity
            reward = equity_now - self.prev_equity
        else:
            raise ValueError("reward_mode must be 'level' or 'delta'")

        self.prev_equity = equity_now

        # Advance time
        self.t += 1
        done = (self.t >= self.N - 1)

        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), done, info

    def _info(self) -> Dict[str, Any]:
        price = float(self.prices[self.t])
        position_value = self.position_shares * price
        unreal = self._unrealized_profit(price)
        balance = self.cash + position_value  # equity
        return {
            "balance": float(balance),
            "spot_dollars": float(self.cash),
            "unrealized_profit": float(unreal),
            "realized_profit": float(self.realized_profit),
            "mean_entry_price": float(self.mean_entry_price),
            "last_entry_price": float(self.last_entry_price),
            "position_shares": float(self.position_shares),
            "price": float(price),
            "t": int(self.t),
        }


# ============== Actor-Critic (Policy Learning) ==============
class PolicyNet(nn.Module):
    """Simple MLP policy for discrete actions {hold, buy, sell}."""
    def __init__(self, input_dim: int, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits
        return self.net(x)


class ValueNet(nn.Module):
    """Simple MLP critic estimating V(s)."""
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    max_episodes: int = 50
    device: str = "cpu"
    seed: int = 123

class ActorCriticAgent:
    def __init__(self, obs_dim: int, cfg: TrainConfig):
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.policy = PolicyNet(obs_dim).to(cfg.device)
        self.value = ValueNet(obs_dim).to(cfg.device)
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.value.parameters(), lr=cfg.lr_critic)
        self.cfg = cfg

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        x = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        logits = self.policy(x)           # [1, 3]
        dist = Categorical(logits=logits)
        action = dist.sample()            # [1]
        logp = dist.log_prob(action)      # [1]
        value = self.value(x)             # [1]
        return int(action.item()), logp.squeeze(0), value.squeeze(0)

    def update(self, logps, values, rewards, dones):
        """
        A2C-style update with Monte Carlo returns (episodic) for simplicity.
        """
        # Convert lists to tensors
        logps_t = torch.stack(logps)         # [T]
        values_t = torch.stack(values)       # [T]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.cfg.device)  # [T]

        # Compute returns (G_t) backward
        returns = []
        G = 0.0
        for r in reversed(rewards_t.tolist()):
            G = r + self.cfg.gamma * G
            returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=self.cfg.device)

        # Advantages
        advantages = returns - values_t.detach()

        # Losses
        actor_loss = -(logps_t * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(values_t, returns)

        # Entropy bonus (for exploration)
        # We approximate by re-sampling dist from stored logits; but we didn't store logits.
        # For simplicity, skip exact entropy; add small L2 to actor instead or ignore.
        # Here we add a tiny L2 on actor to stabilize:
        l2_actor = sum((p**2).sum() for p in self.policy.parameters()) * 1e-6
        loss = actor_loss + critic_loss + l2_actor

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.opt_actor.step()
        self.opt_critic.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "loss": float(loss.item()),
        }


# ============== Training Loop Example ==============
def train_example():
    # ---- Adjust these paths to your project ----
    EMB_PATH = "outputs/embeddings.npy"   # produced by your exporter
    PRICES_CSV = "data/ABB_clean.csv"     # must contain 'close'

    # Environment with ONLY embeddings as state
    env_cfg = EnvConfig(
        embeddings_path=EMB_PATH,
        prices_csv_path=PRICES_CSV,
        trade_amount_usd=5.0,
        start_cash_usd=1000.0,
        reward_mode="level",  # per spec: reward = unrealized + realized
        seed=7,
    )
    env = TradingEnvEmbeddingOnly(env_cfg)

    # Agent
    obs_dim = env.D
    train_cfg = TrainConfig(
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        max_episodes=10,     # increase for real training
        device="cpu",
        seed=7,
    )
    agent = ActorCriticAgent(obs_dim, train_cfg)

    # Training
    for ep in range(train_cfg.max_episodes):
        obs = env.reset()
        done = False
        ep_rewards = []
        logps, values = [], []
        steps = 0

        while not done:
            action, logp, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            logps.append(logp)
            values.append(value)
            ep_rewards.append(reward)

            obs = next_obs
            steps += 1

        stats = agent.update(logps, values, ep_rewards, dones=None)
        ep_return = sum(ep_rewards)
        print(f"Episode {ep+1:03d} | steps={steps} | return={ep_return:.4f} "
              f"| actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    train_example()
