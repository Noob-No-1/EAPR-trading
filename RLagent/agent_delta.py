# RLagent/agent_delta.py
# ---------------------------------------------------------
# RL agent prototype (state = ONLY embedding) with DELTA reward
# - Reward = change in equity between steps
# - Episode length cap + reward scaling
# - Prints per-episode realized/unrealized PnL and their sum
# - NEW: counts of buys/sells, and per-episode trades CSV with timestamps
# ---------------------------------------------------------
from __future__ import annotations
import os, random, csv
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from datetime import datetime


# ============== Environment (delta reward) ==============
@dataclass
class EnvConfig:
    embeddings_path: str                  # npy: [N, D]
    prices_csv_path: str                  # CSV with 'close' and a time column
    time_col: str = "Unnamed: 0"         # timestamp column name (falls back to index if missing)
    trade_amount_usd: float = 5.0         # fixed $ amount per step
    start_cash_usd: float = 1000.0        # initial cash
    episode_len: int = 2000               # max steps per episode
    reward_scale: float = 1.0             # multiply delta equity
    seed: int = 42

class TradingEnvEmbeddingOnlyDelta:
    """
    Observation: ONLY the embedding vector at time t (np.ndarray (D,))
    Actions: 0=hold, 1=buy, 2=sell (fixed USD amount per step)
    Reward: delta equity = (cash + position*price)_t - (cash + position*price)_{t-1}
    Logs trades with timestamps, keeps buy/sell counters.
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

        # timestamps
        if self.cfg.time_col in df.columns:
            ts = pd.to_datetime(df[self.cfg.time_col], errors="coerce")
        else:
            # fallback: use first column if looks like time, else range index
            first_col = df.columns[0]
            ts = pd.to_datetime(df[first_col], errors="coerce")
        if ts.isna().all():
            # build synthetic timeline (5-min spacing) if cannot parse
            ts = pd.date_range(start="2000-01-01 09:30:00", periods=len(df), freq="5min")
        self.timestamps = ts.astype("datetime64[ns]").to_numpy()

        # Align lengths (safeguard)
        N = min(len(self.embeddings), len(self.prices), len(self.timestamps))
        self.embeddings = self.embeddings[:N]
        self.prices = self.prices[:N]
        self.timestamps = self.timestamps[:N]
        self.N = N
        self.D = self.embeddings.shape[1]

        # Episode state
        self.t = 0
        self.steps_left = self.cfg.episode_len
        self.reset()

    def reset(self, start_index: Optional[int] = None) -> np.ndarray:
        """Reset environment; start at a random valid index if not provided."""
        if start_index is None:
            max_start = max(0, self.N - self.cfg.episode_len - 1)
            self.t = int(self.rng.integers(0, max(1, max_start+1)))
        else:
            self.t = int(max(0, min(self.N - 2, start_index)))

        self.steps_left = self.cfg.episode_len
        self.cash = float(self.cfg.start_cash_usd)    # spot dollars
        self.position_shares = 0.0
        self.mean_entry_price = 0.0
        self.last_entry_price = 0.0
        self.realized_profit = 0.0

        # logging / counters
        self.trades: List[Dict[str, Any]] = []
        self.num_buys = 0
        self.num_sells = 0

        self.prev_equity = self._equity(self.t)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        # State is ONLY the embedding vector at time t
        return self.embeddings[self.t].astype(np.float32, copy=False)

    def _equity(self, t_idx: int) -> float:
        price = self.prices[t_idx]
        return self.cash + self.position_shares * price

    def _unrealized_profit(self, price: float) -> float:
        if self.position_shares <= 0.0 or self.mean_entry_price <= 0.0:
            return 0.0
        return (price - self.mean_entry_price) * self.position_shares

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: 0=hold, 1=buy, 2=sell
        Fixed USD size self.cfg.trade_amount_usd. If selling more than position, sell all.
        Reward = (equity_now - prev_equity) * reward_scale
        """
        if self.t >= self.N - 1 or self.steps_left <= 0:
            obs = self._get_obs()
            return obs, 0.0, True, self._info()

        price = float(self.prices[self.t])
        timestamp = pd.Timestamp(self.timestamps[self.t]).to_pydatetime()
        trade_usd = float(self.cfg.trade_amount_usd)

        # Execute action
        if action == 1:  # BUY
            if self.cash > 0.0:
                spend = min(trade_usd, self.cash)
                if spend > 0.0:
                    shares = spend / price
                    total_shares = self.position_shares + shares
                    if total_shares > 0:
                        self.mean_entry_price = (
                            (self.mean_entry_price * self.position_shares) + (price * shares)
                        ) / total_shares
                        self.last_entry_price = price
                    self.position_shares = total_shares
                    self.cash -= spend
                    self.num_buys += 1
                    self._log_trade("BUY", timestamp, self.t, price, shares, spend)

        elif action == 2:  # SELL
            position_value = self.position_shares * price
            if position_value > 1e-12:
                sell_value = min(trade_usd, position_value)
                shares_to_sell = min(sell_value / price, self.position_shares)

                realized_this = (price - self.mean_entry_price) * shares_to_sell
                self.realized_profit += realized_this

                self.position_shares -= shares_to_sell
                self.cash += shares_to_sell * price

                if self.position_shares <= 1e-12:
                    self.position_shares = 0.0
                    self.mean_entry_price = 0.0
                    self.last_entry_price = 0.0

                self.num_sells += 1
                self._log_trade("SELL", timestamp, self.t, price, shares_to_sell, sell_value, realized_this)

        # Delta equity reward
        equity_now = self._equity(self.t)
        reward = (equity_now - self.prev_equity) * self.cfg.reward_scale
        self.prev_equity = equity_now

        # Advance time
        self.t += 1
        self.steps_left -= 1
        done = (self.t >= self.N - 1) or (self.steps_left <= 0)

        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), done, info

    def _log_trade(
        self,
        side: str,
        ts: datetime,
        idx: int,
        price: float,
        shares: float,
        value_usd: float,
        realized_pnl_this: float = 0.0,
    ) -> None:
        self.trades.append({
            "timestamp": ts.isoformat(),
            "index": idx,
            "side": side,
            "price": float(price),
            "shares": float(shares),
            "value_usd": float(value_usd),
            "realized_pnl_this": float(realized_pnl_this),
            "realized_pnl_cum": float(self.realized_profit),
        })

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
            "num_buys": int(self.num_buys),
            "num_sells": int(self.num_sells),
        }

    # helper para guardar CSV de trades
    def save_trades_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ["timestamp", "index", "side", "price", "shares", "value_usd", "realized_pnl_this", "realized_pnl_cum"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in self.trades:
                w.writerow(row)


# ============== Actor-Critic (Policy Learning) ==============
class PolicyNet(nn.Module):
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
        return self.net(x)  # logits


class ValueNet(nn.Module):
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
    max_episodes: int = 20
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
        logits = self.policy(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.value(x)
        return int(action.item()), logp.squeeze(0), value.squeeze(0)

    def update(self, logps, values, rewards):
        logps_t = torch.stack(logps)
        values_t = torch.stack(values)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.cfg.device)

        # Monte Carlo returns
        returns = []
        G = 0.0
        for r in reversed(rewards_t.tolist()):
            G = r + self.cfg.gamma * G
            returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=self.cfg.device)

        advantages = returns - values_t.detach()

        actor_loss = -(logps_t * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(values_t, returns)

        # small L2 regularization
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
    EMB_PATH = "outputs/embeddings.npy"
    PRICES_CSV = "data/ABB_clean.csv"

    env_cfg = EnvConfig(
        embeddings_path=EMB_PATH,
        prices_csv_path=PRICES_CSV,
        time_col="Unnamed: 0",
        trade_amount_usd=5.0,
        start_cash_usd=1000.0,
        episode_len=1500,
        reward_scale=1.0,
        seed=7,
    )
    env = TradingEnvEmbeddingOnlyDelta(env_cfg)

    obs_dim = env.D
    train_cfg = TrainConfig(
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        max_episodes=10,
        device="cpu",
        seed=7,
    )
    agent = ActorCriticAgent(obs_dim, train_cfg)

    os.makedirs("outputs", exist_ok=True)

    for ep in range(train_cfg.max_episodes):
        obs = env.reset()     # random start within data
        init_info = env._info()
        init_equity = init_info["balance"]

        done = False
        ep_rewards = []
        logps, values = [], []
        steps = 0
        last_info = init_info

        while not done:
            action, logp, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            logps.append(logp)
            values.append(value)
            ep_rewards.append(reward)

            obs = next_obs
            steps += 1
            last_info = info

        stats = agent.update(logps, values, ep_rewards)
        ep_return = float(np.sum(ep_rewards))

        realized = float(last_info["realized_profit"])
        unrealized = float(last_info["unrealized_profit"])
        total_pnl = realized + unrealized
        final_equity = float(last_info["balance"])
        approx_check = final_equity - init_equity

        num_buys = last_info["num_buys"]
        num_sells = last_info["num_sells"]

        # ---- imprimir métricas episodio ----
        print(
            f"Episode {ep+1:03d} | steps={steps} | return={ep_return:.4f} "
            f"| actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.4f}"
        )
        print(
            f"    Trades: buys={num_buys} | sells={num_sells}"
        )
        print(
            f"    PnL: realized={realized:.2f} | unrealized={unrealized:.2f} "
            f"| total={total_pnl:.2f} | Δequity≈{approx_check:.2f}"
        )

        # ---- guardar CSV de trades del episodio ----
        trades_csv = f"outputs/trades_ep{ep+1:03d}.csv"
        env.save_trades_csv(trades_csv)
        print(f"    Saved trades CSV -> {trades_csv}")

    print("Training finished.")

if __name__ == "__main__":
    train_example()
