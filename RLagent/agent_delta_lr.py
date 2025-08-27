# RLagent/agent_delta_lr.py
# ---------------------------------------------------------
# RL agent (state = [embedding, linear_regression_prediction])
# - Usa window_end_idx.npy para mapear cada ventana -> barra real del CSV
# - Recompensa = delta equity
# - Episodio largo: recorre todo el histórico (start_index=0, episode_len muy grande)
# - Log de trades con index = índice de barra real (ideal para el plot)
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

from sklearn.linear_model import LinearRegression  # para opción LR on-the-fly


# ============== Environment (delta reward) ==============
@dataclass
class EnvConfig:
    embeddings_path: str                    # npy: [N_windows, D]
    prices_csv_path: str                    # CSV con 'close' y columna tiempo
    time_col: str = "Unnamed: 0"
    trade_amount_usd: float = 5.0
    start_cash_usd: float = 1000.0
    episode_len: int = 10**9                # grande para cubrir todo el histórico
    reward_scale: float = 1.0
    seed: int = 42
    # Predicción de regresión lineal
    predictions_path: Optional[str] = None  # Opción A: npy [N_windows]
    fit_linear_on_the_fly: bool = False     # Opción B: ajusta LR(emb)->y y predice
    y_path: str = "outputs/windows/y.npy"   # solo si fit_linear_on_the_fly=True
    # Mapeo ventana -> barra
    window_end_idx_path: Optional[str] = "outputs/window_end_idx.npy"


class TradingEnvEmbeddingPlusLRDelta:
    """
    Observación: concat([embedding_t (D), lr_pred_t (1)]) -> float32 (D+1,)
    Las operaciones y precios se toman en la barra real indicada por window_end_idx[t].
    Acciones: 0=hold, 1=buy, 2=sell (monto fijo en USD).
    Recompensa: delta equity.
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # --- Embeddings ---
        self.embeddings = np.load(cfg.embeddings_path)  # [N_windows, D]
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D [N,D], got {self.embeddings.shape}")
        self.D = self.embeddings.shape[1]

        # --- Precios & Timestamps (todas las barras del CSV) ---
        df = pd.read_csv(cfg.prices_csv_path)
        if "close" not in df.columns:
            raise ValueError(f"'close' column not found in {cfg.prices_csv_path}. Columns: {list(df.columns)}")
        self.prices = df["close"].astype(float).to_numpy()

        if cfg.time_col in df.columns:
            ts = pd.to_datetime(df[cfg.time_col], errors="coerce")
        else:
            ts = pd.to_datetime(df[df.columns[0]], errors="coerce")
        if ts.isna().all():
            ts = pd.date_range(start="2000-01-01 09:30:00", periods=len(df), freq="5min")
        self.timestamps = ts.astype("datetime64[ns]").to_numpy()

        # --- Mapeo ventana -> índice de barra real ---
        if not (cfg.window_end_idx_path and os.path.isfile(cfg.window_end_idx_path)):
            raise FileNotFoundError(
                f"window_end_idx_path not found: {cfg.window_end_idx_path}. "
                "Generate it with RLagent.make_embeddings."
            )
        end_idx = np.load(cfg.window_end_idx_path)
        if end_idx.ndim != 1:
            raise ValueError("window_end_idx must be 1-D")
        end_idx = end_idx.astype(int)

        # Validar rango
        valid = (end_idx >= 0) & (end_idx < len(self.prices))
        self.window_end_idx = end_idx[valid]

        # --- Predicción LR ---
        # Alinearemos a N_windows válidos (len(window_end_idx))
        Nw = len(self.window_end_idx)
        if cfg.predictions_path and os.path.isfile(cfg.predictions_path):
            lr_pred = np.load(cfg.predictions_path).reshape(-1)[:Nw]
            input(lr_pred)
            print("====================================")
        elif cfg.fit_linear_on_the_fly:
            if not os.path.isfile(cfg.y_path):
                raise FileNotFoundError(f"fit_linear_on_the_fly=True but y_path not found: {cfg.y_path}")
            y = np.load(cfg.y_path).reshape(-1)[:Nw]
            X = self.embeddings[:Nw]
            lr = LinearRegression()
            lr.fit(X, y)
            lr_pred = lr.predict(X)

            print(lr_pred, len(lr_pred), len(self.prices))
            input("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        else:
            lr_pred = np.zeros((Nw,), dtype=np.float32)

            input(lr_pred)
            print("------------------------------------")

        # Recortar embeddings a N_windows válidos
        self.embeddings = self.embeddings[:Nw]
        self.lr_pred = lr_pred.astype(np.float32)
        self.N = Nw  # número de VENTANAS

        # Estado episodio
        self.t = 0              # índice de ventana
        self.steps_left = self.cfg.episode_len
        self.reset()

    # índice de barra real para la ventana t
    def _bar_idx(self, t: int) -> int:
        return int(self.window_end_idx[t])

    def reset(self, start_index: Optional[int] = 0) -> np.ndarray:
        """
        Para usar TODO el CSV: start_index=0 y episode_len grande.
        """
        if start_index is None:
            # inicio aleatorio pero dejando espacio al episodio
            max_start = max(0, self.N - self.cfg.episode_len - 1)
            self.t = int(self.rng.integers(0, max(1, max_start + 1)))
        else:
            self.t = int(max(0, min(self.N - 2, start_index)))

        self.steps_left = self.cfg.episode_len
        self.cash = float(self.cfg.start_cash_usd)
        self.position_shares = 0.0
        self.mean_entry_price = 0.0
        self.last_entry_price = 0.0
        self.realized_profit = 0.0

        self.trades: List[Dict[str, Any]] = []
        self.num_buys = 0
        self.num_sells = 0

        self.prev_equity = self._equity(self.t)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        emb = self.embeddings[self.t].astype(np.float32, copy=False)
        pred = np.array([self.lr_pred[self.t]], dtype=np.float32)
        return np.concatenate([emb, pred], axis=0)

    def _equity(self, t_idx_window: int) -> float:
        idx_bar = self._bar_idx(t_idx_window)
        return self.cash + self.position_shares * float(self.prices[idx_bar])

    def _unrealized_profit_from_price(self, price: float) -> float:
        if self.position_shares <= 0.0 or self.mean_entry_price <= 0.0:
            return 0.0
        return (price - self.mean_entry_price) * self.position_shares

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.t >= self.N - 1 or self.steps_left <= 0:
            obs = self._get_obs()
            return obs, 0.0, True, self._info()

        idx_bar = self._bar_idx(self.t)
        price = float(self.prices[idx_bar])
        timestamp = pd.Timestamp(self.timestamps[idx_bar]).to_pydatetime()
        trade_usd = float(self.cfg.trade_amount_usd)

        # Ejecutar acción
        if action == 1:  # BUY
            if self.cash > 0.0:
                spend = min(trade_usd, self.cash)
                if spend > 0.0:
                    shares = spend / price
                    total = self.position_shares + shares
                    if total > 0:
                        self.mean_entry_price = (
                            (self.mean_entry_price * self.position_shares) + (price * shares)
                        ) / total
                        self.last_entry_price = price
                    self.position_shares = total
                    self.cash -= spend
                    self.num_buys += 1
                    self._log_trade("BUY", timestamp, idx_bar, price, shares, spend)

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
                self._log_trade("SELL", timestamp, idx_bar, price, shares_to_sell, sell_value, realized_this)

        # Recompensa delta equity
        equity_now = self._equity(self.t)
        reward = (equity_now - self.prev_equity) * self.cfg.reward_scale
        self.prev_equity = equity_now

        # Avanzar
        self.t += 1
        self.steps_left -= 1
        done = (self.t >= self.N - 1) or (self.steps_left <= 0)

        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), done, info

    def _log_trade(
        self, side: str, ts: datetime, idx_bar: int, price: float, shares: float,
        value_usd: float, realized_pnl_this: float = 0.0,
    ) -> None:
        self.trades.append({
            "timestamp": ts.isoformat(),
            "index": int(idx_bar),  # índice de BARRA para plot
            "side": side,
            "price": float(price),
            "shares": float(shares),
            "value_usd": float(value_usd),
            "realized_pnl_this": float(realized_pnl_this),
            "realized_pnl_cum": float(self.realized_profit),
        })

    def _info(self) -> Dict[str, Any]:
        idx_bar = self._bar_idx(self.t)
        price = float(self.prices[idx_bar])
        position_value = self.position_shares * price
        unreal = self._unrealized_profit_from_price(price)
        balance = self.cash + position_value
        return {
            "balance": float(balance),
            "spot_dollars": float(self.cash),
            "unrealized_profit": float(unreal),
            "realized_profit": float(self.realized_profit),
            "mean_entry_price": float(self.mean_entry_price),
            "last_entry_price": float(self.last_entry_price),
            "position_shares": float(self.position_shares),
            "price": float(price),
            "t": int(self.t),                  # índice de ventana
            "index_bar": int(idx_bar),         # índice de barra real
            "num_buys": int(self.num_buys),
            "num_sells": int(self.num_sells),
            "lr_pred": float(self.lr_pred[self.t]),
        }

    def save_trades_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ["timestamp","index","side","price","shares","value_usd","realized_pnl_this","realized_pnl_cum"]
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
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    max_episodes: int = 1      # un episodio largo (todo el histórico)
    device: str = "cpu"
    seed: int = 123

class ActorCriticAgent:
    def __init__(self, obs_dim: int, cfg: TrainConfig):
        torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
        self.policy = PolicyNet(obs_dim).to(cfg.device)
        self.value  = ValueNet(obs_dim).to(cfg.device)
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.value.parameters(),  lr=cfg.lr_critic)
        self.cfg = cfg

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        x = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        dist = Categorical(logits=self.policy(x))
        a = dist.sample()
        return int(a.item()), dist.log_prob(a).squeeze(0), self.value(x).squeeze(0)

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

        adv = returns - values_t.detach()
        actor_loss = -(logps_t * adv).mean()
        critic_loss = torch.nn.functional.mse_loss(values_t, returns)
        l2_actor = sum((p**2).sum() for p in self.policy.parameters()) * 1e-6
        loss = actor_loss + critic_loss + l2_actor

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.opt_actor.step(); self.opt_critic.step()

        return {"actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "loss": float(loss.item())}


# ============== Training Loop (recorre TODO el CSV) ==============
def train_example():
    EMB_PATH = "outputs/embeddings.npy"
    PRICES_CSV = "data/ABB_clean.csv"

    env_cfg = EnvConfig(
        embeddings_path=EMB_PATH,
        prices_csv_path=PRICES_CSV,
        time_col="Unnamed: 0",
        trade_amount_usd=5.0,
        start_cash_usd=1000.0,
        episode_len=10**9,        # grande para cubrir todo
        reward_scale=1.0,
        seed=7,
        # LR: elige UNA opción
        # predictions_path="outputs/preds.npy",   # Opción A (precalculada)
        fit_linear_on_the_fly=True,              # Opción B (rápida para probar)
        y_path="outputs/windows/y.npy",
        window_end_idx_path="outputs/window_end_idx.npy",
    )
    env = TradingEnvEmbeddingPlusLRDelta(env_cfg)

    obs_dim = env.D + 1  # embedding + lr_pred
    train_cfg = TrainConfig(
        gamma=0.99, lr_actor=1e-3, lr_critic=1e-3,
        max_episodes=1, device="cpu", seed=7,   # 1 episodio largo
    )
    agent = ActorCriticAgent(obs_dim, train_cfg)

    os.makedirs("outputs", exist_ok=True)

    for ep in range(train_cfg.max_episodes):
        # IMPORTANTE: comenzar desde el principio para cubrir todo el CSV
        obs = env.reset(start_index=0)
        init_info = env._info(); init_equity = init_info["balance"]

        done = False
        ep_rewards = []; logps = []; values = []; steps = 0; last_info = init_info

        while not done:
            action, logp, value = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            logps.append(logp); values.append(value); ep_rewards.append(reward)
            steps += 1; last_info = info

        stats = agent.update(logps, values, ep_rewards)
        ep_return = float(np.sum(ep_rewards))

        realized = float(last_info["realized_profit"])
        unrealized = float(last_info["unrealized_profit"])
        total_pnl = realized + unrealized
        final_equity = float(last_info["balance"])
        approx_check = final_equity - init_equity
        num_buys = last_info["num_buys"]; num_sells = last_info["num_sells"]

        print(
            f"Episode {ep+1:03d} | steps={steps} | return={ep_return:.4f} "
            f"| actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.4f}"
        )
        print(f"    Trades: buys={num_buys} | sells={num_sells}")
        print(
            f"    PnL: realized={realized:.2f} | unrealized={unrealized:.2f} "
            f"| total={total_pnl:.2f} | Δequity≈{approx_check:.2f}"
        )

        trades_csv = f"outputs/trades_ep{ep+1:03d}.csv"
        env.save_trades_csv(trades_csv)
        print(f"    Saved trades CSV -> {trades_csv}")

    print("Training finished.")

if __name__ == "__main__":
    train_example()
