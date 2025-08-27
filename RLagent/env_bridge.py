# RLagent/env_bridge.py (solo los cambios clave)

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import os

@dataclass
class EnvConfig:
    prices_path: str
    embeddings_path: str              # outputs/embeddings.npy (ventana-level)
    predictions_path: Optional[str] = None
    mmap: bool = False
    window_end_idx_path: Optional[str] = "outputs/window_end_idx.npy"  # <--- NUEVO (opcional)

class Environment:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._load_once()

    def give_state(self, episode: int) -> Dict[str, Any]:
        idx_bar = self._validate_index(episode)

        # Si tenemos índices de fin de ventana, buscamos el embedding válido más cercano (<= episodio)
        if self.window_end_idx is not None:
            # end_idx está ordenado; buscamos el último <= episode
            pos = np.searchsorted(self.window_end_idx, idx_bar, side="right") - 1
            if pos < 0:
                # Aún no hay ventana que termine antes o en 'episode'
                emb = np.full((self.embeddings.shape[1],), np.nan, dtype=float)
            else:
                emb = self.embeddings[pos]
        else:
            # Modo simple: asumimos per-bar alignment 1:1 (solo si tu embeddings.npy fue generado así)
            emb = self.embeddings[idx_bar]

        price_prediction = float("nan")
        if self.predictions is not None:
            if self.window_end_idx is not None:
                # si hay pred por ventana, usa el mismo pos encontrado
                pred = self.predictions[pos] if pos >= 0 else np.nan
            else:
                pred = self.predictions[idx_bar]
            if np.ndim(pred) > 0:
                pred = np.asarray(pred).reshape(-1)[0]
            price_prediction = float(pred) if not np.isnan(pred) else float("nan")

        return {
            "current_price": float(self.prices[idx_bar]),
            "embedding": emb if emb.ndim == 1 else np.asarray(emb).reshape(-1),
            "price_prediction": price_prediction,
        }

    def _load_once(self):
        # precios
        df = pd.read_csv(self.cfg.prices_path)
        if "close" not in df.columns:
            raise ValueError(f"'close' column not found in {self.cfg.prices_path}")
        self.prices = df["close"].astype(float).to_numpy()

        # embeddings (ventana-level)
        mmap_mode = "r" if self.cfg.mmap else None
        if not os.path.isfile(self.cfg.embeddings_path):
            raise FileNotFoundError(f"embeddings_path not found: {self.cfg.embeddings_path}")
        self.embeddings = np.load(self.cfg.embeddings_path, mmap_mode=mmap_mode)
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [N_windows, D], got {self.embeddings.shape}")

        # predicciones (opcional, ventana-level)
        self.predictions = None
        if self.cfg.predictions_path:
            if not os.path.isfile(self.cfg.predictions_path):
                raise FileNotFoundError(f"predictions_path not found: {self.cfg.predictions_path}")
            self.predictions = np.load(self.cfg.predictions_path, mmap_mode=mmap_mode)

        # índices de fin de ventana (opcional)
        self.window_end_idx = None
        if self.cfg.window_end_idx_path and os.path.isfile(self.cfg.window_end_idx_path):
            self.window_end_idx = np.load(self.cfg.window_end_idx_path)
            if self.window_end_idx.ndim != 1:
                raise ValueError("window_end_idx must be 1-D")

        # Longitud (en barras) para validación de episodios
        self.N = len(self.prices)

    def _validate_index(self, episode: int) -> int:
        if not isinstance(episode, (int, np.integer)):
            raise TypeError(f"episode must be int, got {type(episode)}")
        if episode < 0 or episode >= self.N:
            raise IndexError(f"episode {episode} out of range [0, {self.N-1}]")
        return int(episode)
