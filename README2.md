# EAPR-trading + RL Bridge

**Goal:**  
Learn robust intraday time-series representations (**embeddings**) using a self-supervised encoder (**TS2Vec**) and provide them as **state** to a **Reinforcement Learning (RL) agent** through an efficient `Environment` class.

This project includes:

1. **Preprocessing**: convert CSV (5-min candles + features) into windows.  
2. **Training** (self-supervised): TS2Vec with **InfoNCE** contrastive loss (robust to “entropy”/noise/occlusions via time masking + Gaussian noise).  
3. **Evaluation** (optional): supervised metrics (RMSE, MAE, ACC, AUC).  
4. **Embeddings export** (new): generate `outputs/embeddings.npy`.  
5. **RL Bridge** (new): `Environment.give_state(episode)` → `{current_price, embedding, price_prediction}`.

---

## 0) Requirements

- Python 3.10+ (tested with 3.11/3.13)
- pip/venv recommended
- PyTorch, NumPy, Pandas, scikit-learn, tqdm, PyYAML

Install dependencies:

```bash
pip install -r requriement.txt
```

> Note: file is named `requriement.txt` (typo) in the original ZIP.

---

## 1) CSV format

Default config (`configs/ts2vec_baseline.yaml`):

```yaml
data:
  csv_path: data/ABB_clean.csv
  time_col: "Unnamed: 0"
  feature_cols: [open, high, low, close, volume, log_return, hl_range, tod_sin, tod_cos]
```

Your **CSV** must include these columns.  
- `time_col` is used for ordering.  
- `feature_cols` go into the encoder.  
- If `close` exists, `y.npy` will be created (future log returns at `target_h` horizon).  

---

## 2) Preprocessing — create windows

Convert CSV into normalized windows + splits:

```bash
python data_preprocessing/make_windows.py   --csv data/ABB_clean.csv   --time_col "Unnamed: 0"   --feature_cols open high low close volume log_return hl_range tod_sin tod_cos   --lin 256   --stride 16   --train_ratio 0.70   --val_ratio 0.15   --out_dir outputs/windows   --target_h 12
```

Generates:
```
outputs/windows/X.npy
outputs/windows/y.npy
outputs/windows/splits.json
outputs/windows/scaler.pkl
```

---

## 3) Train TS2Vec (self-supervised)

```bash
python models/ts2vec/train.py --config configs/ts2vec_baseline.yaml
```

Saves:
```
outputs/ts2vec_baseline/ts2vec_baseline.ckpt
```

---

## 4) (Optional) Supervised evaluation

Check embedding quality for forecasting/classification:

```bash
python -m eval.eval_forecast
# or
PYTHONPATH=. python eval/eval_forecast.py
```

Outputs:
```
outputs/ts2vec_baseline/results_baseline.json
```

---

## 5) **New** — Export embeddings

Training saves a checkpoint but not embeddings.  
Run exporter to create `embeddings.npy` and map windows→bars:

```bash
python -m RLagent.make_embeddings   --ckpt outputs/ts2vec_baseline/ts2vec_baseline.ckpt   --root outputs/windows   --out outputs/embeddings.npy   --out_idx outputs/window_end_idx.npy   --pool mean
```

Creates:
```
outputs/embeddings.npy       # [N_windows, 128]
outputs/window_end_idx.npy   # [N_windows], 5-min bar indices for each window end
```

> Embedding size (`proj_dim`) defaults to **128**.

---

## 6) **New** — RL Bridge: `Environment`

`RLagent/env_bridge.py` loads data once (prices, embeddings, predictions) and provides states efficiently.

Example (`RLagent/ex.py`):

```python
from RLagent.env_bridge import EnvConfig, Environment

cfg = EnvConfig(
  prices_path="data/ABB_clean.csv",
  embeddings_path="outputs/embeddings.npy",
  predictions_path=None,                          # optional (see §7)
  mmap=False,
  window_end_idx_path="outputs/window_end_idx.npy"
)
env = Environment(cfg)

s = env.give_state(episode=1234)  # 1234 * 5 minutes since start
print("current_price:", s["current_price"])
print("embedding shape:", s["embedding"].shape)   # (128,)
print("price_prediction:", s["price_prediction"]) # NaN if no preds
```

Run from repo root:

```bash
python -m RLagent.ex
```

---

## 7) (Optional) Predictions

`price_prediction` is **NaN** unless you provide `predictions_path`.  
To enable:

1. Train a model (e.g., `LinearRegression`) with `embeddings.npy` → `y.npy`.  
2. Save predictions to `outputs/preds.npy` (aligned with embeddings).  
3. Pass it in config: `predictions_path="outputs/preds.npy"`.  

---

## 8) Folder structure

```
EAPR-trading/
├─ configs/
├─ data/
│  └─ ABB_clean.csv
├─ data_preprocessing/
├─ eval/
├─ models/
│  └─ ts2vec/
├─ outputs/
│  ├─ windows/
│  │  ├─ X.npy
│  │  ├─ y.npy
│  │  ├─ splits.json
│  │  └─ scaler.pkl
│  ├─ ts2vec_baseline/
│  │  ├─ ts2vec_baseline.ckpt
│  │  └─ results_baseline.json
│  ├─ embeddings.npy
│  └─ window_end_idx.npy
└─ RLagent/
   ├─ __init__.py
   ├─ make_embeddings.py
   ├─ env_bridge.py
   └─ ex.py
```

---

## 9) Troubleshooting

- **`FileNotFoundError: ABB_clean.csv`** → Check CSV path (`--csv data/ABB_clean.csv`).  
- **`ModuleNotFoundError: No module named 'models'`** → Run as module: `python -m eval.eval_forecast`.  
- **`ModuleNotFoundError: No module named 'env_bridge'`** → Import as package: `from RLagent.env_bridge ...` and ensure `RLagent/__init__.py`.  
- **`embeddings_path not found`** → Run embedding exporter (§5).  
- **`price_prediction: nan`** → Provide `predictions_path` (§7).  

---

## 10) All steps in order

```bash
# 0) deps
pip install -r requriement.txt

# 1) preprocessing
python data_preprocessing/make_windows.py   --csv data/ABB_clean.csv   --time_col "Unnamed: 0"   --feature_cols open high low close volume log_return hl_range tod_sin tod_cos   --lin 256 --stride 16   --train_ratio 0.70 --val_ratio 0.15   --out_dir outputs/windows --target_h 12

# 2) train TS2Vec
python models/ts2vec/train.py --config configs/ts2vec_baseline.yaml

# 3) (optional) evaluation
python -m eval.eval_forecast

# 4) export embeddings
python -m RLagent.make_embeddings   --ckpt outputs/ts2vec_baseline/ts2vec_baseline.ckpt   --root outputs/windows   --out outputs/embeddings.npy   --out_idx outputs/window_end_idx.npy   --pool mean

# 5) run RL environment example
python -m RLagent.ex
```

---

## 11) RL Integration

- **State** = `embedding` (ℝ¹²⁸) + optionally `current_price` and `price_prediction`.  
- RL agent (PPO, SAC, A2C, DDPG, etc.) can consume continuous state vectors directly.  
- `Environment` is efficient: all data is loaded once; `give_state(episode)` is O(1).

-------------------

python -m RLagent.agent_delta_lr

python -m RLagent.plot_trades \
  --prices_csv data/ABB_clean.csv \
  --time_col "Unnamed: 0" \
  --trades_csv outputs/trades_ep001.csv \
  --x_mode index \
  --zoom trades \
  --title "ABB Close with Trades (Ep 1)"

python -m RLagent.plot_trades \
  --prices_csv data/ABB_clean.csv \
  --time_col "Unnamed: 0" \
  --merge_glob "outputs/trades_ep*.csv" \
  --x_mode index \
  --zoom trades \
  --title "ABB Close with Trades (All Episodes)"
