# RLagent/ex.py
from RLagent.env_bridge import EnvConfig, Environment


cfg = EnvConfig(
    prices_path="data/ABB_clean.csv",
    embeddings_path="outputs/embeddings.npy",       # generado en el paso 1
    predictions_path=None,                           # opcional, si la generas
    mmap=False,
    window_end_idx_path="outputs/window_end_idx.npy" # para mapear episodio->ventana
)

env = Environment(cfg)
s = env.give_state(episode=1234)  # 1234 * 5 min desde el inicio
print("current_price:", s["current_price"])
print("embedding shape:", s["embedding"].shape)
print("price_prediction:", s["price_prediction"])
