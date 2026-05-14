# LMAC : LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning

This repository extends the original [EPyMARL](https://github.com/uoe-agents/epymarl) framework with an
LLM-guided communication pipeline. All algorithmic components (controllers, learners, runners) and the
general project layout still follow EPyMARL conventions.

## Environment Setup

```bash
# 1) Create and activate the Conda environment (Python 3.9)
conda create -n epymarl-llm python=3.9 -y
conda activate epymarl-llm

# 2) Install the core MARL dependencies
pip install -r requirements.txt

# 3) Install any environment-specific extras (e.g., SC2, SMACv2, PettingZoo, etc.)
pip install -r env_requirements.txt
# 4) (Optional) Enable development tooling or notebook support
pip install -r dev_requirements.txt

```

### LLM API Keys

Open `src/llm_final.py` and replace the placeholders with your own credentials:

```python
# src/llm_final.py
OPENAI_KEY = "your-openai-key"
# GEMINI_KEY = "your-gemini-key"
# CLAUDE_KEY = "your-claude-key"
```

### Weights & Biases Token

If you plan to log runs to W&B, store the API token locally:

```bash
echo "<your-wandb-key>" > wandb_key.txt
```

### Data Preparation

The `src/llm_final.py` script requires pre-collected trajectory data to be placed in specific directories. The script **does not** collect data itself; it strictly requires existing `.pkl` files.

1.  **Create a `data` directory** in the project root if it doesn't exist.
2.  **Create a subdirectory** matching your `map_name` (for SC2/GRF) or `env_key` (for LBF).
3.  **Place training data (`.pkl` files)** directly inside this subdirectory.
4.  **Create a `test` subdirectory** inside the map directory.
5.  **Place test data (`.pkl` files)** inside the `test` subdirectory.

**Directory Structure Example (for SC2 map `1o_10b_vs_1r`):**

```text
LMAC/Code/
├── data/
│   └── 1o_10b_vs_1r/          <-- BUFFER_DIR
│       ├── training_traj_1.pkl
│       ├── training_traj_2.pkl
│       ├── ...
│       └── test/              <-- TEST_BUFFER_DIR
│           ├── test_traj_1.pkl
│           ├── test_traj_2.pkl
│           └── ...
```

> **Note:** The script checks for `.pkl` files in these directories. Ensure both training and test directories contain sufficient data files (default batch size requires at least 32 files, though `check_buffer_availability` default is 10).

## How to Run

```bash
CUDA_VISIBLE_DEVICES=0 python src/llm_final.py \
  --config=lmac \
  --env=sc2 \
  --map=1o_10b_vs_1r \
  --mse_thres=0.05 \
  --meta_lambda=0.1 \
  --recon_lambda=1 \
  --consistency_lambda=1
```

Key flags:
- `--config`, `--env`, `--map`: select the algorithm and SC2 map (replace with your target setup).
- `--mse_thres`, `--meta_lambda`, `--recon_lambda`, `--consistency_lambda`: tune discriminator and training hyperparameters.
- export or edit keys in `src/llm_final.py` before running if you rely on the LLM pipeline.


## References

- EPyMARL original paper and repository: <https://github.com/uoe-agents/epymarl>
- SMAC / SMACv2 environments: <https://github.com/oxwhirl/smac> and <https://github.com/oxwhirl/smacv2>
- PettingZoo cooperative benchmarks: <https://pettingzoo.farama.org/>
