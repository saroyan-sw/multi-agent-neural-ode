# Multi-Agent Neural ODE

This repository contains a LangGraph workflow that asks several specialized agents to design, generate, execute, check, and repair a compact Neural ODE experiment for the Silverbox benchmark.

```mermaid
flowchart LR
    O[Orchestrator] --> S[Scientist]
    S --> E[Notebook executor]
    E --> C[Result checker]
    C -->|repair needed| E
    C -->|accepted| N[Executed Neural ODE notebook]
    N --> R[CSV metrics and prediction plot]
```

## Repository contents

| Path | Purpose |
| --- | --- |
| `agent/task6.ipynb` | Multi-agent LangGraph workflow and prompts |
| `generated/Duffing_generated.ipynb` | Notebook produced by the agents and executed successfully |
| `results/comparison.csv` | Machine-readable baseline settings and metrics |
| `results/test_prediction.png` | Measured and predicted test signals |
| `run_evidence/` | Execution summary and captured standard output/error |

## Successful baseline

The generated notebook uses the 4096-sample Silverbox excerpt `42650:46746`, with 3072 training samples and 1024 test samples. The model learns acceleration as

\[
y' = v, \qquad v' = \alpha_\theta(y,v,u)
\]

and performs a continuous rollout without resetting the state at the train/test boundary.

| Setting | Value |
| --- | --- |
| Network | `3 -> 64 -> 64 -> 1` |
| Activation | Tanh |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Rollout length | 32 |
| Updates | 800 |
| Training time | 48.69 s |
| Train RMSE | 0.034713 V |
| Test RMSE | 0.039680 V |
| Zero-prediction test RMSE | 0.060492 V |

![Continuous Silverbox test prediction](results/test_prediction.png)

## Run locally

The notebooks expect the Silverbox file `SNLS80mV.mat`. The dataset is not included in this repository. Update `DATA_PATH` in the notebook copies to the location of the file on your computer.

In PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GOOGLE_API_KEY = "your-key-here"
jupyter lab agent/task6.ipynb
```

The API key is read from the environment and is not stored in the repository. Run `generated/Duffing_generated.ipynb` directly if you only want to reproduce the numerical experiment and already have the data file.

## Scope

This run intentionally validates one baseline configuration first. The repository records the resulting end-to-end pipeline and its actual outputs; it does not claim a full architecture search or a final optimized physical model.
