# Multi-Agent Neural ODE

This repository contains two LangGraph workflows. The first asks specialized agents to design, generate, execute, check, and repair a compact Neural ODE experiment for the Silverbox benchmark. The second autonomously optimizes a binary DMD pattern on a noisy simulated optical bench.

```mermaid
flowchart LR
    O[Orchestrator] --> S[Scientist]
    S --> E[Notebook executor]
    O --> H[Human interrupt]
    H -->|resume same thread| O
    E --> C[Result checker]
    C -->|repair needed| E
    C -->|accepted| N[Executed Neural ODE notebook]
    N --> R[CSV metrics and prediction plot]
```

## Repository contents

| Path | Purpose |
| --- | --- |
| `agent/task6.ipynb` | Multi-agent LangGraph workflow and prompts |
| `agent/task2.ipynb` | Autonomous simulated beam-focusing agent and tool |
| `generated/Duffing_generated.ipynb` | Notebook produced by the agents and executed successfully |
| `results/comparison.csv` | Machine-readable baseline settings and metrics |
| `results/test_prediction.png` | Measured and predicted test signals |
| `run_evidence/` | Execution summary and captured standard output/error |
| `human_in_loop_runs/` | Committed checkpoint transcripts and destinations for later demonstration runs |
| `recordings/` | Two human-in-the-loop recordings and one autonomous beam-focusing recording |
| `tests/test_hitl_contract.py` | Offline interrupt, refusal, resume, and code-cell checks |

## Autonomous simulated beam focusing

`agent/task2.ipynb` implements Homework 2 Task 2 as a separate, self-contained notebook. After the user submits `Focus the beam.`, the graph completes three experiments without requesting settings or approvals.

```mermaid
flowchart LR
    U[Focus the beam] --> O[Orchestrator]
    O --> S[Scientist]
    S -->|settings and reason| O
    O --> E[Executor]
    E --> T[focus_beam tool]
    T -->|camera image and enhancement| O
    O -->|three successful experiments| F[Final comparison]
```

The scientist establishes a baseline, then changes one setting at a time based on the previous measured enhancement. The executor is the only role that calls the predefined focusing tool. Every run uses the same fixed complex transmission matrix and an amplitude-only `16 × 16` binary DMD. The simulated `24 × 24` camera includes independent Gaussian noise.

The notebook validates the optical model before starting the agents and saves the completed run under `focusing_runs/`:

| Output | Purpose |
| --- | --- |
| `experiments.csv` | Settings, reasoning, measured enhancement, runtime, and image paths for all three experiments |
| `task2_transcript.json` | Complete autonomous agent dialogue and tool results |
| `run_XX/result.json` | Structured result for one experiment |
| `run_XX/camera_comparison.png` | Before-and-after camera images using one color scale |
| `run_XX/dmd_pattern.png` | Best binary DMD pattern |
| `run_XX/enhancement_history.png` | Best measured enhancement over the search |

The lecture-comparison section remains explicitly marked for the real lecture image or value. The notebook does not invent a reference result.

## Human-in-the-loop extension

Both agent variants now support a fifth `human` destination. The orchestrator's model call selects it when a request is meaningfully ambiguous or when the next action would execute code, overwrite a result, or report a measurement.

The graph is compiled with an in-memory checkpointer. A human question pauses the current thread through `interrupt()`, and `Command(resume=...)` continues that same thread. The reply is appended to the shared dialogue with role `human`. Refusing a hard action returns control to the orchestrator without running the executor.

The notebook includes two checkpointed demonstrations for the notebook-generating graph, followed by equivalent validation cells for the Python-file graph. The demonstrations now pause in the running cell at `YOUR REPLY:`, accept a manually typed answer, and resume the same thread automatically.

## Recorded human decisions

Two real notebook-graph checkpoint snapshots are committed as report evidence:

- [`ambiguous_run.json`](human_in_loop_runs/notebook_graph/ambiguous_run.json) records the ambiguous failure-diagnosis request, the human clarification that forbids repair or execution, and the general agent's explanation. Its `executor_runs` value is `0`.
- [`hard_run_after_refusal.json`](human_in_loop_runs/notebook_graph/hard_run_after_refusal.json) records the scientist's larger-model comparison strategy, the first executor approval question, the human refusal, and the second approval interrupt. Its `executor_runs` value is also `0`, proving the refused action did not run.

These files are checkpoint snapshots rather than invented example conversations. The accepted hard-action transcript and combined screen-recording transcript are created only after the corresponding interactive cells complete successfully.

## Screen recordings

The actual notebook interactions are available as MP4 recordings:

| Recording | Duration | Description |
| --- | ---: | --- |
| [`recording-2026-10-13-003225.mp4`](recordings/recording-2026-10-13-003225.mp4) | 4:44 | Extended interactive human-in-the-loop run, including the scientific planning and decision flow |
| [`recording-2026-10-13-003930.mp4`](recordings/recording-2026-10-13-003930.mp4) | 0:55 | Short follow-up recording of the notebook interaction |
| [`beam-focusing.mp4`](recordings/beam-focusing.mp4) | 5:50 | Autonomous Task 2 run showing the agent choosing settings, calling the focusing tool, and comparing the three experiments |

The videos are stored with Git LFS because the longer recording is larger than GitHub's normal per-file Git limit. Clone with Git LFS enabled to retrieve the MP4 files locally.

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

For the autonomous focusing experiment, open `agent/task2.ipynb`. Task 2 uses OpenRouter's `openrouter/free` router through `ChatOpenAI`. Its API-key cell asks for a fresh `OPENROUTER_API_KEY` without saving it. Run the notebook from top to bottom, enter the key once in the hidden prompt, and then execute the single `Focus the beam.` request.

The API key is read from the environment and is not stored in the repository. Run `generated/Duffing_generated.ipynb` directly if you only want to reproduce the numerical experiment and already have the data file.

## Scope

This run intentionally validates one baseline configuration first. The repository records the resulting end-to-end pipeline and its actual outputs; it does not claim a full architecture search or a final optimized physical model.
