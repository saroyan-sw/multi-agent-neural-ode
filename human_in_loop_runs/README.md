# Human-in-the-loop run transcripts

The updated `agent/task6.ipynb` creates this directory's transcript files when its Homework 2 demonstration cells run.

The repository currently includes two real notebook-graph checkpoint snapshots:

- `notebook_graph/ambiguous_run.json`
- `notebook_graph/hard_run_after_refusal.json`

The ambiguity snapshot records a diagnosis-only clarification and confirms that the executor did not run. The refusal snapshot records the completed scientific strategy, the refused coding-tool action, and the second pending approval question; its executor counter remains zero.

Completing the accepted hard-action path creates:

- `notebook_graph/hard_run_after_acceptance.json`
- `notebook_graph/homework2_demonstrations.json`

After those pass, run the Python-file validation cells. They create the equivalent files under `python_graph/`.

The JSON records include the checkpointed dialogue, active interrupt, decision count, and executor-run count. They do not contain the Google API key. Only transcripts actually produced by the notebook are committed; missing acceptance files are not replaced with fabricated examples.
