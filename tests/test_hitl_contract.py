import unittest
from pathlib import Path

import nbformat
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "agent" / "task6.ipynb"


def first_line(cell):
    return next((line.strip() for line in cell.source.splitlines() if line.strip()), "")


def nth_cell(notebook, starts_with, occurrence=0):
    matches = [cell for cell in notebook.cells if first_line(cell) == starts_with]
    return matches[occurrence]


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def invoke(self, _messages):
        if not self.decisions:
            raise AssertionError("The orchestrator requested an unexpected extra decision.")
        return self.decisions.pop(0)


class HumanInTheLoopContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.notebook)

    def build_scope(self, python_variant=False):
        scope = {}
        occurrence = 1 if python_variant else 0
        exec(nth_cell(self.notebook, "import os", occurrence).source, scope)
        exec(nth_cell(self.notebook, "class State(TypedDict):", occurrence).source, scope)
        exec(nth_cell(self.notebook, "class Decision(BaseModel):", occurrence).source, scope)
        exec(nth_cell(self.notebook, "def dialogue_text(state):", occurrence).source, scope)
        orchestrator_start = (
            "PYTHON_MAX_DECISIONS = 8" if python_variant else "MAX_DECISIONS = 8"
        )
        human_start = (
            "def python_human(state: State):"
            if python_variant
            else "def human_node(state: State):"
        )
        exec(nth_cell(self.notebook, orchestrator_start).source, scope)
        exec(nth_cell(self.notebook, human_start).source, scope)
        return scope

    def initial_state(self, python_variant=False):
        state = {
            "dialogue": [
                {"role": "user", "text": "Run the hard action."},
                {"role": "scientist", "text": "A complete plan already exists."},
            ],
            "next_role": "orchestrator",
            "decisions": 0,
            "has_result": False,
            "human_question": "",
            "proposed_action": "",
            "human_reason": "",
            "proposed_role": "",
        }
        if python_variant:
            state.update({"latest_program": "", "python_executor_runs": 0})
        else:
            state.update({"latest_draft": "", "executor_runs": 0})
        return state

    def exercise_refuse_then_accept(self, python_variant=False):
        scope = self.build_scope(python_variant)
        Decision = scope["Decision"]
        decisions = [
            Decision(
                next_role="human",
                instruction="Run the executor and report its measured result.",
                question="May I run the executor and report the result?",
                reason="Running code and reporting a measurement require approval.",
                proposed_role="executor",
            ),
            Decision(
                next_role="human",
                instruction="Run the executor and report its measured result.",
                question="The refused step did not run. May I run it now?",
                reason="The refused step did not run; the human requested another question.",
                proposed_role="executor",
            ),
            Decision(
                next_role="executor",
                instruction="Run the approved executor action.",
                question="",
                reason="The human approved the exact pending action.",
                proposed_role=None,
            ),
        ]
        scope["decision_model"] = ScriptedModel(decisions)

        counter_name = "python_executor_runs" if python_variant else "executor_runs"
        orchestrator = scope["python_orchestrator" if python_variant else "orchestrator"]
        human = scope["python_human" if python_variant else "human_node"]

        def executor_probe(state):
            return {
                "dialogue": [{"role": "executor", "text": "probe executed"}],
                counter_name: state[counter_name] + 1,
            }

        builder = StateGraph(scope["State"])
        builder.add_node("orchestrator", orchestrator)
        builder.add_node("human", human)
        builder.add_node("executor", executor_probe)
        builder.add_edge(START, "orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            lambda state: state["next_role"],
            {"human": "human", "executor": "executor"},
        )
        builder.add_edge("human", "orchestrator")
        builder.add_edge("executor", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {
            "configurable": {
                "thread_id": "python-probe" if python_variant else "notebook-probe"
            }
        }

        first = graph.invoke(self.initial_state(python_variant), config)
        self.assertIn("__interrupt__", first)
        refused = graph.invoke(Command(resume="No. Ask me again."), config)
        self.assertIn("__interrupt__", refused)
        refused_state = graph.get_state(config).values
        self.assertEqual(refused_state[counter_name], 0)
        self.assertEqual(refused_state["dialogue"][-1]["role"], "orchestrator")
        self.assertIn("did not run", refused_state["dialogue"][-1]["text"])

        accepted = graph.invoke(Command(resume="Yes. Run it now."), config)
        self.assertEqual(accepted[counter_name], 1)
        human_messages = [
            message for message in accepted["dialogue"] if message["role"] == "human"
        ]
        self.assertEqual(len(human_messages), 2)
        self.assertEqual(accepted["dialogue"][-1]["role"], "executor")

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"task6.ipynb cell {index + 1}", "exec")

    def test_both_variants_have_checkpointed_human_nodes(self):
        joined = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertIn("def human_node(state: State):", joined)
        self.assertIn("def python_human(state: State):", joined)
        self.assertEqual(joined.count("compile(checkpointer="), 2)
        self.assertGreaterEqual(joined.count("Command(resume="), 2)
        self.assertGreaterEqual(joined.count('add_node("human"'), 2)
        self.assertEqual(joined.count("executor approval cannot be bypassed"), 2)

    def test_notebook_graph_refuses_then_accepts(self):
        self.exercise_refuse_then_accept(python_variant=False)

    def test_python_graph_refuses_then_accepts(self):
        self.exercise_refuse_then_accept(python_variant=True)


if __name__ == "__main__":
    unittest.main()
