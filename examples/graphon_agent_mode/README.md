# Graphon Agent Mode Example

This example runs an agent-style Graphon workflow:

`start -> loop -> output`

Inside the loop, the workflow uses:

- an `LLM` node to decide the next action as structured JSON
- an `Output` node to log that decision for the current round
- an `IfElse` node to branch between answering directly and calling a tool
- a `Tool` node that can read, write, and delete files or run bash commands
- loop-owned variables so each tool result is carried into the next round

The loop is capped at 100 rounds. If the model chooses to answer directly, the
workflow stores that answer on the loop node, breaks via `LoopEnd`, and then the
final `Output` node exposes the result.

## Files

- `workflow.py`: runnable example script
- `support.py`: Slim runtime and local environment helpers
- `workspace_tools.py`: tool runtime for workspace file and bash operations
- `.env.example`: template configuration
- `.env`: local configuration file for this example only

## Run

1. Change into this directory:

```bash
cd examples/graphon_agent_mode
```

2. Copy the template:

```bash
cp .env.example .env
```

3. Fill in the required values in `.env`.

The example now defaults to longer runtime ceilings so slower model calls do not
fail as quickly:

- `SLIM_PYTHON_ENV_INIT_TIMEOUT=300`
- `SLIM_MAX_EXECUTION_TIMEOUT=1800`
- `SLIM_OPENAI_CONNECT_TIMEOUT_SECONDS=30`
- `SLIM_OPENAI_WRITE_TIMEOUT_SECONDS=30`
- `AGENT_BASH_COMMAND_TIMEOUT_SECONDS=120`

Raise them further in `.env` if your provider or local environment still needs
more time.

4. Run the example:

```bash
python3 workflow.py "Inspect README.md and summarize how this repository is organized."
```

Planner `<think>...</think>` output and round-by-round agent decisions are
written to stderr.
The final answer is written to stdout.

## Notes

- The agent tools operate relative to the repository root.
- `write_file` overwrites the target file with the provided content.
- `run_bash` executes in the repository root and uses
  `AGENT_BASH_COMMAND_TIMEOUT_SECONDS`.
- The round log output node is only for observability; it is not referenced by
  the LLM prompt, so it does not feed back into the model context.
- Planner reasoning is streamed from `<think>...</think>` blocks to stderr only.
  The stored planner output strips those blocks before the JSON decision is
  parsed and passed to the rest of the workflow.
- Like the other example, `workflow.py` falls back to the local `src/`
  directory when `graphon` is not installed and can re-exec with the repository
  `.venv` if needed.
