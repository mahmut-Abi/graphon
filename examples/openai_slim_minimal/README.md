# OpenAI Slim Minimal Example

A tiny Graphon workflow:

`start -> llm -> output`

## What You Need

- `workflow.py`: runnable example
- `.env.example`: template settings
- `.env`: your local copy of the template

## Run

```bash
cd examples/openai_slim_minimal
cp .env.example .env
python3 workflow.py
```

Fill in `.env` before running. The script reads `.env` from this directory.

## Custom Prompt

```bash
python3 workflow.py "Explain graph sparsity in one sentence."
```

The example streams text to stdout as it arrives. If nothing is streamed, it prints the final answer at the end.
