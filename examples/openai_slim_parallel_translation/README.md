# OpenAI Slim Parallel Translation Example

A fan-out / fan-in workflow:

`start -> 3 llm -> end`

The `start` node takes `content`. Three LLM nodes translate it into Chinese,
English, and Japanese in parallel. The `end` node returns all three
translations.

## What You Need

- `workflow.py`: runnable example
- `.env.example`: template settings
- `.env`: your local copy of the template

## Run

```bash
cd examples/openai_slim_parallel_translation
cp .env.example .env
python3 workflow.py "Graph execution is a coordination problem."
```

Fill in `.env` before running. The script reads `.env` from this directory.

## Useful Flags

```bash
python3 workflow.py --no-stream "Graph execution is a coordination problem."
```

By default, the example streams each translation as it becomes available, then
prints the final structured outputs.
