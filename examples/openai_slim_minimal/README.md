# OpenAI Slim Minimal Example

This example runs a minimal Graphon workflow:

`start -> llm -> output`

## Files

- `workflow.py`: runnable example script
- `.env.example`: template configuration
- `.env`: local configuration file for this example only

## Run

1. Change into this directory:

```bash
cd examples/openai_slim_minimal
```

2. Copy the template:

```bash
cp .env.example .env
```

3. Fill in the required values in `.env`.

4. Run the example:

```bash
python3 workflow.py
```

The CLI streams LLM text to stdout as chunks arrive.

You can also pass a custom prompt:

```bash
python3 workflow.py "Explain graph sparsity in one sentence."
```

## Notes

- `workflow.py` first tries to import an installed `graphon` package.
- If `graphon` is not installed, it falls back to the local repository `src/`
  directory automatically. That lets you run the example directly from this
  checkout without setting `PYTHONPATH`.
- If your current interpreter is missing runtime dependencies but the repository
  `.venv` exists, `workflow.py` will re-exec itself with that local virtualenv
  interpreter automatically.
- No `slim` executable is bundled in this example directory. Provide
  `dify-plugin-daemon-slim` via `PATH` or set `SLIM_BINARY_PATH` in `.env` if
  you keep it elsewhere.
- Path-like variables in `.env` are resolved relative to this example
  directory, not relative to your shell's current working directory.
- By default, `SLIM_PLUGIN_FOLDER` resolves to the repository-root
  `.slim/plugins` cache. That keeps generated plugin files out of this example
  directory while still letting you run `python3 workflow.py` from here.
