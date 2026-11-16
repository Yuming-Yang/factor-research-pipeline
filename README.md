# factor-research-pipeline

A factor research pipeline from signal to portfolio in which every step carries its own evidence against overfitting. It is checked on synthetic data with a known signal before it is run on real data.

Work in progress. See [SPEC.md](SPEC.md) for scope, acceptance criteria and milestones.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
```
