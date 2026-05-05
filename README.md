# CUDA-Debugger-Bench: Code Package

NeurIPS 2026 E&D Track Submission - companion code to the CUDA-Debugger-Bench
benchmark dataset.

## What's here

- `harness/` - evaluation harness (Python)
- `analysis/` - result aggregation + table rendering
- `configs/` - 5 evaluation experiment axes (iter, repeated, feedback, history, fewshot)
- `tests/` - smoke + unit tests
- `examples/` - reproduce paper tables
- `benchmark/` - pointer to companion HuggingFace dataset
- `docs/` - protocol, performance measurement, reproduction, anonymization notes

## Quick start (5 min smoke test)

```bash
./setup_env.sh
source .venv/bin/activate
pytest tests/test_smoke.py -v
```

## Reproduce paper tables

See `docs/REPRODUCE.md`.

## Authors

Anonymous Authors (NeurIPS 2026 E&D Track submission). Will be deanonymized at
acceptance.

## License

CC BY 4.0. See `LICENSE`.
