# SPICA

## Predictive Semantic Transport

The original full-vector JEPA path remains available as the T0 control. The transport path consumes raw sketches, uses a trainable CLIP visual tower up to its pre-projection hidden state, applies a frozen photo-CLIP projection for `z0`, and predicts residual or tangent/geodesic transport. Text is optional loss-only classification supervision and is never an inference input.

Train with Hydra (the experiment files under `configs/experiments/` document the ablations):

```bash
PYTHONPATH=src python -m spica.train_transport --config-name train_transport \
  transport_mode=tangent K=1 rho_max_degrees=15 max_steps=5400
```

Evaluate a saved checkpoint with sketch-only inference:

```bash
PYTHONPATH=src python -m spica.evaluate_transport --config-name evaluate_transport \
  checkpoint_path=outputs/experiments/<run>/<timestamp>/checkpoints/transport_step73.pt
```

Regenerate the versioned report and plots:

```bash
PYTHONPATH=src python scripts/summarize_transport.py --date 2026-09-02
```

The current measured summary is `outputs/research_summary_transport_2026-09-02.md`; official-test values in it are diagnostic only, while pseudo-unseen validation selects configurations.
