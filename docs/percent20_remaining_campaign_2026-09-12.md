# Official 20% — remaining TU → QuickDraw, 2026-09-12

User explicitly authorized starting the remaining two datasets after Sketchy finished training/tests but failed the final source guard. No Sketchy retraining, checkpoint resume, or rewriting its raw failure status.

Original root: `outputs/percent20_official_execution_20260912T133000Z/`; Sketchy4446, all five official test rows present and exact online/local. Audit worker mistakenly created `scripts/audit_sda.py` despite ignored-output-only instructions; guard failed at14:45:35Z. Parent relocated it to ignored storage review; source restored exactly to `155d94e4f4b4207cd5cbb8075a8cbf5f88aed9279c3fb9c39a8e425474ac65e3`. Incident: `outputs/storage_migration_review_20260912/campaign_source_incident.json`. This was an assistant operational error, not training instability. Preserve `FAILED_NO_RETRY`.

New runner option `--datasets tuberlin_220_30 quickdraw_80_30` selects only fresh runs/order; omitted option retains all three. It does not change dataset definitions, trainer, model, losses, evaluation, gate checks, or numerical components. Duplicate/unknown datasets rejected. Manifest records actual order.

- TU:1189 updates, warmup59; tests238/476/714/952/1189.
- QuickDraw:18229 updates, warmup911; tests3646/7292/10938/14584/18229.
- Fresh seed42/B32 MP-Q, mainMP(mu_I), inferenceq, no SREF/QMP.
- Same six online test curves; commit=True; full clean+9 each boundary, final full details, intermediates metrics-only; no selection, no retry/resume/newseed/search.

Gate root `outputs/percent20_remaining_gate_20260912/`:32 targeted CPU tests including synthetic full orchestration of exactly TU→Q, no Sketchy. Rebind only modified runner component to fresh CPU receipt; retain unchanged original two-update CUDA/restore/retrieval evidence under `outputs/percent20_gate_20260912/`. No additional CUDA smoke is claimed or needed for unchanged numerical components. Full gate validation required before launch.

Planned fresh root `outputs/percent20_remaining_execution_20260912/`; adjacent launch receipt and runtime/PIDs authoritative; inspect before launch, no duplicates. Freeze all nonignored source/config/docs while active, including untracked scripts. Ignored operational records only. Do not remove any gate-bound evidence. Datasets/cache and current gates remain on NVMe; historical outputs archived under `sda/DoAn/spica_archive_20260912/outputs/` with original paths retained. Avoid sleep/reboot. Raw completion remains unverified official replay scope.
