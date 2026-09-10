#!/usr/bin/env python3
"""CPU-only smoke gate for the production TQMP objective."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from check_fusion_qmp_cpu import active, batch, tiny_model, _loss_parity, ARCHIVE_LOSS  # noqa: E402
from diagnose_fusion_mp_tq import _losses as diagnostic_losses  # noqa: E402
from spica.coupled_predictive_losses import coupled_region_loss  # noqa: E402
from spica.data.coupled_training import _arm_protocol  # noqa: E402
import spica.train_coupled_predictive as trainer  # noqa: E402

CAMPAIGN = "coupled_predictive_fusion_tqmp_v1"
OBJECTIVE = "multi_positive_three_head_contrastive"
TQ_RECEIPT = ROOT / "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_verified.json"
SIG_RECEIPT = ROOT / "outputs/fusion_sigreg_diagnostic_20260909T055900Z/diagnostic_verified.json"
ARGS = batch()
KW = dict(lambda_sig=0.0, sigreg=None)


def read_tq_receipt() -> tuple[dict[str, Any], float, float]:
    payload = json.loads(TQ_RECEIPT.read_text(encoding="utf-8"))
    assert payload.get("status") == "PASS" and payload.get("scope", "").startswith("RAW_GRADIENT")
    selection = payload.get("selection", {})
    lt, lq = selection.get("lambda_t"), selection.get("lambda_q")
    assert isinstance(lt, (int, float)) and isinstance(lq, (int, float)) and lt > 0 and lq > 0
    assert selection.get("rho_t") == 0.1 and selection.get("rho_q") == 0.1
    assert selection.get("binding_scope") == "pooled_head"
    return payload, float(lt), float(lq)


def check_receipts() -> dict[str, Any]:
    payload, lt, lq = read_tq_receipt()
    validated = trainer._validate_f2_tqmp_diagnostic(TQ_RECEIPT)
    assert lt == 0.17877235601108843 and lq == 0.023774345199536452
    assert validated["lambda_mp_t"] == lt and validated["lambda_mp_q"] == lq
    assert len(validated["input_sha256"]) == 14
    assert validated["input_sha256"] == trainer.F2_TQMP_INPUT_SHA256
    raw = json.loads((ROOT / validated["raw_diagnostic_path"]).read_text(encoding="utf-8"))
    assert validated["source_snapshot_hash"] == trainer.F2_TQMP_SOURCE_SNAPSHOT_SHA256
    assert validated["data_identity"] == raw["data_identity"]
    assert validated["initialization_hashes"] == raw["initialization_hashes"]
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory) / "fake.json"
        fake.write_text(json.dumps(payload), encoding="utf-8")
        for candidate in (fake, ROOT / "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_result.json", SIG_RECEIPT):
            try:
                trainer._validate_f2_tqmp_diagnostic(candidate)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"invalid TQMP receipt accepted: {candidate}")
    return {"tq_receipt": str(TQ_RECEIPT), "lambda_t": lt, "lambda_q": lq,
            "exact_14_file_manifest_source_data_init_identities": True,
            "fake_wrong_sig_rejected": True}


def check_route() -> dict[str, Any]:
    expected = {"architecture": "predictive_fusion_v2", "method_version": CAMPAIGN,
                "positive_pool": "full", "main_photo_objective": OBJECTIVE}
    assert _arm_protocol("F2_TQMP", CAMPAIGN, TQ_RECEIPT) == expected
    original = trainer._device
    calls = 0
    def forbidden(value: str) -> torch.device:
        nonlocal calls
        calls += 1
        raise AssertionError(f"device reached for invalid route: {value}")
    trainer._device = forbidden
    try:
        for campaign, diagnostic in (("wrong", None), (CAMPAIGN, object()), (CAMPAIGN, ROOT / "missing.json")):
            args = argparse.Namespace(arm="F2_TQMP", campaign_id=campaign, diagnostic=diagnostic,
                                     max_steps=2, smoke=True, device="cuda")
            try:
                trainer._train_impl(args, ROOT / "never-created-tqmp-output")
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("invalid TQMP route accepted")
    finally:
        trainer._device = original
    assert calls == 0
    assert trainer._evaluation_query("F2_TQMP") == "q"
    selected: list[str | None] = []
    old_factory = trainer._eval_factory
    trainer._eval_factory = lambda *args, **kwargs: selected.append(kwargs.get("query")) or (lambda: {})
    try:
        trainer._make_eval_probe({}, None, None, torch.device("cpu"), "F2_TQMP")
    finally:
        trainer._eval_factory = old_factory
    assert selected == ["q"]
    return {"protocol": expected, "invalid_arm_campaign_diagnostic_and_path_rejected_before_device": True,
            "evaluation_query": "q", "eval_probe_route": True}


def prod(model: torch.nn.Module, objective: str, lt: float = 0.0, lq: float = 0.0) -> dict[str, Tensor]:
    return coupled_region_loss(model, *ARGS, **KW, main_photo_objective=objective,
                               lambda_mp_t=lt, lambda_mp_q=lq)


def grad(loss: Tensor, model: torch.nn.Module) -> dict[str, Tensor]:
    params = active(model)
    values = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True, retain_graph=True)
    return {name: (value.detach() if value is not None else torch.zeros_like(params[name]))
            for (name, _), value in zip(params.items(), values)}


def direct_mp(query: Tensor, bank: Tensor, bank_labels: Tensor, labels: Tensor) -> Tensor:
    logits = F.normalize(query, dim=-1) @ F.normalize(bank, dim=-1).T
    positives = bank_labels[None, :] == labels[:, None]
    assert bool(positives.any(dim=1).all())
    return (-(F.log_softmax(logits / 0.07, dim=-1) * positives).sum(dim=1)
            / positives.sum(dim=1)).mean()

def direct_tq(model: torch.nn.Module) -> tuple[Tensor, Tensor]:
    clean, corrupted, photos, _, _, labels, photo_labels, _ = ARGS
    output = model(torch.cat((clean, corrupted), dim=0))
    bank = model.encode_photo(photos)
    n = clean.shape[0]
    return (
        0.5 * (direct_mp(output.mu_t[:n], bank, photo_labels, labels)
                + direct_mp(output.mu_t[n:], bank, photo_labels, labels)),
        0.5 * (direct_mp(output.q[:n], bank, photo_labels, labels)
                + direct_mp(output.q[n:], bank, photo_labels, labels)),
    )

def check_objective() -> dict[str, Any]:
    _, lt, lq = read_tq_receipt()
    base_model, new_model, diag_model = tiny_model(42), tiny_model(42), tiny_model(42)
    base = prod(base_model, "multi_positive_supervised_contrastive")
    new = prod(new_model, OBJECTIVE, lt, lq)
    diag = diagnostic_losses(diag_model, dict(zip(
        ("clean", "corrupted", "photos", "positive_indices", "negative_indices", "labels", "photo_labels", "photo_ids"), ARGS, strict=True)))
    direct_t, direct_q = direct_tq(new_model)
    oracle = diag["base"] + lt * direct_t + lq * direct_q
    torch.testing.assert_close(new["total"], oracle, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(new["clean_rank_t_mp"] + new["masked_rank_t_mp"], 2 * direct_t, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(new["clean_rank_q_mp"] + new["masked_rank_q_mp"], 2 * direct_q, rtol=2e-5, atol=2e-6)
    for key in base:
        if key != "total":
            assert torch.equal(base[key], new[key]), key
    zero = prod(tiny_model(42), OBJECTIVE)
    for key in base:
        assert torch.equal(base[key], zero[key]), f"lambda-zero:{key}"
    ng, bg = grad(new["total"], new_model), grad(diag["base"], diag_model)
    tg, qg = grad(direct_t, new_model), grad(direct_q, new_model)
    for name in ng:
        expected = bg[name] + lt * tg[name] + lq * qg[name]
        torch.testing.assert_close(ng[name], expected, rtol=3e-5, atol=3e-6)
    assert {"clean_rank_i", "masked_rank_i", "clean_rank_t_mp", "masked_rank_t_mp",
            "clean_rank_q_mp", "masked_rank_q_mp"} <= new.keys()
    return {"scalar_combined_oracle": True, "gradient_combined_oracle": True,
            "unchanged_terms_exact": True, "lambda_zero_base_exact": True, "term_count": len(new)}


def check_routing_and_step() -> dict[str, Any]:
    model = tiny_model(43)
    terms = prod(model, OBJECTIVE, *read_tq_receipt()[1:])
    tgrad = grad(terms["clean_rank_t_mp"] + terms["masked_rank_t_mp"], model)
    qgrad = grad(terms["clean_rank_q_mp"] + terms["masked_rank_q_mp"], model)
    assert any(float(v.abs().sum()) > 0 for n, v in tgrad.items() if n.startswith(("student_visual.", "predictor.")))
    assert any(float(v.abs().sum()) > 0 for n, v in qgrad.items() if n.startswith(("student_visual.", "pooled_head.")))
    assert all(float(v.abs().sum()) == 0 for n, v in tgrad.items() if n.startswith("pooled_head."))
    assert all(float(v.abs().sum()) == 0 for n, v in qgrad.items() if n.startswith("predictor."))
    frozen = {n: p.detach().clone() for n, p in model.original_clip.state_dict().items()}
    optimizer = trainer._make_optimizer(model)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prod(model, OBJECTIVE, *read_tq_receipt()[1:])["total"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0 for p in active(model).values())
        optimizer.step()
    state = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("original_clip.")}
    restored = tiny_model(43)
    missing, unexpected = restored.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all(n.startswith("original_clip.") for n in missing)
    assert all(torch.equal(state[n], restored.state_dict()[n]) for n in state)
    assert all(torch.equal(v, model.original_clip.state_dict()[n]) for n, v in frozen.items())
    return {"mu_t_routes_predictor_not_pooled": True, "q_routes_pooled_not_predictor": True,
            "optimizer_steps": 2, "reload_exact": True, "teacher_unchanged": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "outputs" / f"fusion_tqmp_cpu_worker_{stamp}").resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    checks: list[dict[str, Any]] = []
    for name, fn in (("receipts", check_receipts), ("route", check_route), ("objective", check_objective), ("routing_step", check_routing_and_step)):
        try:
            checks.append({"name": name, "status": "PASS", "details": fn()})
        except Exception as error:  # noqa: BLE001
            checks.append({"name": name, "status": "FAIL", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            break
    try:
        _loss_parity(ARCHIVE_LOSS, "spica.fusion_tqmp_legacy_archive_loss")
        checks.append({"name": "legacy_parity", "status": "PASS"})
    except Exception as error:  # noqa: BLE001
        checks.append({"name": "legacy_parity", "status": "FAIL", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
    failed = [row for row in checks if row["status"] != "PASS"]
    result = {"schema_version": 1, "status": "FAIL" if failed else "PASS", "verified": not failed,
              "gate": "fusion_tqmp_cpu", "device": "cpu", "pretrained_weights": False,
              "network": False, "campaign_training": False, "checks": checks,
              "summary": {"pass": len(checks) - len(failed), "fail": len(failed), "skip": 0},
              "component_sha256": {p: trainer._sha256_file(ROOT / p) for p in (
                  "scripts/check_fusion_tqmp_cpu.py", "scripts/check_fusion_qmp_cpu.py",
                  "scripts/diagnose_fusion_mp_tq.py", "scripts/run_fusion_multipositive.py",
                  "src/spica/coupled_predictive_losses.py", "src/spica/data/coupled_training.py",
                  "src/spica/train_coupled_predictive.py", "src/spica/models/coupled_predictive.py",
                  "src/spica/evaluation/coupled_predictive.py")}}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
