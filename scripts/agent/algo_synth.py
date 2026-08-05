"""PROC-1: process→algorithm compiler — turn a *surviving* finding into a registered algorithm.

The discovery loop used to stop at a hypothesis: `it_discovery.py` turns IT-engine findings
into queue entries and the runner scores them, but nothing ever produced a tradeable
`MicrostructureAlgorithm`, so the roster stayed hand-authored and frozen. This module closes
that loop — and its most important behaviour is **refusing** to close it.

A compiler that emits code for whatever it is handed is the Q4 failure mode (FINDINGS §4.6)
with a `@register` decorator on it: five hand-authored "winners" became false discoveries
because nothing structurally stopped an unearned result from becoming a strategy. So
admission is a gate, not a formality:

  * `informative` must be True, `z_null` must clear the PROC-12 null threshold, and `bh_q`
    must clear the PROC-13 FDR alpha. **Both thresholds are imported** — `it_engine.toml`
    via `load_null_config()` and `processes.fdr.DEFAULT_FDR_ALPHA` — never written here
    (guardrail: gates imported, not invented).
  * **Polarity must be explicit.** Mutual information is non-negative and direction-blind:
    it says *there is information*, never *which way to trade*. A finding without a signed
    polarity is not a rule, and guessing one manufactures an edge the measurement never
    made. Refuse.
  * Feature and regime names are rendered into source, so anything that is not a plain
    Python identifier is rejected before it reaches a template.
  * A generated algorithm may never take a registered algorithm's name, and never silently
    overwrites a file on disk.

What it emits (deterministically — same finding ⇒ byte-identical source, so the diff is
reviewable and provenance is reproducible):

    kind="threshold"     rolling z-score of the finding's column, signed by polarity
    kind="regime_gated"  the same rule, muted outside the finding's regime bucket
    kind="combiner"      NotImplementedError — a multi-column combiner is PROC-3's job,
                         and faking it here would emit a rule no process validated

Provenance (MI bits, null z, BH q, horizon, label, symbol, run_id, git_sha, the finding's
own timestamp) is written into the generated docstring. The emitted unit still has to earn
its maturity through the ladder — synthesis produces a PRELIM algorithm, never a promotion.

Spec: `docs/specs/process_layer.md` §1. Contract: `docs/contracts/algorithm.md`.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from it_engine.null_calibration import load_null_config          # noqa: E402
from processes import fdr as _fdr                                # noqa: E402

#: Where generated algorithms land (git-tracked: generated code is reviewed, not trusted).
DEFAULT_OUT_DIR = _SCRIPTS / "algorithms" / "generated"

#: A column name we are willing to render into source: plain identifier, not dunder.
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

#: Rolling window (ticks/bars) for the z-score rule. Not a gate threshold — a template
#: constant, recorded in the generated docstring so it is visible in review.
Z_WINDOW = 128


def null_z_threshold() -> float:
    """PROC-12 significance floor, from `config/it_engine.toml`."""
    return float(load_null_config()["null_z_threshold"])


def min_bits_above_null() -> float:
    """PROC-12 effect-size floor, from `config/it_engine.toml`."""
    return float(load_null_config()["i_min"])


def fdr_alpha() -> float:
    """PROC-13 BH-FDR alpha (the process layer's single alpha)."""
    return float(_fdr.DEFAULT_FDR_ALPHA)


def existing_algorithm_names() -> set[str]:
    """Names the registry answers to that a generated unit must never take.

    PROC-1 modules carry a ``PROC1_GENERATED`` marker and are excluded: re-rendering a
    finding whose module is already imported is the *same* unit, and the file-overwrite
    flag governs that. The rule protects hand-written algorithms from being shadowed.
    """
    from algorithms import registry
    try:                                     # populate the registry if the package lazy-loads
        import algorithms  # noqa: F401
    except Exception:                        # pragma: no cover - import side effects only
        pass
    names = set()
    for name in registry.list_algorithms():
        cls = registry._REGISTRY.get(name)
        module = sys.modules.get(getattr(cls, "__module__", ""), None)
        if getattr(module, "PROC1_GENERATED", False):
            continue
        names.add(name)
    return names


def _valid_name(value: str) -> bool:
    return bool(value) and bool(_IDENT.match(value)) and not value.startswith("__")


@dataclass(frozen=True)
class PromotedFinding:
    """One surface row, validated into the tuple a rule can be compiled from.

    Mirrors `processes.surface.SURFACE_COLUMNS` plus `polarity`, which the surface itself
    cannot supply (MI is unsigned) and which must therefore be attached by whatever
    directional statistic promoted the finding.
    """

    combo_id: str
    horizon: str
    label_def: str
    regime_var: str
    regime_bin: str
    mi_bits: Optional[float]
    z_null: Optional[float]
    p_value: Optional[float]
    bh_q: Optional[float]
    informative: bool
    n: Optional[int]
    symbol: str
    timeframe: str
    process: str
    run_id: str
    git_sha: str
    generated_at: str
    polarity: Optional[int]
    kind: str = "threshold"

    # ── construction ─────────────────────────────────────────────────────────────
    @classmethod
    def from_surface_row(cls, row: dict, kind: Optional[str] = None) -> "PromotedFinding":
        regime_var = (row.get("regime_var") or "").strip()
        regime_bin = (row.get("regime_bin") or "all").strip()
        gated = bool(regime_var) and regime_bin not in ("", "all")
        polarity = row.get("polarity")
        return cls(
            combo_id=str(row.get("combo_id") or ""),
            horizon=str(row.get("horizon") or ""),
            label_def=str(row.get("label_def") or ""),
            regime_var=regime_var,
            regime_bin=regime_bin,
            mi_bits=row.get("mi_bits"),
            z_null=row.get("z_null"),
            p_value=row.get("p_value"),
            bh_q=row.get("bh_q"),
            informative=bool(row.get("informative")),
            n=row.get("n"),
            symbol=str(row.get("symbol") or ""),
            timeframe=str(row.get("timeframe") or ""),
            process=str(row.get("process") or ""),
            run_id=str(row.get("run_id") or ""),
            git_sha=str(row.get("git_sha") or ""),
            generated_at=str(row.get("generated_at") or ""),
            polarity=None if polarity is None else int(polarity),
            kind=kind or ("regime_gated" if gated else "threshold"),
        )

    # ── identity ─────────────────────────────────────────────────────────────────
    @property
    def algorithm_name(self) -> str:
        """Deterministic, collision-resistant name carrying its own coordinates."""
        parts = ["gen", self.combo_id, f"h{self.horizon}"]
        if self.kind == "regime_gated":
            parts.append(f"{self.regime_var}_{self.regime_bin}")
        parts.append("pos" if (self.polarity or 0) >= 0 else "neg")
        slug = "_".join(p for p in parts if p)
        return re.sub(r"[^A-Za-z0-9_]", "_", slug).lower()

    @property
    def required_columns(self) -> list[str]:
        cols = [self.combo_id]
        if self.kind == "regime_gated":
            cols.append(self.regime_var)
        return cols

    # ── the gate ─────────────────────────────────────────────────────────────────
    def is_compilable(self) -> tuple[bool, list[str]]:
        """(admitted, reasons-for-refusal). Every threshold here is imported."""
        reasons: list[str] = []

        if not _valid_name(self.combo_id):
            reasons.append(
                f"feature name {self.combo_id!r} is not a plain column identifier — "
                "refusing to render it into source")
        if self.kind == "regime_gated" and not _valid_name(self.regime_var):
            reasons.append(
                f"regime variable name {self.regime_var!r} is not a plain identifier")

        if not self.informative:
            reasons.append("finding is not marked informative by its process")

        if self.z_null is None:
            reasons.append("no null-calibration z (PROC-12) — an uncalibrated finding "
                           "cannot be distinguished from the estimator's noise floor")
        elif float(self.z_null) < null_z_threshold():
            reasons.append(f"null-calibration z {float(self.z_null):.2f} < "
                           f"{null_z_threshold():.2f} (PROC-12 threshold)")

        if self.bh_q is None:
            reasons.append("no BH q-value — the finding was never fdr-corrected (PROC-13)")
        elif float(self.bh_q) > fdr_alpha():
            reasons.append(f"fdr q {float(self.bh_q):.3f} > alpha {fdr_alpha():.3f}")

        if not self.polarity:
            reasons.append(
                "no explicit polarity — mutual information is unsigned and carries no "
                "direction; a trading rule needs one and this compiler will not guess it")
        elif int(self.polarity) not in (-1, 1):
            reasons.append(f"polarity {self.polarity!r} must be -1 or +1")

        return (not reasons), reasons


# ── rendering ────────────────────────────────────────────────────────────────────
_TEMPLATE = '''"""GENERATED by PROC-1 (`agent/algo_synth.py`) — do not hand-edit.

{title}

Rule (kind={kind}):
    z_t   = (x_t - mean(x, {window})) / std(x, {window})     x = `{feature}`
    signal = {polarity:+d} * clip(z_t, -3, 3){gate_doc}

Provenance — the finding this was compiled from:
    process     {process}
    combo       {feature}
    horizon     {horizon}          label       {label_def}
    regime      {regime_desc}
    MI          {mi_bits} bits     null z      {z_null}
    p           {p_value}          BH q        {bh_q}
    n           {n}                symbol      {symbol} @ {timeframe}
    run_id      {run_id}           git_sha     {git_sha}
    found_at    {generated_at}

Maturity: PRELIM. Synthesis is not promotion — this unit still has to earn BETA/PROVEN
through the imported gates (`docs/contracts/algorithm.md`).
"""

from __future__ import annotations

import numpy as np

from algorithms.base import AlgorithmFeature, MicrostructureAlgorithm
from algorithms.registry import register

WINDOW = {window}
POLARITY = {polarity:+d}
FEATURE = "{feature}"{gate_consts}


@register
class {class_name}(MicrostructureAlgorithm):
    """Rolling-z rule compiled from a null-calibrated, FDR-passed finding (PROC-1)."""

    def name(self) -> str:
        return "{algo_name}"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [AlgorithmFeature(name="{out_name}", warmup=WINDOW,
                                 description="compiled signal for {feature}")]

    def required_columns(self) -> list[str]:
        return {required!r}

    def reset(self) -> None:
        self._buf: list[float] = []

    def __init__(self) -> None:
        self.reset()

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan = {{"{out_name}": float("nan")}}
        x = tick.get(FEATURE)
        if x is None or not np.isfinite(x):
            return nan
{gate_step}
        self._buf.append(float(x))
        if len(self._buf) > WINDOW:
            self._buf.pop(0)
        if len(self._buf) < WINDOW:
            return nan
        arr = np.asarray(self._buf, dtype=np.float64)
        sd = float(arr.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            return nan
        z = (float(x) - float(arr.mean())) / sd
        return {{"{out_name}": float(POLARITY * np.clip(z, -3.0, 3.0))}}


#: Marks this module as PROC-1 output (see `existing_algorithm_names`).
PROC1_GENERATED = True

#: Handle used by the PROC-1 test harness to instantiate without the registry.
ALGORITHM_CLASS = {class_name}
'''

_GATE_CONSTS = '\nREGIME_COL = "{regime_var}"\nREGIME_BIN = "{regime_bin}"'

_GATE_STEP = """        g = tick.get(REGIME_COL)
        if g is None or not np.isfinite(g):
            return nan
"""


def _class_name(algo_name: str) -> str:
    return "".join(p.capitalize() for p in algo_name.split("_") if p) or "GeneratedAlgorithm"


def render_source(finding: PromotedFinding) -> str:
    """Deterministic source for `finding`. Raises if it is not admissible."""
    if finding.kind == "combiner":
        raise NotImplementedError(
            "combiner findings are not compilable here: a multi-column combination rule is "
            "PROC-3's synergy-aware selection, and inventing one would emit a rule no "
            "process validated")
    if finding.kind not in ("threshold", "regime_gated"):
        raise NotImplementedError(f"unsupported finding kind {finding.kind!r}")

    ok, reasons = finding.is_compilable()
    if not ok:
        raise ValueError("finding is not compilable:\n  - " + "\n  - ".join(reasons))

    algo_name = finding.algorithm_name
    gated = finding.kind == "regime_gated"
    regime_desc = (f"{finding.regime_var} = {finding.regime_bin}" if gated
                   else "unconditional (all)")
    return _TEMPLATE.format(
        title=f"Compiled from a {finding.process} finding on `{finding.combo_id}`.",
        kind=finding.kind,
        window=Z_WINDOW,
        feature=finding.combo_id,
        polarity=int(finding.polarity),
        gate_doc=("  , muted while the regime column is unavailable" if gated else ""),
        process=finding.process,
        horizon=finding.horizon,
        label_def=finding.label_def,
        regime_desc=regime_desc,
        mi_bits=finding.mi_bits,
        z_null=finding.z_null,
        p_value=finding.p_value,
        bh_q=finding.bh_q,
        n=finding.n,
        symbol=finding.symbol,
        timeframe=finding.timeframe,
        run_id=finding.run_id,
        git_sha=finding.git_sha,
        generated_at=finding.generated_at,
        class_name=_class_name(algo_name),
        algo_name=algo_name,
        out_name=f"alg_{algo_name}_signal",
        required=finding.required_columns,
        gate_consts=(_GATE_CONSTS.format(regime_var=finding.regime_var,
                                         regime_bin=finding.regime_bin) if gated else ""),
        gate_step=(_GATE_STEP if gated else ""),
    )


def synthesize(finding: PromotedFinding, out_dir: Optional[Path] = None,
               overwrite: bool = False) -> Path:
    """Compile `finding` into a registered algorithm module; return the path written.

    Refuses (raises) rather than emitting anything when the finding fails admission, when
    its name would shadow a registered algorithm, or when the target file already exists
    and `overwrite` was not requested explicitly.
    """
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    source = render_source(finding)                       # validates before any I/O

    if finding.algorithm_name in existing_algorithm_names():
        raise ValueError(
            f"algorithm {finding.algorithm_name!r} is already registered — a generated unit "
            "must never shadow an existing one")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{finding.algorithm_name}.py"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass overwrite=True to replace it (generated code is reviewed, "
            "so clobbering it is an explicit act)")
    path.write_text(source)
    return path
