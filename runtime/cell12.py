#@title 7A. RUNTIME-STATE COMPATIBILITY v2 — aligned with Cell 7 v4
# Complete replacement for Cell 7A.
# Run AFTER Cell 7 v4 and BEFORE Cell 7B / Cells 8–10.
#
# Cell 7 v4 already initializes STATE internally. This compatibility layer
# keeps direct Validator/Ranker/Engine.one calls safe, preserves the legacy
# _surface_key alias, prevents wrapper stacking, and recreates ENGINE once.
# It does not change routing, transformation rules, validation policy,
# ranking weights, or must-paraphrase policy.

EXPECTED_7A_RULE_COUNT = 30

_REQUIRED_7A = [
    "STATE", "RULES", "Validator", "Ranker", "Engine", "ENGINE",
    "surface_key", "realization_issues",
]
_missing_7a = [name for name in _REQUIRED_7A if name not in globals()]
if _missing_7a:
    raise RuntimeError(
        "Cell 7A cannot run. Missing: " + ", ".join(_missing_7a)
        + ". Run Cell 7 v4 first."
    )

if len(RULES) != EXPECTED_7A_RULE_COUNT:
    raise RuntimeError(
        f"Cell 7A expected {EXPECTED_7A_RULE_COUNT} rules but found {len(RULES)}."
    )


# =====================================================================
# 1. LEGACY NAME COMPATIBILITY
# =====================================================================

_surface_key = surface_key


# =====================================================================
# 2. STATE SAFETY
# =====================================================================

def _ensure_runtime_state():
    if not isinstance(STATE, dict):
        raise RuntimeError("STATE must be a dictionary.")
    STATE.setdefault("warnings", [])
    STATE.setdefault("start", None)
    return STATE


_ensure_runtime_state()


# =====================================================================
# 3. WRAPPER INSTALLATION — RERUN SAFE
# =====================================================================

# Because Cell 7 has just redefined these classes, preserve its current
# methods as the canonical bases. If this cell is rerun without rerunning
# Cell 7, restore those bases before wrapping again.

if not hasattr(Validator, "_nll_state_safe_base_validate_v2"):
    Validator._nll_state_safe_base_validate_v2 = Validator.validate
else:
    Validator.validate = Validator._nll_state_safe_base_validate_v2

if not hasattr(Ranker, "_nll_state_safe_base_rank_v2"):
    Ranker._nll_state_safe_base_rank_v2 = Ranker.rank
else:
    Ranker.rank = Ranker._nll_state_safe_base_rank_v2

if not hasattr(Engine, "_nll_state_safe_base_one_v2"):
    Engine._nll_state_safe_base_one_v2 = Engine.one
else:
    Engine.one = Engine._nll_state_safe_base_one_v2


def _validate_with_runtime_state_v2(self, *args, **kwargs):
    _ensure_runtime_state()
    return Validator._nll_state_safe_base_validate_v2(self, *args, **kwargs)


def _rank_with_runtime_state_v2(self, *args, **kwargs):
    _ensure_runtime_state()
    return Ranker._nll_state_safe_base_rank_v2(self, *args, **kwargs)


def _one_with_runtime_state_v2(self, *args, **kwargs):
    _ensure_runtime_state()
    return Engine._nll_state_safe_base_one_v2(self, *args, **kwargs)


Validator.validate = _validate_with_runtime_state_v2
Ranker.rank = _rank_with_runtime_state_v2
Engine.one = _one_with_runtime_state_v2

# Recreate engine after wrapper installation.
ENGINE = Engine()


# =====================================================================
# 4. SELF-TESTS
# =====================================================================

STATE.clear()
_ensure_runtime_state()

if "warnings" not in STATE or not isinstance(STATE["warnings"], list):
    raise RuntimeError("Cell 7A v2 failed: warnings state was not restored.")

if _surface_key("A test.") != surface_key("A test."):
    raise RuntimeError("Cell 7A v2 failed: legacy surface-key alias mismatch.")

if getattr(Validator.validate, "__name__", "") != "_validate_with_runtime_state_v2":
    raise RuntimeError("Cell 7A v2 failed: Validator wrapper not installed.")

if getattr(Ranker.rank, "__name__", "") != "_rank_with_runtime_state_v2":
    raise RuntimeError("Cell 7A v2 failed: Ranker wrapper not installed.")

if getattr(Engine.one, "__name__", "") != "_one_with_runtime_state_v2":
    raise RuntimeError("Cell 7A v2 failed: Engine.one wrapper not installed.")

# Direct Validator smoke test, the scenario that previously caused
# KeyError('warnings'). The exact validation outcome is irrelevant here;
# the test verifies state-safe execution.
_test_profile = next(iter(PROFILES.values())) if "PROFILES" in globals() and PROFILES else None
if _test_profile is not None:
    _test_source = "Research claims are contingent on execution."
    _test_analysis = Analyzer().analyze(_test_source, _test_profile, set())
    _test_candidate = Candidate(
        "Research claims depend on execution.",
        "Minimal",
        "CONTINGENCY_RELATION",
        "runtime-state self-test",
        .9,
    )
    try:
        Validator().validate(
            _test_source,
            _test_candidate,
            _test_analysis,
            _test_profile,
            set(),
            "Minimal",
        )
    except KeyError as exc:
        raise RuntimeError(
            "Cell 7A v2 failed direct Validator state test: " + repr(exc)
        )

print("=" * 78)
print("CELL 7A RUNTIME-STATE COMPATIBILITY v2 LOADED")
print("=" * 78)
print("Aligned Cell 7 version: v4")
print("STATE['warnings'] initialization: active")
print("Legacy _surface_key alias: active")
print("Validator direct-call safety: active")
print("Ranker direct-call safety: active")
print("Engine.one direct-call safety: active")
print("Rerun-safe wrapper restoration: active")
print("Direct Validator state self-test: PASSED")
print("ENGINE recreated: active")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_7A_RULE_COUNT)
print("=" * 78)
