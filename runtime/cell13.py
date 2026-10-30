#@title 7B. DETERMINISTIC MUST-PARAPHRASE ORCHESTRATOR v2.1
# Complete replacement for Cell 7B.
# Run after Cell 7A v2 and before Cells 8, 8A, 9, and 10.
#
# This orchestration layer does not generate language itself.
# It performs a bounded search over the three existing deterministic modes,
# delegates candidate judgment to Cell 7 v4, preserves all hard safeguards,
# and keeps the source only when every safe search path is exhausted.

from collections import Counter

EXPECTED_7B_RULE_COUNT = 30

_REQUIRED_7B = [
    "STATE", "RULES", "Engine", "ENGINE", "Validator", "Ranker",
    "Generator", "Analyzer", "PROFILES", "Candidate",
    "surface_key", "realization_issues", "_ensure_runtime_state",
]
_missing_7b = [name for name in _REQUIRED_7B if name not in globals()]
if _missing_7b:
    raise RuntimeError(
        "Cell 7B cannot run. Missing: " + ", ".join(_missing_7b)
        + ". Run Cell 7 v4.3 and Cell 7A v2 first."
    )

if len(RULES) != EXPECTED_7B_RULE_COUNT:
    raise RuntimeError(
        f"Cell 7B expected {EXPECTED_7B_RULE_COUNT} rules but found {len(RULES)}."
    )

# Verify that 7A v2, not an older wrapper, is installed.
if getattr(Engine.one, "__name__", "") != "_one_with_runtime_state_v2":
    raise RuntimeError(
        "Cell 7B requires Cell 7A v2 immediately before it. "
        "Expected Engine.one wrapper '_one_with_runtime_state_v2'."
    )


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

MUST_PARAPHRASE_ENABLED = True

_MODE_RETRY_ORDER = {
    "Minimal": ["Minimal", "Balanced", "Structural"],
    "Balanced": ["Balanced", "Structural", "Minimal"],
    "Structural": ["Structural", "Balanced", "Minimal"],
}

_HARD_REJECTION_CODES = {
    "TERMINOLOGY_REJECTION",
    "POLARITY_REJECTION",
    "MODALITY_REJECTION",
    "QUANTITY_REJECTION",
    "ENUMERATION_REJECTION",
    "GRAMMAR_REJECTION",
    "SEMANTIC_REJECTION",
}


def _mp_changed(original, revised):
    return surface_key(original) != surface_key(revised)


def _mp_trace(result):
    trace = getattr(result, "diagnostics", None)
    if trace is None:
        trace = getattr(result, "trace", None)
    return dict(trace or {})


def _mp_checks(result):
    return dict(getattr(result, "checks", None) or {})


def _mp_result_summary(result, attempted_mode):
    trace = _mp_trace(result)
    checks = _mp_checks(result)
    revised = getattr(result, "revised", "")
    original = getattr(result, "original", "")
    selected = trace.get("selected") or {}
    failures = list(checks.get("failures", []) or [])
    realization = list(realization_issues(revised) or [])

    return {
        "attempted_mode": attempted_mode,
        "status": getattr(result, "status", ""),
        "changed": _mp_changed(original, revised),
        "selected_family": getattr(result, "family", None) or selected.get("family"),
        "selected_lane": selected.get("lane"),
        "selected_score": selected.get("score"),
        "failures": failures,
        "hard_failures": [code for code in failures if code in _HARD_REJECTION_CODES],
        "realization_issues": realization,
        "generated_by_lane": trace.get("generated_by_lane", {}),
        "generated_before_final_filter": trace.get("generated_before_final_filter"),
        "generated_after_final_filter": trace.get("generated_after_final_filter"),
        "realization_rejections": trace.get("realization_rejections", []),
        "terminal_rejections": trace.get("rejections", {}),
        "reason": getattr(result, "reason", ""),
        "revised": revised,
    }


def _mp_is_safe_success(result):
    if getattr(result, "status", "") != "SUCCESS":
        return False

    original = getattr(result, "original", "")
    revised = getattr(result, "revised", "")
    if not _mp_changed(original, revised):
        return False

    checks = _mp_checks(result)
    failures = set(checks.get("failures", []) or [])
    if failures & _HARD_REJECTION_CODES:
        return False

    if realization_issues(revised):
        return False

    return True


def _mp_failure_reason(attempts):
    if not attempts:
        return (
            "No rewrite attempt was completed. The original sentence was preserved "
            "because the deterministic judge had no result to assess."
        )

    generated_total = 0
    realization_rejections = 0
    rejection_codes = Counter()
    statuses = Counter()

    for attempt in attempts:
        statuses[attempt.get("status", "UNKNOWN")] += 1

        generated = attempt.get("generated_after_final_filter")
        if generated is None:
            generated = sum(
                int(value or 0)
                for value in (attempt.get("generated_by_lane", {}) or {}).values()
            )
        generated_total += int(generated or 0)
        realization_rejections += len(attempt.get("realization_rejections", []) or [])

        for code in attempt.get("failures", []) or []:
            rejection_codes[code] += 1
        for code, count in (attempt.get("terminal_rejections", {}) or {}).items():
            rejection_codes[code] += int(count or 0)

    if generated_total == 0 and realization_rejections == 0:
        return (
            "The original sentence was preserved because no registered transformation "
            "produced a distinct candidate in Minimal, Balanced, or Structural mode."
        )

    if realization_rejections and generated_total == 0:
        return (
            "The original sentence was preserved because every generated alternative "
            "failed the grammatical realization gate."
        )

    if rejection_codes:
        detail = ", ".join(
            f"{code}: {count}" for code, count in sorted(rejection_codes.items())
        )
        return (
            "The original sentence was preserved because all distinct alternatives "
            f"were rejected by the deterministic judge ({detail})."
        )

    status_detail = ", ".join(
        f"{status}: {count}" for status, count in sorted(statuses.items())
    )
    return (
        "The original sentence was preserved because no attempted mode produced a "
        "safe validated rewrite. Attempt outcomes: " + status_detail + "."
    )


# =====================================================================
# 2. RERUN-SAFE ENGINE.ONE ORCHESTRATION
# =====================================================================

# Preserve the post-7A-v2 Engine.one once. If 7B v2 is rerun without
# rerunning 7A, restore that exact base before installing this wrapper again.
if not hasattr(Engine, "_nll_must_paraphrase_base_one_v2"):
    Engine._nll_must_paraphrase_base_one_v2 = Engine.one
else:
    Engine.one = Engine._nll_must_paraphrase_base_one_v2


def _one_with_must_paraphrase_v2(self, sentence, profile, user_terms, mode):
    _ensure_runtime_state()

    requested_mode = mode if mode in _MODE_RETRY_ORDER else "Balanced"

    if not MUST_PARAPHRASE_ENABLED:
        return Engine._nll_must_paraphrase_base_one_v2(
            self, sentence, profile, user_terms, requested_mode
        )

    attempts = []
    attempted_results = []

    for attempted_mode in _MODE_RETRY_ORDER[requested_mode]:
        result = Engine._nll_must_paraphrase_base_one_v2(
            self, sentence, profile, user_terms, attempted_mode
        )
        attempt = _mp_result_summary(result, attempted_mode)
        attempts.append(attempt)
        attempted_results.append(result)

        if _mp_is_safe_success(result):
            trace = _mp_trace(result)
            trace["must_paraphrase"] = {
                "enabled": True,
                "requested_mode": requested_mode,
                "accepted_mode": attempted_mode,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "outcome": "SAFE_CHANGED_OUTPUT",
                "hard_safeguards_relaxed": False,
            }

            try:
                result.diagnostics = trace
            except Exception:
                pass
            try:
                result.trace = trace
            except Exception:
                pass

            original_reason = str(getattr(result, "reason", "") or "").strip()
            suffix = (
                f" Must-paraphrase judge accepted the {attempted_mode} attempt "
                f"after {len(attempts)} bounded attempt(s)."
            )
            try:
                result.reason = (original_reason + suffix).strip()
            except Exception:
                pass
            return result

    # Exhausted all three modes. Preserve the source and expose precisely why.
    fallback = attempted_results[0]
    reason = _mp_failure_reason(attempts)
    trace = _mp_trace(fallback)
    trace["must_paraphrase"] = {
        "enabled": True,
        "requested_mode": requested_mode,
        "accepted_mode": None,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "outcome": "ORIGINAL_PRESERVED_AFTER_EXHAUSTIVE_SAFE_SEARCH",
        "preservation_reason": reason,
        "hard_safeguards_relaxed": False,
    }

    try:
        fallback.diagnostics = trace
    except Exception:
        pass
    try:
        fallback.trace = trace
    except Exception:
        pass
    try:
        fallback.reason = reason
    except Exception:
        pass

    return fallback


Engine.one = _one_with_must_paraphrase_v2
ENGINE = Engine()


# =====================================================================
# 3. SELF-TESTS
# =====================================================================

if getattr(Engine.one, "__name__", "") != "_one_with_must_paraphrase_v2":
    raise RuntimeError("Cell 7B v2.1 failed: Engine.one wrapper not installed.")

if not hasattr(Engine, "_nll_must_paraphrase_base_one_v2"):
    raise RuntimeError("Cell 7B v2.1 failed: base Engine.one was not preserved.")

for preferred, order in _MODE_RETRY_ORDER.items():
    if len(order) != 3 or order[0] != preferred or len(set(order)) != 3:
        raise RuntimeError(
            f"Cell 7B v2 failed retry order for {preferred}: {order}"
        )

# End-to-end orchestration smoke test using a known transformable sentence.
_test_profile, _ = ENGINE.route(
    "The next step is to inspect how the handler checks configuration.",
    "Auto Detect",
)
_test_result = ENGINE.one(
    "The next step is to inspect how the handler checks configuration.",
    _test_profile,
    set(),
    "Balanced",
)
_test_trace = _mp_trace(_test_result)
_test_mp = _test_trace.get("must_paraphrase", {}) or {}

if getattr(_test_result, "status", "") != "SUCCESS":
    raise RuntimeError(
        "Cell 7B v2 smoke test failed: known transformable sentence did not succeed."
    )
if not _mp_changed(_test_result.original, _test_result.revised):
    raise RuntimeError(
        "Cell 7B v2 smoke test failed: successful result was unchanged."
    )
if _test_mp.get("outcome") != "SAFE_CHANGED_OUTPUT":
    raise RuntimeError(
        "Cell 7B v2 smoke test failed: orchestration trace missing safe-change outcome."
    )
if _test_mp.get("requested_mode") != "Balanced":
    raise RuntimeError(
        "Cell 7B v2.1 smoke test failed: requested mode was not preserved."
    )

# Verify the live ENGINE instance is bound to the orchestration wrapper,
# not merely the Engine class.
_live_func = getattr(getattr(ENGINE, "one", None), "__func__", None)
if _live_func is not Engine.one:
    raise RuntimeError(
        "Cell 7B v2.1 failed: live ENGINE.one is not bound to current Engine.one."
    )
if getattr(ENGINE.one, "__name__", "") != "_one_with_must_paraphrase_v2":
    raise RuntimeError(
        "Cell 7B v2.1 failed: live ENGINE instance is not using must-paraphrase."
    )

# Document-level trace survival test. This catches the exact integration
# failure where ENGINE.one is wrapped but Engine.process returns sentence
# results without the must_paraphrase diagnostics consumed by Cell 9.
_doc_probe = ENGINE.process(
    "The next step is to inspect how the handler checks configuration.",
    "Edit / Paraphrase Only",
    "Balanced",
    "Auto Detect",
    "",
)
if not getattr(_doc_probe, "sentences", None):
    raise RuntimeError(
        "Cell 7B v2.1 failed: document-level probe returned no sentence results."
    )
_doc_sentence = _doc_probe.sentences[0]
_doc_trace = _mp_trace(_doc_sentence)
_doc_mp = _doc_trace.get("must_paraphrase", {}) or {}
if int(_doc_mp.get("attempt_count", 0) or 0) < 1:
    raise RuntimeError(
        "Cell 7B v2.1 failed: must-paraphrase attempt trace did not survive Engine.process."
    )
if _doc_mp.get("outcome") not in {
    "SAFE_CHANGED_OUTPUT",
    "ORIGINAL_PRESERVED_AFTER_EXHAUSTIVE_SAFE_SEARCH",
}:
    raise RuntimeError(
        "Cell 7B v2.1 failed: document-level orchestration outcome is missing."
    )

# Stable runtime markers for later cells. Prefer these markers over relying
# only on function-name strings.
MUST_PARAPHRASE_RUNTIME = {
    "version": "2.1",
    "active": True,
    "engine_class_wrapper": getattr(Engine.one, "__name__", ""),
    "engine_instance_wrapper": getattr(ENGINE.one, "__name__", ""),
    "document_trace_verified": True,
    "rule_count": len(RULES),
}

print("=" * 78)
print("CELL 7B DETERMINISTIC MUST-PARAPHRASE ORCHESTRATOR v2.1 LOADED")
print("=" * 78)
print("Aligned Cell 7 version: v4.3")
print("Aligned Cell 7A version: v2")
print("Requested-mode-first search: active")
print("Minimal/Balanced/Structural retry: active")
print("Validator + Ranker deterministic judge: active")
print("Hard safeguard relaxation: disabled")
print("Realization safeguard relaxation: disabled")
print("Unchanged-output preservation reason: active")
print("Bounded retry loop: active")
print("Sentence-level orchestration smoke test: PASSED")
print("Live ENGINE binding check: PASSED")
print("Document-level trace survival: PASSED")
print("Runtime marker: MUST_PARAPHRASE_RUNTIME active")
print("LLM generation: not used")
print("ENGINE recreated: active")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_7B_RULE_COUNT)
print("=" * 78)
