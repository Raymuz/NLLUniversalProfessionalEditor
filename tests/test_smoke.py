"""Execute: python -m tests.test_smoke (requires production dependencies/model)."""
from runtime import engine as nll

def test_contract():
    assert len(nll.RULES) == 30, f"expected 30 original rules, found {len(nll.RULES)}"
    names = [getattr(rule, "name", str(key)) for key, rule in enumerate(nll.RULES.values())] if isinstance(nll.RULES, dict) else [getattr(r, "name", "") for r in nll.RULES]
    assert len(names) == len(set(names)), "duplicate rule names"
    assert "parser" in nll.nlp().pipe_names, "missing dependency parser"
    assert getattr(nll, "MUST_PARAPHRASE_ACTIVE", False), "must-paraphrase disabled"

def test_rewrite():
    sample = "Many Android indicators are binary or discrete, meaning adversarial perturbations must preserve feature semantics and valid ranges."
    result = nll.ENGINE.process(sample, "Edit / Paraphrase Only", "Balanced", "Auto Detect", "", None)
    assert isinstance(result.revised, str) and result.revised.strip()
    assert "Android" in result.revised

if __name__ == "__main__":
    test_contract()
    test_rewrite()
    print("Smoke tests passed")
