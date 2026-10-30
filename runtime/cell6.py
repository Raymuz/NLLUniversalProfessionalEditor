#@title 4B. Universal academic pattern taxonomy and naturalness analyzer

from dataclasses import dataclass, field
from collections import Counter
import re

# ============================================================
# 1. TAXONOMY
# ============================================================

PATTERN_TAXONOMY = {
    "VOICE": {
        "PASSIVE", "AGENTLESS_PASSIVE", "AGENTIVE_PASSIVE",
        "MODAL_PASSIVE", "STACKED_PASSIVE",
        "PASSIVE_EVALUATION_CONTEXT", "PASSIVE_PROCESS_CONTEXT",
        "NEGATED_CLASSIFICATION_PASSIVE",
        "COORDINATED_PASSIVE_PREDICATES", "PASSIVE_BASIS_FRAME",
    },
    "CLAUSE_STRUCTURE": {
        "WHEREBY_CLAUSE", "SO_THAT_CLAUSE", "PURPOSE_RESULT_CLAUSE",
        "FINITE_WHILE_CLAUSE", "WHILE_PARTICIPIAL",
        "PARTICIPIAL_RESULT_CLAUSE", "WHEN_CLAUSE", "WHEREAS_CLAUSE",
        "ALTHOUGH_CLAUSE", "BECAUSE_CLAUSE", "IF_CLAUSE",
        "CONTENT_COMPLEMENT", "RELATIVE_CLAUSE",
        "MULTI_EMBEDDED_CLAUSES", "CONTRASTIVE_COORDINATED_CLAUSES",
        "SEMICOLON_NEGATED_ASSUMPTION",
    },
    "NOMINALIZATION": {
        "GERUND_SUBJECT", "NOMINALIZATION", "ABSTRACT_NOUN_STACK",
        "NOUN_HEAVY_PREDICATE", "NOMINALIZED_PROCESS",
        "NOMINALIZED_EVALUATION",
    },
    "CONTRAST": {
        "CONTRAST", "NEGATED_PURPOSE_CONTRAST", "NOT_X_BUT_Y",
        "RATHER_THAN_CONTRAST", "HOWEVER_CONTRAST", "WHEREAS_CONTRAST",
        "CONTRASTIVE_COORDINATED_CLAUSES",
        "SEMICOLON_NEGATED_ASSUMPTION",
    },
    "REPORTING_EVIDENCE": {
        "REPORTING_FRAME", "EVIDENTIAL_COMPLEMENT", "ATTRIBUTION_FRAME",
        "RESULT_REPORTING_FRAME", "INDEPENDENT_REPORTING",
    },
    "EVALUATION_METHOD": {
        "EVALUATION_FRAME", "PASSIVE_EVALUATION_CONTEXT",
        "PASSIVE_PROCESS_CONTEXT", "PASSIVE_BASIS_FRAME",
        "METHOD_RELATION", "COMPARATIVE_FRAME", "METHOD_APPLICATION_FRAME",
        "EVALUATION_CONDITION_FRAME",
    },
    "COORDINATION": {
        "COORDINATED_PREDICATES", "COORDINATED_OBJECTS_OR_ENUMERATION",
        "COORDINATED_PROPOSITIONS", "LONG_ENUMERATION",
        "PARALLEL_ENUMERATION", "COORDINATED_PASSIVE_PREDICATES",
        "CONTRASTIVE_COORDINATED_CLAUSES",
    },
    "DISCOURSE": {
        "DISCOURSE_MARKER", "CONSEQUENCE", "RESULT_CLAUSE",
        "REPETITIVE_TRANSITION", "FORMULAIC_TRANSITION",
        "OVEREXPLICIT_LOGIC", "SEMICOLON_NEGATED_ASSUMPTION",
    },
    "COMPLEXITY": {
        "LONG_SENTENCE", "VERY_LONG_SENTENCE", "MULTI_EMBEDDED_CLAUSES",
        "CLAUSE_OVERLOADING", "LONG_PREPOSITIONAL_CHAIN",
        "HEAVY_POSTMODIFICATION", "EXCESSIVE_COORDINATION",
    },
    "FORMULAIC_ACADEMIC": {
        "FORMULAIC_PURPOSE", "FORMULAIC_RESULT", "FORMULAIC_COMPARISON",
        "FORMULAIC_EVALUATION", "FORMULAIC_METHOD",
        "FORMULAIC_CONCLUSION", "FORMULAIC_TRANSITION",
        "OVEREXPLICIT_LOGIC", "REPETITIVE_SENTENCE_FRAME",
    },
    "NATURALNESS": {
        "AWKWARD_PASSIVE", "UNNATURAL_NOMINALIZATION",
        "EXCESSIVE_SYMMETRY", "CLAUSE_OVERLOADING", "REPETITIVE_SYNTAX",
        "REDUNDANT_EXPLICITNESS", "STILTED_CONNECTIVE",
        "HEAVY_NOUN_PHRASE", "ABSTRACT_NOUN_STACK",
        "MECHANICAL_PARALLELISM",
    },
}

# ============================================================
# 2. RESULT OBJECT
# ============================================================

@dataclass
class PatternProfile:
    grammar_families: set = field(default_factory=set)
    formulaic_flags: set = field(default_factory=set)
    naturalness_flags: set = field(default_factory=set)
    complexity_flags: set = field(default_factory=set)
    discourse_flags: set = field(default_factory=set)
    metrics: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

# ============================================================
# 3. HELPERS
# ============================================================

def _pt_clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pt_words(text):
    return re.findall(r"\b[\w'-]+\b", str(text or ""))


def _add_evidence(profile, flag, evidence):
    profile.evidence.setdefault(flag, [])
    if evidence not in profile.evidence[flag]:
        profile.evidence[flag].append(evidence)


def _detect_content_complement(lower_text):
    if re.search(r"\bwhether\b", lower_text):
        return True

    determiner = re.search(
        r"\bthat\s+(?:type|kind|form|method|model|approach|system|procedure|"
        r"framework|process|design|strategy|technique|case|example|result|"
        r"study|analysis|experiment|feature|representation|classifier|"
        r"algorithm|dataset|environment|platform|component|domain|alignment|"
        r"space|constraint|condition|measure|attack|sample|architecture|"
        r"methodology)\b",
        lower_text,
    )
    if determiner:
        return False

    governor = re.search(
        r"\b(?:show|shows|showed|indicate|indicates|indicated|suggest|"
        r"suggests|suggested|demonstrate|demonstrates|demonstrated|report|"
        r"reports|reported|find|finds|found|observe|observes|observed|confirm|"
        r"confirms|confirmed|conclude|concludes|concluded|establish|"
        r"establishes|established|assume|assumes|assumed|ensure|ensures|"
        r"ensured|mean|means|meant|reveal|reveals|revealed|evidence|finding|"
        r"findings|result|results)\s+that\b",
        lower_text,
    )
    if governor:
        return True

    clause = re.search(
        r"\bthat\s+(?:the|a|an|this|these|those|it|they|we|researchers|"
        r"participants|results|findings|evidence|data|models?|systems?|"
        r"methods?|samples?|features?)\s+(?:is|are|was|were|has|have|had|"
        r"can|could|may|might|must|should|will|would|does|do|did|"
        r"[A-Za-z][A-Za-z-]*(?:s|ed))\b",
        lower_text,
    )
    return bool(clause)

# ============================================================
# 4. GRAMMAR-FAMILY DETECTOR
# ============================================================

def detect_grammar_families(text):
    text = _pt_clean(text)
    lower_text = text.lower()
    families = set()

    if re.search(r"\bwhereby\b", lower_text):
        families.add("WHEREBY_CLAUSE")

    if re.search(r"\bso\s+that\b", lower_text):
        families.update({"SO_THAT_CLAUSE", "PURPOSE_RESULT_CLAUSE"})

    if re.search(
        r"\bwhile\s+(?:the\s+)?[A-Za-z][A-Za-z0-9_/-]*"
        r"(?:\s+[A-Za-z][A-Za-z0-9_/-]*){0,5}\s+"
        r"(?:is|are|was|were|has|have|had|does|do|did|tracks?|measures?|"
        r"shows?|indicates?|preserves?|provides?|uses?|evaluates?|compares?|"
        r"examines?|remains?|increases?|decreases?|continues?|represents?|"
        r"performs?|captures?|retains?|allows?|enables?)\b",
        lower_text,
    ):
        families.add("FINITE_WHILE_CLAUSE")

    if re.search(r"\bwhile\s+[A-Za-z][A-Za-z-]*ing\b", lower_text):
        families.add("WHILE_PARTICIPIAL")

    if re.search(r",\s*(?:allowing|enabling|permitting)\b", lower_text):
        families.add("PARTICIPIAL_RESULT_CLAUSE")

    for family, pattern in {
        "WHEN_CLAUSE": r"\bwhen\b",
        "WHEREAS_CLAUSE": r"\bwhereas\b",
        "ALTHOUGH_CLAUSE": r"\balthough\b",
        "BECAUSE_CLAUSE": r"\bbecause\b",
        "IF_CLAUSE": r"\bif\b",
    }.items():
        if re.search(pattern, lower_text):
            families.add(family)

    if _detect_content_complement(lower_text):
        families.add("CONTENT_COMPLEMENT")

    if re.search(r"\bnot\b.+?\bbut\b", lower_text, re.S):
        families.add("CONTRAST")

    if re.search(
        r"\b(?:purpose|aim|objective|goal)\b.+?\bis\s+not\s+to\b.+?\bbut\s+to\b",
        lower_text,
        re.S,
    ):
        families.add("NEGATED_PURPOSE_CONTRAST")

    if re.match(r"^[A-Za-z][A-Za-z-]*ing\b", text, re.I):
        families.add("GERUND_SUBJECT")

    if re.search(
        r"\b(?:is|are|was|were|has been|have been|had been)\s+(?:not\s+)?"
        r"(?:\w+\s+){0,2}\w+(?:ed|en)\b",
        lower_text,
    ):
        families.add("PASSIVE")

    if re.search(r"\b(?:can|could|may|might|must|should)\s+be\s+\w+(?:ed|en)\b", lower_text):
        families.add("MODAL_PASSIVE")

    if re.search(
        r"\b(?:is|are|was|were|has been|have been)\s+evaluated\s+"
        r"(?:at|across|under|within|on|by|with|using)\b",
        lower_text,
    ):
        families.add("PASSIVE_EVALUATION_CONTEXT")

    if re.search(
        r"\b(?:is|are|was|were|has been|have been|had been)\s+"
        r"[A-Za-z][A-Za-z-]*(?:ed|en)(?:\s+(?:independently|separately|"
        r"jointly|simultaneously|iteratively|individually|concurrently))?\s+"
        r"(?:on|in|within|under|across|using|with|from|for)\b",
        lower_text,
    ):
        families.add("PASSIVE_PROCESS_CONTEXT")

    if re.search(
        r"\b(?:is|are|was|were|has been|have been|had been)\s+not\s+"
        r"(?:treated|regarded|interpreted|classified|considered)\s+as\b",
        lower_text,
    ):
        families.add("NEGATED_CLASSIFICATION_PASSIVE")

    # NEW: coordinated passive predicates.
    if re.search(
        r"\b(?:is|are|was|were|has been|have been|had been)\s+"
        r"[A-Za-z][A-Za-z-]*(?:ed|en)\s+(?:and|or)\s+"
        r"[A-Za-z][A-Za-z-]*(?:ed|en)"
        r"(?:\s+(?:independently|separately|jointly|simultaneously|individually))?\b",
        lower_text,
    ):
        families.add("COORDINATED_PASSIVE_PREDICATES")

    # NEW: contrastive coordinated clauses.
    if re.search(r",\s*(?:but|whereas|while)\s+.+", lower_text, re.S):
        families.update({"CONTRASTIVE_COORDINATED_CLAUSES", "CONTRAST"})

    # NEW: passive basis frame.
    if re.search(
        r"\b(?:is|are|was|were|has been|have been)\s+based\s+on\s+",
        lower_text,
    ):
        families.update({"PASSIVE_BASIS_FRAME", "PASSIVE"})

    # NEW: semicolon followed by a negated assumption/implication clause.
    if re.search(
        r";\s*(?:it|this|that|the\s+[A-Za-z][A-Za-z-]*)\s+"
        r"(?:does|do|did)\s+not\s+"
        r"(?:assume|imply|indicate|mean|require|establish|demonstrate|suggest)\s+that\b",
        lower_text,
    ):
        families.update({"SEMICOLON_NEGATED_ASSUMPTION", "CONTRAST"})

    if re.search(
        r"\b(?:is|are|was|were|has been|have been)\s+"
        r"(?:commonly|frequently|generally|typically|often|usually)\s+"
        r"evaluated\s+(?:using|with)\b",
        lower_text,
    ):
        families.add("EVALUATION_FRAME")

    if re.search(
        r"\b(?:results|findings|outcomes|observations)\s+reported\s+"
        r"(?:independently|separately)\b",
        lower_text,
    ):
        families.add("REPORTING_FRAME")

    if re.search(r"\bthrough\b", lower_text):
        families.add("METHOD_RELATION")

    if re.search(r"\b(?:intended|designed)\s+to\b", lower_text):
        families.add("PURPOSE_INFINITIVE")

    if re.search(
        r"\bbasis\s+for\s+(?:determining|evaluating|assessing|comparing)\b",
        lower_text,
    ):
        families.add("PURPOSE_RESULT_CLAUSE")

    if re.match(
        r"(?i)^(?:however|therefore|consequently|thus|accordingly|"
        r"nevertheless|nonetheless|in contrast|by contrast|as a result)\b",
        text,
    ):
        families.add("DISCOURSE_MARKER")

    if re.match(r"(?i)^(?:therefore|consequently|thus|accordingly|as a result)\b", text):
        families.update({"CONSEQUENCE", "RESULT_CLAUSE"})

    return families

# ============================================================
# 5. FORMULAIC-STYLE DETECTOR
# ============================================================

FORMULAIC_PATTERNS = {
    "FORMULAIC_PURPOSE": [
        r"\bthe purpose (?:of this|is to)\b",
        r"\bthe aim (?:of this|is to)\b",
        r"\bthe objective (?:of this|is to)\b",
    ],
    "FORMULAIC_RESULT": [
        r"\bas a result\b", r"\bconsequently\b", r"\btherefore\b",
        r"\bthis suggests that\b", r"\bthese results indicate that\b",
    ],
    "FORMULAIC_COMPARISON": [
        r"\bin contrast\b", r"\bby contrast\b", r"\bcompared with\b", r"\bcompared to\b",
    ],
    "FORMULAIC_EVALUATION": [
        r"\bis evaluated using\b", r"\bare evaluated using\b",
        r"\bis evaluated across\b", r"\bare evaluated across\b",
        r"\bis assessed using\b", r"\bare assessed using\b",
    ],
    "FORMULAIC_METHOD": [
        r"\busing this approach\b", r"\bunder this framework\b",
        r"\bwithin this framework\b", r"\bthe methodology allows\b",
    ],
    "FORMULAIC_CONCLUSION": [
        r"\btaken together\b", r"\boverall, these\b", r"\bcollectively, these\b",
    ],
    "OVEREXPLICIT_LOGIC": [
        r"\bthis means that\b", r"\bthis indicates that\b",
        r"\bthis demonstrates that\b", r"\bit is important to note that\b",
        r"\bit should be noted that\b",
    ],
    "STILTED_CONNECTIVE": [
        r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b", r"\baccordingly\b",
    ],
}


def detect_formulaic_patterns(text, profile):
    lower_text = _pt_clean(text).lower()
    for flag, patterns in FORMULAIC_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lower_text, re.I)
            if match:
                profile.formulaic_flags.add(flag)
                _add_evidence(profile, flag, match.group(0))

# ============================================================
# 6. NATURALNESS / COMPLEXITY DETECTOR
# ============================================================

def detect_naturalness(text, profile):
    text = _pt_clean(text)
    words = _pt_words(text)
    word_count = len(words)
    profile.metrics["word_count"] = word_count

    if word_count >= 35:
        profile.complexity_flags.add("LONG_SENTENCE")
    if word_count >= 50:
        profile.complexity_flags.add("VERY_LONG_SENTENCE")

    clause_markers = re.findall(
        r"\b(?:that|which|whereby|whereas|although|because|while|when|whether|if|so that)\b",
        text,
        re.I,
    )
    profile.metrics["clause_marker_count"] = len(clause_markers)
    if len(clause_markers) >= 3:
        profile.complexity_flags.add("MULTI_EMBEDDED_CLAUSES")
    if word_count >= 35 and len(clause_markers) >= 2:
        profile.naturalness_flags.add("CLAUSE_OVERLOADING")

    prepositions = re.findall(
        r"\b(?:of|in|within|under|across|among|through|between|for|from|with|by|on|at)\b",
        text,
        re.I,
    )
    profile.metrics["preposition_count"] = len(prepositions)
    if len(prepositions) >= 6 and word_count >= 25:
        profile.complexity_flags.add("LONG_PREPOSITIONAL_CHAIN")

    passive_hits = re.findall(
        r"\b(?:is|are|was|were|has been|have been|had been)\s+(?:not\s+)?"
        r"(?:\w+\s+){0,2}\w+(?:ed|en)\b",
        text,
        re.I,
    )
    profile.metrics["passive_count"] = len(passive_hits)
    if len(passive_hits) >= 2:
        profile.naturalness_flags.add("AWKWARD_PASSIVE")

    transitions = re.findall(
        r"\b(?:however|therefore|consequently|thus|moreover|furthermore|"
        r"accordingly|nevertheless|nonetheless)\b",
        text,
        re.I,
    )
    profile.metrics["transition_count"] = len(transitions)
    if len(transitions) >= 2:
        profile.discourse_flags.add("REPETITIVE_TRANSITION")

    nominalizations = re.findall(
        r"\b[A-Za-z]+(?:tion|sion|ment|ance|ence|ity|ness)\b",
        text,
        re.I,
    )
    profile.metrics["nominalization_count"] = len(nominalizations)
    if len(nominalizations) >= 4 and word_count >= 25:
        profile.naturalness_flags.add("UNNATURAL_NOMINALIZATION")

    and_count = len(re.findall(r"\band\b", text, re.I))
    profile.metrics["and_count"] = and_count
    if and_count >= 4 and word_count >= 30:
        profile.naturalness_flags.add("MECHANICAL_PARALLELISM")

# ============================================================
# 7. MAIN ANALYZER / REPORT
# ============================================================

def analyze_pattern_taxonomy(text):
    profile = PatternProfile()
    profile.grammar_families.update(detect_grammar_families(text))
    detect_formulaic_patterns(text, profile)
    detect_naturalness(text, profile)
    return profile


def taxonomy_categories(flag):
    return [category for category, members in PATTERN_TAXONOMY.items() if flag in members]


def pattern_report(text):
    profile = analyze_pattern_taxonomy(text)
    return {
        "grammar_families": sorted(profile.grammar_families),
        "formulaic_flags": sorted(profile.formulaic_flags),
        "naturalness_flags": sorted(profile.naturalness_flags),
        "complexity_flags": sorted(profile.complexity_flags),
        "discourse_flags": sorted(profile.discourse_flags),
        "metrics": dict(profile.metrics),
        "evidence": dict(profile.evidence),
    }

# ============================================================
# 8. SELF-TESTS
# ============================================================

_PATTERN_TESTS = {
    "while_participial": "The comparison preserves the constraints while applying the methodology.",
    "while_finite": "Accuracy measures overall performance, while recall tracks missed examples.",
    "whereby": "The design uses a common procedure whereby each model is evaluated so that comparisons remain consistent.",
    "evaluation_context": "The models are evaluated across several conditions.",
    "negated_purpose": "The purpose is not to rank the methods but to compare their behavior.",
    "passive_process": "The models are trained independently on their respective feature representations.",
    "negated_classification": "The representations are not treated as evidence of invariant features.",
    "participial_result": "The models operate in separate environments, allowing their effects to be evaluated.",
    "that_determiner": "The study does not implement that type of domain alignment.",
    "that_content_complement": "The results indicate that the models preserve their original behavior.",
    "coordinated_passive": "The models are trained and evaluated separately.",
    "contrastive_coordination": "The models are trained separately, but both environments use the same procedure.",
    "passive_basis": "The interpretation is based on changes in the measured outcomes.",
    "semicolon_negated_assumption": "The comparison uses separate measurements; it does not assume that the representations are equivalent.",
}

_PATTERN_RESULTS = {name: pattern_report(sentence) for name, sentence in _PATTERN_TESTS.items()}


def _assert_pattern(test_name, expected_family):
    detected = _PATTERN_RESULTS[test_name]["grammar_families"]
    if expected_family not in detected:
        raise RuntimeError(
            f"Pattern taxonomy failed {expected_family} self-test for '{test_name}'. Detected: {detected}"
        )


for test_name, expected in [
    ("while_participial", "WHILE_PARTICIPIAL"),
    ("while_finite", "FINITE_WHILE_CLAUSE"),
    ("whereby", "WHEREBY_CLAUSE"),
    ("evaluation_context", "PASSIVE_EVALUATION_CONTEXT"),
    ("negated_purpose", "NEGATED_PURPOSE_CONTRAST"),
    ("passive_process", "PASSIVE_PROCESS_CONTEXT"),
    ("negated_classification", "NEGATED_CLASSIFICATION_PASSIVE"),
    ("participial_result", "PARTICIPIAL_RESULT_CLAUSE"),
    ("that_content_complement", "CONTENT_COMPLEMENT"),
    ("coordinated_passive", "COORDINATED_PASSIVE_PREDICATES"),
    ("contrastive_coordination", "CONTRASTIVE_COORDINATED_CLAUSES"),
    ("passive_basis", "PASSIVE_BASIS_FRAME"),
    ("semicolon_negated_assumption", "SEMICOLON_NEGATED_ASSUMPTION"),
]:
    _assert_pattern(test_name, expected)

if "CONTENT_COMPLEMENT" in _PATTERN_RESULTS["that_determiner"]["grammar_families"]:
    raise RuntimeError("Pattern taxonomy incorrectly classified determiner 'that' as CONTENT_COMPLEMENT.")

# ============================================================
# 9. STATUS
# ============================================================

print("=" * 72)
print("CELL 4B PATTERN TAXONOMY SUCCESSFULLY LOADED")
print("=" * 72)
print("Grammar-family taxonomy: active")
print("Formulaic-expression analysis: active")
print("Naturalness analysis: active")
print("Complexity analysis: active")
print("Discourse analysis: active")
print("\nVerified grammar routes:")
for family in [
    "FINITE_WHILE_CLAUSE", "WHILE_PARTICIPIAL", "WHEREBY_CLAUSE",
    "PASSIVE_EVALUATION_CONTEXT", "PASSIVE_PROCESS_CONTEXT",
    "NEGATED_CLASSIFICATION_PASSIVE", "PARTICIPIAL_RESULT_CLAUSE",
    "NEGATED_PURPOSE_CONTRAST", "CONTENT_COMPLEMENT",
    "COORDINATED_PASSIVE_PREDICATES", "CONTRASTIVE_COORDINATED_CLAUSES",
    "PASSIVE_BASIS_FRAME", "SEMICOLON_NEGATED_ASSUMPTION",
]:
    print(f"  [OK] {family}")
print("  [OK] determiner 'that' exclusion")
print("\nCell 4B is ready. Update and run Cell 5 next.")
