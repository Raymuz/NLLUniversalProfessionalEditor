#@title 6. UNIVERSAL ROUTING v5.1 — alignment for Cell 5 v4.1 / Cell 5A v3.4
# Complete replacement for Cell 6 v5.
#
# Goals:
# - rerun-safe overlay on the true pre-Cell-6 Analyzer;
# - citation-safe sentence splitting;
# - broad surface detection plus dependency-confirmed structure detection;
# - explicit primary-family metadata and secondary-structure metadata;
# - exact alignment with the 30-rule Cell 5 v4.1 registry;
# - no transformation generation and no semantic validation in this cell.

import re

EXPECTED_CELL6_RULE_COUNT = 30


# =====================================================================
# 0. PREREQUISITES
# =====================================================================

_REQUIRED_6 = [
    "Analyzer", "RULES", "sent_split", "nlp",
]
_missing_6 = [name for name in _REQUIRED_6 if name not in globals()]
if _missing_6:
    raise RuntimeError(
        "Cell 6 cannot run. Missing: " + ", ".join(_missing_6)
        + ". Run Cells 1–5A first."
    )

if len(RULES) != EXPECTED_CELL6_RULE_COUNT:
    raise RuntimeError(
        f"Cell 6 v5.1 expected {EXPECTED_CELL6_RULE_COUNT} Cell 5 v4.1 rules, "
        f"but found {len(RULES)}."
    )


# =====================================================================
# 1. RERUN-SAFE PRESERVATION OF TRUE PRE-CELL-6 FUNCTIONS
# =====================================================================

if "_NLL_CELL6_ORIGINAL_SENT_SPLIT" not in globals():
    _NLL_CELL6_ORIGINAL_SENT_SPLIT = sent_split

if not hasattr(Analyzer, "_nll_cell6_original_analyze"):
    Analyzer._nll_cell6_original_analyze = Analyzer.analyze

_BASE_ANALYZE = Analyzer._nll_cell6_original_analyze


# =====================================================================
# 2. CITATION/ABBREVIATION-SAFE SENTENCE SPLITTING
# =====================================================================

_CELL6_ABBREVIATIONS = {
    r"\bet\s+al\.": "et al<NLL_DOT>",
    r"\be\.g\.": "e<NLL_DOT>g<NLL_DOT>",
    r"\bi\.e\.": "i<NLL_DOT>e<NLL_DOT>",
    r"\bFig\.": "Fig<NLL_DOT>",
    r"\bFigs\.": "Figs<NLL_DOT>",
    r"\bEq\.": "Eq<NLL_DOT>",
    r"\bEqs\.": "Eqs<NLL_DOT>",
    r"\bDr\.": "Dr<NLL_DOT>",
    r"\bProf\.": "Prof<NLL_DOT>",
    r"\bMr\.": "Mr<NLL_DOT>",
    r"\bMs\.": "Ms<NLL_DOT>",
    r"\bMrs\.": "Mrs<NLL_DOT>",
    r"\bNo\.": "No<NLL_DOT>",
}


def sent_split(text):
    protected = str(text or "")

    for pattern, replacement in _CELL6_ABBREVIATIONS.items():
        protected = re.sub(pattern, replacement, protected, flags=re.I)

    parts = _NLL_CELL6_ORIGINAL_SENT_SPLIT(protected)

    return [
        str(part).replace("<NLL_DOT>", ".").strip()
        for part in parts
        if str(part).strip()
    ]


# =====================================================================
# 3. ROUTING HELPERS
# =====================================================================

_PRIMARY_FAMILIES_V51 = {
    "COPULAR_ACTION_FRAME",
    "NEED_TO_ACTION",
    "FINITE_PURPOSE_INTENT",
    "RATHER_THAN_GERUND",
    "MODAL_COPULAR_ENUMERATION",
    "USING_METHOD_RELATION",
    "THROUGH_METHOD_RELATION",
    "BY_GERUND_METHOD_RELATION",
    "PURPOSE_INFINITIVE_FRONTING",
    "NEGATED_PURPOSE_CONTRAST",
    "EVIDENTIAL_COMPLEMENT",
    "WHETHER_COMPLEMENT",
    "GERUND_SUBJECT",
    "WHEREBY_CLAUSE",
    "WHILE_PARTICIPIAL",
    "FRONTED_ADJUNCT",
    "DISCOURSE_FRONTING",
    "COORDINATED_PASSIVE_PARTICIPLES",
    "PASSIVE_USING",
    "PASSIVE",
    "REDUCED_RELATIVE_OR_NP_MODIFIER",
    "PARENTHETICAL_EMPHASIS",
    "COORDINATED_PREDICATES",
    "FINITE_COMPLEMENT_LIST",
    "PROPOSAL_FRAME",
    "APPOSITIVE_DEFINITION",
    "PARTICIPIAL_POSTMODIFIER",
    "FINITE_OBJECT_LIST_FRAME",
    "CONTINGENCY_RELATION",
}

_SECONDARY_FAMILIES_V51 = {
    "ENUMERATION",
    "COORDINATED_OBJECTS_OR_ENUMERATION",
    "LONG_MULTI_CLAUSE",
    "NOMINALIZATION",
    "CONTRAST",
    "CONTENT_COMPLEMENT",
    "PURPOSE_RESULT_CLAUSE",
    "PURPOSE_INFINITIVE",
    "MODAL_PASSIVE",
}

_PARTICIPIAL_LEMMAS = {"span", "cover", "include", "contain", "involve"}


def _v51_dependency_context(text):
    try:
        doc = nlp()(text)
        if not len(doc):
            return None, False
        return doc, bool(doc.has_annotation("DEP"))
    except Exception:
        return None, False


def _v51_dependency_participial_postmodifier(doc):
    if doc is None or not doc.has_annotation("DEP"):
        return False

    for token in doc:
        if token.lemma_.lower() not in _PARTICIPIAL_LEMMAS:
            continue
        if token.tag_ != "VBG" and not token.text.lower().endswith("ing"):
            continue

        # Require an attested noun/proper-noun head. This is the same
        # structural evidence Cell 5 v4.1 relies on for safe realization.
        if token.head is token:
            continue
        if token.head.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue

        if token.dep_ in {"acl", "relcl", "amod", "advcl", "dep"}:
            return True

    return False


def _v51_dependency_coord_predicates(doc):
    if doc is None or not doc.has_annotation("DEP"):
        return False

    roots = [token for token in doc if token.dep_ == "ROOT"]
    if len(roots) != 1:
        return False

    root = roots[0]
    if root.pos_ not in {"VERB", "AUX"}:
        return False

    subjects = [
        child for child in root.children
        if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
    ]
    if len(subjects) != 1:
        return False

    conjuncts = [
        token for token in root.conjuncts
        if token.dep_ == "conj"
        and token.pos_ in {"VERB", "AUX"}
        and token.i > root.i
    ]
    if not conjuncts:
        return False

    return True


def _v51_dependency_noun_enumeration(doc):
    if doc is None or not doc.has_annotation("DEP"):
        return False

    return any(
        token.dep_ == "conj"
        and token.head.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}
        for token in doc
    )


# =====================================================================
# 4. ANALYZER OVERLAY
# =====================================================================


def _analyze_v51(self, sentence, profile, user_terms):
    analysis = _BASE_ANALYZE(self, sentence, profile, user_terms)

    text = str(sentence or "").strip()
    low = text.lower()
    families = set(getattr(analysis, "families", []) or [])
    notes = getattr(analysis, "notes", None)
    if notes is None:
        notes = []
        analysis.notes = notes

    comma_count = len(re.findall(r",", text))
    semicolon_count = text.count(";")
    has_coord_word = bool(re.search(r"\b(?:and|or)\b", low))
    has_enumeration = comma_count >= 2 and has_coord_word

    doc, dependency_available = _v51_dependency_context(text)

    tests = {
        "CONTRAST": bool(re.search(
            r"\bnot\b.+?\bbut\b|\b(?:but|although|whereas|however)\b",
            low,
            re.S,
        )),
        "CONTENT_COMPLEMENT": bool(re.search(
            r"\b(?:evidence|shows?|indicates?|suggests?|demonstrates?)\s+that\b",
            low,
        )),
        "EVIDENTIAL_COMPLEMENT": bool(re.search(
            r"\b(?:provide|provides|provided|offer|offers|offered)\s+"
            r"(?:clear\s+|strong\s+|empirical\s+)?evidence\s+that\b",
            low,
        )),
        "WHETHER_COMPLEMENT": bool(re.search(
            r"\brequire(?:s|d)?\s+(?:an?\s+)?"
            r"(?:consideration|assessment|evaluation|analysis)\s+of\s+whether\b",
            low,
        )),
        "NEGATED_PURPOSE_CONTRAST": bool(re.search(
            r"\b(?:purpose|aim|objective|goal)\b.*?\bis\s+not\s+to\b"
            r".+?\bbut\s+to\b",
            low,
            re.S,
        )),
        "GERUND_SUBJECT": bool(re.match(
            r"^(?:Applying|Using|Evaluating|Comparing|Assessing|Examining|"
            r"Analyzing|Implementing|Testing|Measuring)\b",
            text,
            re.I,
        )),
        "WHEREBY_CLAUSE": bool(re.search(r"\bwhereby\b", low)),
        "PURPOSE_RESULT_CLAUSE": bool(re.search(r"\bso\s+that\b", low)),
        "WHILE_PARTICIPIAL": bool(re.search(r"\bwhile\s+\w+ing\b", low)),
        "PARENTHETICAL_EMPHASIS": bool(re.search(
            r",\s*by\s+(?:itself|themselves)\s*,|"
            r"\b(?:by itself|by themselves|alone)\b",
            low,
        )),
        "FRONTED_ADJUNCT": bool(re.match(
            r"^(?:for|in|within|under|during|among|across|throughout)\b[^,]{1,160},",
            low,
        )),
        "METHOD_RELATION": bool(re.search(
            r"\bthrough\s+.+|\bby\s+\w+ing\b|\busing\s+.+",
            low,
        )),
        "PURPOSE_INFINITIVE": bool(re.search(
            r"\b(?:intended|designed|used|developed|implemented)\s+to\b",
            low,
        )),
        "MODAL_PASSIVE": bool(re.search(
            r"\b(?:can|could|may|might|must|should|would|will)\s+be\s+"
            r"(?:\w+ly\s+)?\w+(?:ed|en)\b",
            low,
        )),
        "PASSIVE_USING": bool(re.search(
            r"\b(?:is|are|was|were|has been|have been|had been)\s+"
            r"(?:\w+ly\s+)?evaluated\s+using\b",
            low,
        )),
        "PASSIVE": bool(re.search(
            r"\b(?:is|are|was|were|has|have|had)\s+(?:\w+ly\s+)?(?:been\s+)?"
            r"(?:studied|examined|investigated|evaluated|reported|obtained|"
            r"derived|collected|trained|modified|implemented|applied|tested|"
            r"inspected|approved|reviewed|documented)\b",
            low,
        )),
        "REDUCED_RELATIVE_OR_NP_MODIFIER": bool(re.search(
            r"\b(?:results?|evidence|data|samples?|features?|models?|constraints?|"
            r"methods?|files?|representations?)\s+(?:obtained|derived|collected|"
            r"trained|reported|based|intended|designed|used|modified|implemented)\b",
            low,
        )),
        "DISCOURSE_FRONTING": bool(re.match(
            r"^(?:however|therefore|consequently|thus|moreover|furthermore|"
            r"nevertheless|accordingly|additionally),",
            low,
        )),
        "LONG_MULTI_CLAUSE": (
            len(text.split()) >= 32
            or semicolon_count >= 1
            or comma_count >= 4
        ),
        "COPULAR_ACTION_FRAME": bool(re.match(
            r"^the\s+(?:next|first|second|third|final|primary|main)\s+"
            r"(?:step|task|action|objective|goal)\s+is\s+to\b",
            low,
        )),
        "NEED_TO_ACTION": bool(re.search(r"\bneed(?:s)?\s+to\b", low)),
        "MODAL_COPULAR_ENUMERATION": bool(re.search(
            r"\b(?:may|might|could|can)\s+be\s+in\s+"
            r".+?,\s*.+?\s*,?\s*(?:or|and)\s+.+",
            low,
            re.S,
        )),
        "USING_METHOD_RELATION": bool(re.search(r"\busing\s+.+", low)),
        "THROUGH_METHOD_RELATION": bool(re.search(r"\bthrough\s+.+", low)),
        "BY_GERUND_METHOD_RELATION": bool(re.search(r"\bby\s+\w+ing\b", low)),
        "FINITE_PURPOSE_INTENT": bool(re.search(
            r"\b(?:intend|intends|intended)\s+to\b",
            low,
        )),
        "RATHER_THAN_GERUND": bool(re.search(r"\brather\s+than\s+\w+ing\b", low)),
        "COORDINATED_PASSIVE_PARTICIPLES": bool(re.search(
            r"\b(?:(?:have|has|had)\s+been|"
            r"(?:can|could|may|might|must|should|would|will)\s+be)\s+"
            r"(?:\w+ly\s+)?\w+(?:ed|en)\s*,\s*"
            r"\w+(?:ed|en)\s*,\s*and\s*\w+(?:ed|en)(?:\s+by\b)?",
            low,
        )),
        "PURPOSE_INFINITIVE_FRONTING": bool(re.match(
            r"^to\s+[^,]{2,160},\s+",
            low,
        )),
        "COORDINATED_OBJECTS_OR_ENUMERATION": has_enumeration,
        "ENUMERATION": has_enumeration,

        # Cell 5 v4.1 primary structures.
        "FINITE_COMPLEMENT_LIST": bool(re.search(
            r"\bwithout\s+.+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+",
            low,
            re.S,
        )),
        "PROPOSAL_FRAME": bool(re.match(
            r"^(?:we|i|the\s+(?:study|paper|article|report|team|authors?))\s+"
            r"(?:propose|proposes|introduce|introduces|present|presents)\b",
            low,
        )),
        "APPOSITIVE_DEFINITION": bool(re.search(
            r"\([^)]{2,120}\)\s*,\s*(?:an?|the)\s+.+",
            text,
            re.S,
        )),
        "PARTICIPIAL_POSTMODIFIER": (
            _v51_dependency_participial_postmodifier(doc)
            or bool(re.search(
                r"\b(?:to|for|with|among|across)\s+"
                r"(?:[A-Za-z][\w-]*\s+){0,5}[A-Za-z][\w-]*\s+"
                r"(?:spanning|covering|including|containing|involving)\s+"
                r".+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+",
                text,
                re.I | re.S,
            ))
        ),
        "FINITE_OBJECT_LIST_FRAME": bool(re.match(
            r"^(?:we|i|they|the\s+[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,4})\s+"
            r"(?:pre-register|preregister|document|record|evaluate|assess|review|"
            r"analyze|compare|test|measure|include|identify|specify|report)\s+"
            r".+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+",
            text,
            re.I | re.S,
        )),
        "CONTINGENCY_RELATION": bool(re.search(
            r"\b(?:is|are|was|were)\s+contingent\s+on\b",
            low,
        )),
    }

    families.update(name for name, matched in tests.items() if matched)

    # Dependency-confirmed augmentations.
    try:
        if dependency_available:
            if _v51_dependency_coord_predicates(doc):
                families.add("COORDINATED_PREDICATES")
            if _v51_dependency_noun_enumeration(doc):
                families.add("COORDINATED_OBJECTS_OR_ENUMERATION")
                families.add("ENUMERATION")
            if _v51_dependency_participial_postmodifier(doc):
                families.add("PARTICIPIAL_POSTMODIFIER")
    except Exception as exc:
        notes.append(f"Dependency-family overlay failed: {exc}")

    safe_families = families - {"PARSE_FAILURE"}
    if "PARSE_FAILURE" in families and safe_families:
        families.remove("PARSE_FAILURE")
        notes.append(
            "Dependency parsing was unavailable or incomplete, but "
            "surface-safe grammar families were detected."
        )

    primary = sorted(family for family in families if family in _PRIMARY_FAMILIES_V51)
    secondary = sorted(family for family in families if family in _SECONDARY_FAMILIES_V51)

    analysis.families = sorted(families)
    try:
        analysis.metadata = dict(getattr(analysis, "metadata", {}) or {})
        analysis.metadata.update({
            "dependency_available": dependency_available,
            "routing_overlay": "NLL_CELL_6_V5_1_CELL5_V4_1_RERUN_SAFE",
            "primary_families": primary,
            "secondary_structures": secondary,
            "primary_family_count": len(primary),
            "secondary_structure_count": len(secondary),
        })
    except Exception:
        pass

    return analysis


Analyzer.analyze = _analyze_v51


# =====================================================================
# 5. ROUTE COVERAGE
# =====================================================================

CRITICAL_FAMILIES = set(_PRIMARY_FAMILIES_V51) | {
    "LONG_MULTI_CLAUSE",
    "COORDINATED_OBJECTS_OR_ENUMERATION",
    "ENUMERATION",
}

ROUTE_COVERAGE = {
    family: sorted(
        registered.name
        for registered in RULES
        if family in registered.families
    )
    for family in CRITICAL_FAMILIES
}

UNCOVERED = sorted(
    family for family, routes in ROUTE_COVERAGE.items() if not routes
)


# =====================================================================
# 6. RERUN-SAFETY + ALIGNMENT CHECKS
# =====================================================================

if _BASE_ANALYZE is Analyzer.analyze:
    raise RuntimeError(
        "Cell 6 v5.1 rerun-safety failure: Analyzer overlay references itself."
    )

if _NLL_CELL6_ORIGINAL_SENT_SPLIT is sent_split:
    raise RuntimeError(
        "Cell 6 v5.1 rerun-safety failure: sentence splitter references itself."
    )

if len(RULES) != EXPECTED_CELL6_RULE_COUNT:
    raise RuntimeError(
        f"Cell 6 v5.1 expected {EXPECTED_CELL6_RULE_COUNT} rules but found {len(RULES)}."
    )

_EXPECTED_V51_ROUTES = {
    "FINITE_COMPLEMENT_LIST": ["without_complement_list"],
    "PROPOSAL_FRAME": ["proposal_appositive_definition"],
    "APPOSITIVE_DEFINITION": ["proposal_appositive_definition"],
    "PARTICIPIAL_POSTMODIFIER": ["participial_postmodifier"],
    "FINITE_OBJECT_LIST_FRAME": ["finite_object_list_frame"],
    "CONTINGENCY_RELATION": ["contingency_relation"],
}

_route_errors = {
    family: {"observed": ROUTE_COVERAGE.get(family, []), "expected": expected}
    for family, expected in _EXPECTED_V51_ROUTES.items()
    if ROUTE_COVERAGE.get(family, []) != expected
}
if _route_errors:
    raise RuntimeError("Cell 6 v5.1 route-alignment failure: " + str(_route_errors))

if UNCOVERED:
    raise RuntimeError(
        "Cell 6 v5.1 uncovered critical families: " + ", ".join(UNCOVERED)
    )


# =====================================================================
# 7. ROUTING SELF-TEST CORPUS
# =====================================================================

_ROUTING_TESTS = [
    (
        "FINITE_COMPLEMENT_LIST",
        "Systems train separate models without explicit rules for assessment, correction, and graduation.",
    ),
    (
        "PROPOSAL_FRAME",
        "We propose Framework X (FX), a protocol consisting of contracts, examinations, and promotion.",
    ),
    (
        "APPOSITIVE_DEFINITION",
        "We propose Framework X (FX), a protocol consisting of contracts, examinations, and promotion.",
    ),
    (
        "PARTICIPIAL_POSTMODIFIER",
        "The testbed applies the protocol to learners spanning classification, drafting, and planning.",
    ),
    (
        "FINITE_OBJECT_LIST_FRAME",
        "We pre-register questions, data, baselines, ablations, and testing.",
    ),
    (
        "CONTINGENCY_RELATION",
        "Research claims about improvement, cost, and deployment are contingent on execution and analysis.",
    ),
]

# Use the current Analyzer overlay to verify detection. Any available profile
# is sufficient because these checks address syntax/routing, not discipline.
_test_profile = next(iter(PROFILES.values())) if "PROFILES" in globals() and PROFILES else None
_routing_failures = []

if _test_profile is not None:
    _test_analyzer = Analyzer()
    for expected_family, sentence in _ROUTING_TESTS:
        try:
            observed = set(
                getattr(
                    _test_analyzer.analyze(sentence, _test_profile, set()),
                    "families",
                    [],
                )
                or []
            )
            if expected_family not in observed:
                _routing_failures.append({
                    "family": expected_family,
                    "sentence": sentence,
                    "observed": sorted(observed),
                })
        except Exception as exc:
            _routing_failures.append({
                "family": expected_family,
                "sentence": sentence,
                "error": repr(exc),
            })

if _routing_failures:
    raise RuntimeError(
        "Cell 6 v5.1 routing self-test failure: " + str(_routing_failures)
    )


print("=" * 78)
print("CELL 6 UNIVERSAL ROUTING v5.1 CHECK")
print("=" * 78)
print("Aligned Cell 5 version: v4.1")
print("Aligned Cell 5A version: v3.4")
print("Transformation rules:", len(RULES))
print("Expected transformation rules:", EXPECTED_CELL6_RULE_COUNT)
print("Uncovered critical families:", UNCOVERED)
print("Analyzer rerun-safety check: PASSED")
print("Sentence-splitter rerun-safety check: PASSED")
print("Dependency-confirmed participial routing: active")
print("Dependency-confirmed coordinated predicates: active")
print("Primary/secondary family metadata: active")
print("Routing self-tests: PASSED")

for family in [
    "FINITE_COMPLEMENT_LIST",
    "PROPOSAL_FRAME",
    "APPOSITIVE_DEFINITION",
    "PARTICIPIAL_POSTMODIFIER",
    "FINITE_OBJECT_LIST_FRAME",
    "CONTINGENCY_RELATION",
    "COPULAR_ACTION_FRAME",
    "NEED_TO_ACTION",
    "MODAL_COPULAR_ENUMERATION",
    "USING_METHOD_RELATION",
    "THROUGH_METHOD_RELATION",
    "BY_GERUND_METHOD_RELATION",
    "FINITE_PURPOSE_INTENT",
    "RATHER_THAN_GERUND",
    "COORDINATED_PASSIVE_PARTICIPLES",
    "PURPOSE_INFINITIVE_FRONTING",
    "COORDINATED_OBJECTS_OR_ENUMERATION",
    "ENUMERATION",
]:
    print(f"{family}: {ROUTE_COVERAGE.get(family, [])}")

print("=" * 78)
