#@title 5A. CANDIDATE REALIZATION RELIABILITY PATCH v3.4
# Complete replacement for Cell 5A v3.3.
# Run immediately after Cell 5 v4.1 and before Cell 6 v5.
#
# Purpose:
#   - reject malformed candidates at both generation stages;
#   - preserve valid relative clauses, appositives, lists, and terminology;
#   - remain rerun-safe after Generator is rebuilt by Cell 5;
#   - add no transformation families and change no semantic policy;
#   - expose precise rejection evidence to Cell 7, Cell 7B, and Cell 9.

import re
from collections import Counter


# =====================================================================
# 0. PREREQUISITE AND VERSION CHECKS
# =====================================================================

_REQUIRED_5A = [
    "RULES", "rule", "Candidate", "Generator", "clean", "K",
    "lower_initial", "surface_key", "nlp",
]

_missing_5a = [name for name in _REQUIRED_5A if name not in globals()]
if _missing_5a:
    raise RuntimeError(
        "Cell 5A cannot run. Missing: " + ", ".join(_missing_5a)
        + ". Run the complete Cell 5 v4.1 first."
    )

EXPECTED_5A_RULE_COUNT = 30
if len(RULES) != EXPECTED_5A_RULE_COUNT:
    raise RuntimeError(
        f"Cell 5A expected {EXPECTED_5A_RULE_COUNT} Cell 5 v4.1 rules, "
        f"but found {len(RULES)}. Rerun Cell 5 v4.1 first."
    )


# =====================================================================
# 1. HIGH-CONFIDENCE REALIZATION CHECKS
# =====================================================================

_SUBJECT_PRONOUNS = {"i", "we", "he", "she", "they", "who"}
_OBJECT_PRONOUNS = {"me", "us", "him", "her", "them", "whom"}
_FINITE_TAGS = {"VBD", "VBP", "VBZ", "MD"}
_DECLARATIVE_STARTERS = {
    "i", "we", "you", "he", "she", "it", "they",
    "this", "that", "these", "those", "the", "a", "an",
}
_RELATIVE_MARKERS = {"that", "which", "who", "whose", "where"}


def _r34_split_sentences(text):
    """Return bounded sentence units without modifying source content."""
    value = clean(text)
    try:
        document = nlp()(value)
        units = [sent.text.strip() for sent in document.sents if sent.text.strip()]
        if units:
            return units
    except Exception:
        pass

    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", value)
        if part.strip()
    ]


def _r34_parse(text):
    try:
        document = nlp()(str(text or ""))
        return document if len(document) else None
    except Exception:
        return None


def _r34_has_finite_predicate(document):
    if document is None:
        return False

    for token in document:
        if token.tag_ in _FINITE_TAGS:
            return True
        if token.pos_ in {"VERB", "AUX"} and any(
            child.dep_ in {"aux", "auxpass"} and child.tag_ in _FINITE_TAGS
            for child in token.children
        ):
            return True
    return False


def _r34_relative_clause_is_finite(document):
    """Validate finite predicates after explicit relative markers."""
    if document is None:
        return True

    for token in document:
        if token.lower_ not in _RELATIVE_MARKERS:
            continue

        following = document[token.i + 1:]
        if not following:
            return False

        if not any(
            item.tag_ in _FINITE_TAGS
            or item.pos_ == "AUX"
            for item in following
        ):
            return False

    return True


def _r34_orphan_participial_fragment(sentence):
    """
    Detect noun/preposition fragments such as:
      'Protocol to small cybersecurity learners spanning ...'
    while allowing a valid finite relative clause:
      'learners that span ...'
    """
    low = sentence.lower().strip()

    if re.search(r"\b(?:that|which|who)\s+(?:span|spans|cover|covers|include|includes|contain|contains|involve|involves)\b", low):
        return False

    document = _r34_parse(sentence)
    if document is not None and document.has_annotation("DEP"):
        root = next((token for token in document if token.dep_ == "ROOT"), None)
        if root is not None:
            if root.pos_ in {"VERB", "AUX"} and _r34_has_finite_predicate(document):
                return False

            participles = [
                token for token in document
                if token.text.lower() in {
                    "spanning", "covering", "including", "containing", "involving"
                }
            ]
            if participles and not _r34_has_finite_predicate(document):
                return True

    return bool(re.match(
        r"(?is)^(?:[A-Z][\w-]*|The|A|An)\b.+\s+"
        r"(?:spanning|covering|including|containing|involving)\s+.+$",
        sentence,
    )) and not bool(re.search(
        r"\b(?:is|are|was|were|has|have|had|do|does|did|can|could|may|might|"
        r"must|should|will|would|span|spans|cover|covers|include|includes|"
        r"contain|contains|involve|involves)\b",
        low,
    ))


def _r34_clause_issues(text):
    issues = []

    for index, sentence in enumerate(_r34_split_sentences(text), start=1):
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", sentence)
        if not words:
            issues.append(f"EMPTY_SENTENCE_{index}")
            continue

        first = words[0].lower()
        document = _r34_parse(sentence)

        if first in _DECLARATIVE_STARTERS and not _r34_has_finite_predicate(document):
            issues.append(f"NONFINITE_DECLARATIVE_FRAGMENT_{index}")

        if document is not None:
            root = next((token for token in document if token.dep_ == "ROOT"), None)
            if (
                first in _DECLARATIVE_STARTERS
                and root is not None
                and root.pos_ not in {"VERB", "AUX"}
                and not _r34_has_finite_predicate(document)
            ):
                issues.append(f"NOMINAL_ROOT_FRAGMENT_{index}")

            if not _r34_relative_clause_is_finite(document):
                issues.append(f"NONFINITE_RELATIVE_CLAUSE_{index}")
        else:
            if re.match(
                r"(?is)^(?:I|We|You|He|She|It|They|The|A|An|This|That|These|Those)\s+"
                r"(?:[A-Z][\w-]*\s+){1,8}"
                r"(?:Framework|System|Model|Protocol|Method|Procedure|Process|Team|Study)\b",
                sentence,
            ):
                issues.append(f"LIKELY_NOMINAL_FRAGMENT_{index}")

        if _r34_orphan_participial_fragment(sentence):
            issues.append(f"ORPHAN_PARTICIPIAL_FRAGMENT_{index}")

    return issues


def _r34_balanced_delimiters(text):
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in str(text or ""):
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    return not stack


def realization_issues(text):
    """Return only high-confidence grammatical/realization defects."""
    value = clean(text)
    low = value.lower()
    issues = []

    if not value:
        return ["EMPTY_OUTPUT"]

    if not _r34_balanced_delimiters(value):
        issues.append("UNBALANCED_DELIMITERS")

    if re.search(
        r"\b(?:for|to|with|by|from|of|about)\s+(?:i|we|he|she|they|who)\b",
        low,
    ):
        issues.append("SUBJECT_PRONOUN_AFTER_PREPOSITION")

    if re.match(
        r"^(?:me|us|him|her|them|whom)\s+"
        r"(?:am|is|are|was|were|must|should|can|could|may|might|will|would)\b",
        low,
    ):
        issues.append("OBJECT_PRONOUN_AS_SUBJECT")

    if re.match(
        r"^(?:inspect|distinguish|evaluate|examine|analyze|compare|apply|use|"
        r"test|measure|determine|identify|review|assess|implement|investigate)"
        r"\b.+\s+is\s+the\s+(?:next|first|second|third|final|primary|main)\s+"
        r"(?:step|task|action|objective|goal)\b",
        low,
    ):
        issues.append("BARE_ACTION_AS_NOMINAL_SUBJECT")

    if re.search(r"\b(?:is|are|was|were)\s+(?:is|are|was|were)\b", low):
        issues.append("DUPLICATED_COPULA")

    if re.search(r"\b(?:has|have|had)\s+been\s+been\b", low):
        issues.append("DUPLICATED_AUXILIARY")

    if re.search(r"\b(?:a|an)\s+(?:are|were|have|do)\b", low):
        issues.append("DETERMINER_BEFORE_FINITE_VERB")

    if re.search(
        r"\b(?:of|for|with|by|from|to|in|on|under|within|across)\s+"
        r"(?:The|A|An)\s+\w+",
        value,
    ):
        issues.append("INTERNAL_CAPITALIZED_DETERMINER")

    if re.search(r"\s+[,.!?;:]", value):
        issues.append("SPACE_BEFORE_PUNCTUATION")

    if re.search(r"[.!?]{2,}", value):
        issues.append("DUPLICATED_TERMINAL_PUNCTUATION")

    if len(re.findall(r"\b\w+\b", value)) < 3:
        issues.append("TOO_SHORT")

    issues.extend(_r34_clause_issues(value))
    return list(dict.fromkeys(issues))


# =====================================================================
# 2. RELIABLE SAME-NAME RULE OVERRIDES
# =====================================================================

@rule(
    "copular_action_frame",
    {"Balanced", "Structural"},
    {"COPULAR_ACTION_FRAME"},
    .96,
)
def tr_copular_action_reliable(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^The\s+(?P<order>next|first|second|third|final|primary|main)\s+"
        r"(?P<label>step|task|action|objective|goal)\s+is\s+to\s+"
        r"(?P<action>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []

    action = match["action"].strip()
    first_match = re.match(r"(?is)^(?P<verb>[A-Za-z-]+)(?P<rest>.*)$", action)
    if not first_match:
        return []

    verb = first_match["verb"]
    rest = first_match["rest"]
    low = verb.lower()
    irregular = {
        "be": "being", "have": "having", "begin": "beginning",
        "run": "running", "set": "setting", "get": "getting",
    }

    if low in irregular:
        gerund = irregular[low]
    elif low.endswith("ie"):
        gerund = verb[:-2] + "ying"
    elif low.endswith("e") and not low.endswith(("ee", "ye", "oe")):
        gerund = verb[:-1] + "ing"
    else:
        gerund = verb + "ing"

    candidate = K(
        f"The {match['order']} {match['label']} involves {gerund}{rest}",
        "Balanced",
        "COPULAR_ACTION_FRAME",
        "Converted a copular action frame into a grammatical event frame.",
        .96,
    )
    return [] if realization_issues(candidate.text) else [candidate]


@rule(
    "need_to_action",
    {"Balanced", "Structural"},
    {"NEED_TO_ACTION"},
    .94,
)
def tr_need_to_reliable(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+(?P<need>need|needs|needed)\s+to\s+"
        r"(?P<action>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []

    modal = "had to" if match["need"].lower() == "needed" else "must"
    candidate = K(
        f"{match['subject'].strip()} {modal} {match['action'].strip()}",
        "Balanced",
        "NEED_TO_ACTION",
        "Recast a need-to construction as a subject-preserving necessity frame.",
        .94,
    )
    return [] if realization_issues(candidate.text) else [candidate]


@rule(
    "modal_copular_enumeration",
    {"Balanced", "Structural"},
    {"MODAL_COPULAR_ENUMERATION", "COORDINATED_OBJECTS_OR_ENUMERATION", "ENUMERATION"},
    .95,
)
def tr_modal_copular_enumeration_reliable(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+(?P<modal>may|might|could|can)\s+be\s+in\s+"
        r"(?P<items>.+?,\s*.+?\s*,?\s*(?:or|and)\s+.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []

    candidate = K(
        f"Possible sources of {lower_initial(match['subject'])} "
        f"{match['modal'].lower()} include {match['items']}",
        "Balanced",
        "MODAL_COPULAR_ENUMERATION",
        "Recast a modal copular list while preserving uncertainty and every member.",
        .95,
    )
    return [] if realization_issues(candidate.text) else [candidate]


@rule(
    "passive_using_nominal",
    {"Balanced", "Structural"},
    {"PASSIVE_USING"},
    .96,
)
def tr_passive_using_reliable(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?:(?P<marker>However|Therefore|Consequently|Thus|Moreover|"
        r"Furthermore|Nevertheless|Accordingly),\s+)?"
        r"(?P<subject>.+?)\s+"
        r"(?:is|are|was|were|has been|have been|had been)\s+"
        r"(?P<adverb>\w+ly\s+)?evaluated\s+using\s+"
        r"(?P<object>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []

    marker = (match.group("marker") or "").strip()
    adverb = (match.group("adverb") or "").strip()
    prefix = f"{marker}, " if marker else ""
    adverb_text = f"{adverb} " if adverb else ""

    candidate = K(
        f"{prefix}Evaluations of {lower_initial(match['subject'])} "
        f"{adverb_text}use {match['object']}",
        "Balanced",
        "PASSIVE_USING",
        "Recast a passive evaluation as an evaluation event without inventing an actor.",
        .96,
    )
    return [] if realization_issues(candidate.text) else [candidate]


# =====================================================================
# 3. FIRST-STAGE FILTER — RERUN SAFE
# =====================================================================

if not hasattr(Generator, "_nll_reliability_base_lane"):
    Generator._nll_reliability_base_lane = Generator.lane
else:
    Generator.lane = Generator._nll_reliability_base_lane


def _lane_with_realization_filter(self, analysis, profile, user_terms, lane, trace):
    candidates = Generator._nll_reliability_base_lane(
        self, analysis, profile, user_terms, lane, trace
    )
    accepted = []

    for candidate in candidates:
        issues = realization_issues(candidate.text)
        if issues:
            trace.setdefault("realization_rejections", []).append({
                "stage": "lane",
                "rule": getattr(candidate, "family", ""),
                "lane": getattr(candidate, "lane", lane),
                "text": getattr(candidate, "text", ""),
                "issues": issues,
            })
            continue
        accepted.append(candidate)

    return accepted


Generator.lane = _lane_with_realization_filter


# =====================================================================
# 4. FINAL/COMPOSED FILTER — RERUN SAFE
# =====================================================================

if not hasattr(Generator, "_nll_reliability_base_generate"):
    Generator._nll_reliability_base_generate = Generator.generate
else:
    Generator.generate = Generator._nll_reliability_base_generate


def _generate_with_final_realization_filter(
    self, analysis, profile, user_terms, mode
):
    candidates, trace = Generator._nll_reliability_base_generate(
        self, analysis, profile, user_terms, mode
    )

    accepted = []
    seen = set()

    for candidate in candidates:
        text = getattr(candidate, "text", "")
        issues = realization_issues(text)
        if issues:
            trace.setdefault("realization_rejections", []).append({
                "stage": "final_pool",
                "rule": getattr(candidate, "family", ""),
                "lane": getattr(candidate, "lane", mode),
                "text": text,
                "issues": issues,
            })
            continue

        key = surface_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        accepted.append(candidate)

    trace["generated_before_final_filter"] = len(candidates)
    trace["generated_after_final_filter"] = len(accepted)
    trace["generated"] = len(accepted)
    return accepted, trace


Generator.generate = _generate_with_final_realization_filter


# =====================================================================
# 5. SELF-TESTS
# =====================================================================

_BAD_REALIZATIONS = [
    "Inspect how the handler works is the next step.",
    "Distinguish the possibilities is necessary for we.",
    "For they, the review is necessary.",
    "The system is is evaluated.",
    "Evaluations of The service use latency measurements.",
    "We propose. We Adaptive Learning and Competency Evaluation Framework.",
    "Protocol to small cybersecurity learners spanning classification, drafting, and planning.",
    "The learners that spanning classification, drafting, and planning.",
    "The learners that classification, drafting, and planning.",
]

_GOOD_REALIZATIONS = [
    "The next step involves inspecting how the handler works.",
    "We must distinguish the possibilities using the existing source.",
    "Possible sources of the problem could include lookup, initialization, or reporting logic.",
    "Evaluations of the service use latency, throughput, and failure-rate measurements.",
    "We propose the Adaptive Learning and Competency Evaluation Framework.",
    "The experimental testbed applies this protocol to small cybersecurity learners that span classification, generative drafting, and structured diagram planning.",
    "ALCEF-RSF is a domain-independent orchestration protocol consisting of native model contracts, protected examinations, and auditable promotion proposed by us.",
]

for bad_text in _BAD_REALIZATIONS:
    if not realization_issues(bad_text):
        raise RuntimeError(
            "Cell 5A v3.4 failed to reject malformed text: " + bad_text
        )

for good_text in _GOOD_REALIZATIONS:
    detected = realization_issues(good_text)
    if detected:
        raise RuntimeError(
            "Cell 5A v3.4 incorrectly rejected valid text: "
            + good_text + " -> " + str(detected)
        )

_rule_names = [registered.name for registered in RULES]
_duplicates = sorted(
    name for name, count in Counter(_rule_names).items() if count > 1
)
if _duplicates:
    raise RuntimeError(
        "Cell 5A v3.4 duplicate rule registration: " + ", ".join(_duplicates)
    )
if len(RULES) != EXPECTED_5A_RULE_COUNT:
    raise RuntimeError(
        f"Cell 5A v3.4 expected {EXPECTED_5A_RULE_COUNT} rules but found {len(RULES)}."
    )

if getattr(Generator.lane, "__name__", "") != "_lane_with_realization_filter":
    raise RuntimeError("Cell 5A v3.4 lane wrapper was not installed.")
if getattr(Generator.generate, "__name__", "") != "_generate_with_final_realization_filter":
    raise RuntimeError("Cell 5A v3.4 generate wrapper was not installed.")

print("=" * 78)
print("CELL 5A CANDIDATE RELIABILITY PATCH v3.4 LOADED")
print("=" * 78)
print("Aligned Cell 5 version: v4.1")
print("Finite declarative-clause checks: active")
print("Finite relative-clause checks: active")
print("Orphan participial-fragment checks: active")
print("Delimiter-balance checks: active")
print("Pronoun-case checks: active")
print("Internal capitalization checks: active")
print("First-stage filtering: active")
print("Final/composed filtering: active")
print("Rerun-safe Generator.lane wrapping: active")
print("Rerun-safe Generator.generate wrapping: active")
print("Malformed candidate self-tests: PASSED")
print("Valid candidate self-tests: PASSED")
print("Duplicate rule registrations: none")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_5A_RULE_COUNT)
print("=" * 78)
