#@title 5. STANDALONE UNIVERSAL TRANSFORMATION ENGINE v4.1 — 30 deterministic rules
# Complete standalone replacement for Cell 5.
# Requires earlier notebook cells to define Candidate, CFG, Analyzer, nlp,
# optional, wn, and getInflection. This cell defines the rule registry,
# all 25 base rules, the Generator, and the v4.1 primary-clause upgrade.

from dataclasses import dataclass
from typing import Callable
from collections import Counter
import re

_REQUIRED_CELL5 = ["Candidate", "CFG", "Analyzer", "nlp"]
_missing_cell5 = [name for name in _REQUIRED_CELL5 if name not in globals()]
if _missing_cell5:
    raise RuntimeError(
        "Standalone Cell 5 cannot run. Missing: "
        + ", ".join(_missing_cell5)
        + ". Run Cells 1–4/4B first."
    )

@dataclass
class Rule:
    name: str
    lanes: set
    families: set
    priority: float
    fn: Callable

RULES = []

def rule(name, lanes, families, priority=.5):
    """Register or replace one deterministic transformation rule."""
    def deco(fn):
        global RULES
        RULES = [registered for registered in RULES if registered.name != name]
        RULES.append(
            Rule(
                str(name),
                set(lanes),
                set(families),
                float(priority),
                fn,
            )
        )
        return fn
    return deco


def clean(text):
    value = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        re.sub(r"\s+", " ", str(text or "")),
    ).strip()
    if value:
        value = value[0].upper() + value[1:]
    return value if not value or value[-1] in ".!?" else value + "."


def bare(text):
    return re.sub(r"[.?!]+$", "", clean(text)).strip()


def lower_initial(text):
    value = str(text or "").strip()
    if not value or re.match(r"^[A-Z]{2,}\b", value):
        return value
    return value[:1].lower() + value[1:]


def surface_key(text):
    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


_surface_key = surface_key


def K(text, lane, family, reason, priority=.5):
    return Candidate(
        clean(text),
        lane,
        family,
        reason,
        float(priority),
    )


def emit(text, lane, family, reason, priority=.5):
    value = clean(text)
    return [] if not value else [K(value, lane, family, reason, priority)]


def _safe_subtree_text(token):
    """Return a token subtree without requiring a separate helper cell."""
    try:
        items = list(token.subtree)
        if not items:
            return token.text
        start = min(item.i for item in items)
        end = max(item.i for item in items) + 1
        return token.doc[start:end].text
    except Exception:
        return getattr(token, "text", "")


# =====================================================================
# BASE RULES 1–25
# =====================================================================

@rule("copular_action_frame", {"Balanced", "Structural"}, {"COPULAR_ACTION_FRAME"}, .92)
def tr_copular_action(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^The\s+(?P<head>next|first|second|third|final|primary|main)\s+"
        r"(?P<noun>step|task|action|objective|goal)\s+is\s+to\s+"
        r"(?P<action>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"The {match['head']} {match['noun']} involves {match['action']}",
            "Balanced",
            "COPULAR_ACTION_FRAME",
            "Converted a copular action frame into an event frame.",
            .92,
        )
    ]


@rule("need_to_action", {"Balanced", "Structural"}, {"NEED_TO_ACTION"}, .88)
def tr_need_to(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+(?P<need>need|needs|needed)\s+to\s+"
        r"(?P<action>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    modal = "had to" if match["need"].lower() == "needed" else "must"
    return [
        K(
            f"{match['subject']} {modal} {match['action']}",
            "Balanced",
            "NEED_TO_ACTION",
            "Recast a need-to construction while retaining its subject and action.",
            .88,
        )
    ]


@rule(
    "finite_intent_rather_than",
    {"Balanced", "Structural"},
    {"FINITE_PURPOSE_INTENT", "RATHER_THAN_GERUND"},
    .90,
)
def tr_finite_intent(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+"
        r"(?P<intent>intend|intends|intended)\s+to\s+"
        r"(?P<action>.+?)\s+rather\s+than\s+"
        r"(?P<alternative>\w+ing\b.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"Rather than {match['alternative']}, {lower_initial(match['subject'])} "
            f"{match['intent']} to {match['action']}",
            "Balanced",
            "FINITE_PURPOSE_INTENT",
            "Fronted the rejected alternative while preserving the stated intent.",
            .90,
        )
    ]


@rule(
    "modal_copular_enumeration",
    {"Balanced", "Structural"},
    {"MODAL_COPULAR_ENUMERATION", "COORDINATED_OBJECTS_OR_ENUMERATION", "ENUMERATION"},
    .91,
)
def tr_modal_enumeration(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+"
        r"(?P<modal>may|might|could|can)\s+be\s+in\s+"
        r"(?P<items>.+?,\s*.+?\s*,?\s*(?:or|and)\s+.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"Possible sources of {lower_initial(match['subject'])} "
            f"{match['modal'].lower()} include {match['items']}",
            "Balanced",
            "MODAL_COPULAR_ENUMERATION",
            "Recast a modal copular list while preserving uncertainty and every member.",
            .91,
        )
    ]


@rule(
    "enumeration_frame",
    {"Minimal", "Balanced"},
    {"COORDINATED_OBJECTS_OR_ENUMERATION", "ENUMERATION"},
    .65,
)
def tr_enumeration_frame(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+(?:consists?\s+of|comprises?)\s+"
        r"(?P<items>.+?,.+?(?:and|or).+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"{match['subject']} includes {match['items']}",
        "Minimal",
        "ENUMERATION",
        "Varied an explicit enumeration frame without changing its members.",
        .65,
    )


@rule(
    "using_method_relation",
    {"Balanced", "Structural"},
    {"USING_METHOD_RELATION", "METHOD_RELATION"},
    .89,
)
def tr_using_method(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<main>.+?)\s+using\s+(?P<method>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"Using {match['method']}, {lower_initial(match['main'])}",
            "Balanced",
            "USING_METHOD_RELATION",
            "Fronted the explicitly stated method relation.",
            .89,
        )
    ]


@rule(
    "through_method_relation",
    {"Balanced", "Structural"},
    {"THROUGH_METHOD_RELATION", "METHOD_RELATION"},
    .86,
)
def tr_through_method(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<main>.+?)\s+through\s+(?P<method>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"Through {match['method']}, {lower_initial(match['main'])}",
        "Balanced",
        "THROUGH_METHOD_RELATION",
        "Fronted an explicitly stated through-method relation.",
        .86,
    )


@rule(
    "by_gerund_method",
    {"Balanced", "Structural"},
    {"BY_GERUND_METHOD_RELATION", "METHOD_RELATION"},
    .84,
)
def tr_by_gerund(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<main>.+?)\s+by\s+(?P<method>\w+ing\b.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"By {match['method']}, {lower_initial(match['main'])}",
        "Balanced",
        "BY_GERUND_METHOD_RELATION",
        "Fronted an explicitly stated by-gerund method relation.",
        .84,
    )


@rule(
    "purpose_infinitive_relocation",
    {"Balanced", "Structural"},
    {"PURPOSE_INFINITIVE_FRONTING"},
    .88,
)
def tr_purpose_front(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^To\s+(?P<purpose>[^,]{2,160}),\s*(?P<main>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"{match['main']} to {match['purpose']}",
        "Balanced",
        "PURPOSE_INFINITIVE_FRONTING",
        "Moved the explicit purpose clause after the main proposition.",
        .88,
    )


@rule(
    "purpose_contrast",
    {"Balanced", "Structural"},
    {"NEGATED_PURPOSE_CONTRAST"},
    .98,
)
def tr_purpose(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<head>.*?\b(?:purpose|aim|objective|goal)\b.*?)\s+"
        r"is\s+not\s+to\s+(?P<x>.+?)\s+but\s+to\s+(?P<y>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"{match['head']} is to {match['y']} rather than to {match['x']}",
            "Balanced",
            "NEGATED_PURPOSE_CONTRAST",
            "Reordered contrasted objectives while preserving both scopes.",
            .98,
        ),
        K(
            f"{match['head']} does not seek to {match['x']}. "
            f"Instead, it aims to {match['y']}",
            "Structural",
            "NEGATED_PURPOSE_CONTRAST",
            "Separated the excluded and intended objectives.",
            .90,
        ),
    ]


@rule(
    "evidential_complement",
    {"Balanced", "Structural"},
    {"EVIDENTIAL_COMPLEMENT"},
    .94,
)
def tr_evidence(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<src>.+?)\s+"
        r"(?:provide|provides|provided|offer|offers|offered)\s+"
        r"(?:clear\s+|strong\s+|empirical\s+)?evidence\s+that\s+"
        r"(?P<claim>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return [
        K(
            f"Evidence from {lower_initial(match['src'])} indicates that {match['claim']}",
            "Balanced",
            "EVIDENTIAL_COMPLEMENT",
            "Recast the evidential frame while preserving the claim.",
            .94,
        ),
        K(
            f"According to the evidence provided by {lower_initial(match['src'])}, "
            f"{match['claim']}",
            "Structural",
            "EVIDENTIAL_COMPLEMENT",
            "Reorganized the source and claim around the same evidential relation.",
            .76,
        ),
    ]


@rule("whether_complement", {"Balanced", "Structural"}, {"WHETHER_COMPLEMENT"}, .89)
def tr_whether(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+require(?:s|d)?\s+"
        r"(?:an?\s+)?(?:consideration|assessment|evaluation|analysis)\s+"
        r"of\s+whether\s+(?P<clause>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    subject = re.sub(r"\balso\b", "", match["subject"], flags=re.I).strip()
    return emit(
        f"Assessing whether {match['clause']} is also required when considering "
        f"{lower_initial(subject)}",
        "Balanced",
        "WHETHER_COMPLEMENT",
        "Converted a nominalized whether-complement into an assessment clause.",
        .89,
    )


@rule("gerund_subject", {"Balanced", "Structural"}, {"GERUND_SUBJECT"}, .88)
def tr_gerund(analysis, profile, user_terms):
    source = bare(analysis.text)
    match = re.match(
        r"(?is)^(?P<verb>Applying|Using|Evaluating|Comparing|Assessing|Examining|"
        r"Analyzing|Implementing|Testing|Measuring)\s+(?P<object>.+?)\s+"
        r"(?P<predicate>reduces?|provides?|enables?|supports?|allows?|improves?|"
        r"limits?|increases?|decreases?|preserves?|reveals?|shows?)\b(?P<tail>.*)$",
        source,
    )
    if not match:
        return []
    nouns = {
        "applying": "application",
        "using": "use",
        "evaluating": "evaluation",
        "comparing": "comparison",
        "assessing": "assessment",
        "examining": "examination",
        "analyzing": "analysis",
        "implementing": "implementation",
        "testing": "testing",
        "measuring": "measurement",
    }
    event = nouns[match["verb"].lower()]
    return emit(
        f"The {event} of {match['object']} {match['predicate']}{match['tail']}",
        "Balanced",
        "GERUND_SUBJECT",
        "Converted a gerund event into an explicit nominal subject.",
        .88,
    )


@rule("whereby_mechanism", {"Balanced", "Structural"}, {"WHEREBY_CLAUSE"}, .84)
def tr_whereby(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<main>.+?)\s+whereby\s+(?P<x>.+?)"
        r"(?:\s+so\s+that\s+(?P<y>.+?))?[.]?$",
        analysis.text,
    )
    if not match:
        return []
    middle = (
        f"{match['main']}; under this design, {lower_initial(match['x'])}"
        + (f", so that {match['y']}" if match.group("y") else "")
    )
    output = [
        K(
            middle,
            "Balanced",
            "WHEREBY_CLAUSE",
            "Separated the main proposition from its mechanism.",
            .84,
        )
    ]
    if match.group("y"):
        output.append(
            K(
                f"{match['main']}. Under this design, {match['x']}. "
                f"This arrangement is intended so that {match['y']}",
                "Structural",
                "WHEREBY_CLAUSE",
                "Separated the claim, mechanism, and purpose.",
                .72,
            )
        )
    return output


@rule("while_participial", {"Balanced", "Structural"}, {"WHILE_PARTICIPIAL"}, .84)
def tr_while(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<main>.+?)\s+while\s+(?P<x>\w+ing\b.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"While {match['x']}, {lower_initial(match['main'])}",
        "Balanced",
        "WHILE_PARTICIPIAL",
        "Reordered a simultaneous participial clause without changing its relation.",
        .84,
    )


@rule("fronted_adjunct", {"Balanced", "Structural"}, {"FRONTED_ADJUNCT"}, .73)
def tr_front(analysis, profile, user_terms):
    if re.match(
        r"(?is)^\s*(?:for purposes of|as used in|for the purposes of|"
        r"in this (?:agreement|contract|plan|section|policy|specification))\b",
        analysis.text,
    ):
        return []
    match = re.match(
        r"(?is)^(?P<adj>(?:For|In|Within|Under|During|Among|Across|Throughout)\s+"
        r"[^,]+),\s*(?P<main>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"{match['main']} {lower_initial(match['adj'])}",
        "Balanced",
        "FRONTED_ADJUNCT",
        "Moved an existing domain adjunct after the main proposition.",
        .73,
    )


@rule("discourse_marker", {"Minimal", "Balanced"}, {"DISCOURSE_FRONTING"}, .72)
def tr_discourse(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<mark>However|Therefore|Consequently|Thus|Moreover|Furthermore|"
        r"Nevertheless|Accordingly|Additionally),\s+(?P<main>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    synonyms = {
        "However": "Nevertheless",
        "Therefore": "For this reason",
        "Consequently": "As a result",
        "Thus": "Accordingly",
        "Moreover": "In addition",
        "Furthermore": "Additionally",
        "Nevertheless": "Even so",
        "Accordingly": "Therefore",
        "Additionally": "In addition",
    }
    return emit(
        f"{synonyms[match['mark']]}, {match['main']}",
        "Minimal",
        "DISCOURSE_FRONTING",
        "Varied an existing discourse marker within the same relation class.",
        .62,
    )


@rule(
    "coordinated_passive_participles",
    {"Minimal", "Balanced"},
    {"COORDINATED_PASSIVE_PARTICIPLES", "COORDINATED_PREDICATES"},
    .86,
)
def tr_passive_series(analysis, profile, user_terms):
    source = bare(analysis.text)
    match = re.search(
        r"(?is)\b(?P<noun>[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,5})\s+that\s+"
        r"(?:have|has|had)\s+been\s+"
        r"(?P<series>(?:\w+ly\s+)?\w+(?:ed|en)\s*,\s*"
        r"\w+(?:ed|en)\s*,\s*and\s*\w+(?:ed|en)(?:\s+by\s+[^.;]+)?)",
        source,
    )
    if not match:
        return []
    revised = source[:match.start()] + f"{match['noun']} {match['series']}" + source[match.end():]
    return emit(
        revised,
        "Minimal",
        "COORDINATED_PASSIVE_PARTICIPLES",
        "Reduced a relative passive clause while preserving its coordinated series.",
        .86,
    )


@rule("passive_using_nominal", {"Balanced", "Structural"}, {"PASSIVE_USING"}, .91)
def tr_passive_using(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+"
        r"(?:is|are|was|were|has been|have been|had been)\s+"
        r"(?P<adverb>\w+ly\s+)?evaluated\s+using\s+"
        r"(?P<object>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    adverb = (match.group("adverb") or "").strip()
    return emit(
        f"Evaluations of {lower_initial(match['subject'])} "
        f"{adverb + ' ' if adverb else ''}use {match['object']}",
        "Balanced",
        "PASSIVE_USING",
        "Recast a passive evaluation without inventing an actor.",
        .91,
    )


@rule("agentless_study_passive", {"Balanced", "Structural"}, {"PASSIVE"}, .70)
def tr_agentless(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<topic>.+?)\s+(?:has|have)\s+(?P<also>also\s+)?been\s+"
        r"(?:studied|examined|investigated)\s+"
        r"(?P<context>(?:within|in|across)\s+.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"Studies {match['context']} have {match['also'] or ''}examined {match['topic']}",
        "Balanced",
        "PASSIVE",
        "Recast an agentless research passive without inventing a specific actor.",
        .70,
    )


@rule("explicit_agent_active", {"Structural"}, {"PASSIVE"}, .82)
def tr_explicit_agent(analysis, profile, user_terms):
    match = re.match(
        r"(?is)^(?P<theme>.+?)\s+"
        r"(?:was|were|is|are|has been|have been|had been)\s+"
        r"(?P<verb>evaluated|examined|studied|investigated|tested|measured|reported|"
        r"implemented|applied|modified|trained|collected|derived|obtained|inspected|"
        r"approved|reviewed|documented)\s+by\s+(?P<agent>.+?)[.]?$",
        analysis.text,
    )
    if not match:
        return []
    return emit(
        f"{match['agent']} {match['verb']} {lower_initial(match['theme'])}",
        "Structural",
        "PASSIVE",
        "Converted passive to active because the source supplied the actor.",
        .82,
    )


@rule(
    "reduced_modifier",
    {"Minimal", "Balanced"},
    {"REDUCED_RELATIVE_OR_NP_MODIFIER"},
    .68,
)
def tr_reduced(analysis, profile, user_terms):
    source = bare(analysis.text)
    output = []
    for pattern, replacement in [
        (r"\bresults?\s+obtained\s+from\b", "results derived from"),
        (r"\bsamples?\s+obtained\s+from\b", "samples drawn from"),
        (r"\bdata\s+collected\s+from\b", "data gathered from"),
        (r"\bresults?\s+reported\s+independently\b", "independently reported results"),
    ]:
        if re.search(pattern, source, re.I):
            output.append(
                K(
                    re.sub(pattern, replacement, source, count=1, flags=re.I),
                    "Minimal",
                    "REDUCED_RELATIVE_OR_NP_MODIFIER",
                    "Recast a reduced participial noun modifier.",
                    .68,
                )
            )
    return output


@rule(
    "parenthetical_emphasis",
    {"Minimal", "Balanced"},
    {"PARENTHETICAL_EMPHASIS"},
    .78,
)
def tr_parenthetical(analysis, profile, user_terms):
    revised = re.sub(
        r",\s*by\s+(?:itself|themselves)\s*,",
        " alone",
        analysis.text,
        flags=re.I,
    )
    if surface_key(revised) == surface_key(analysis.text):
        return []
    return emit(
        revised,
        "Minimal",
        "PARENTHETICAL_EMPHASIS",
        "Integrated parenthetical emphasis into the clause.",
        .78,
    )


@rule("coordinated_predicate_split", {"Structural"}, {"COORDINATED_PREDICATES"}, .80)
def tr_coord_base(analysis, profile, user_terms):
    try:
        document = nlp()(bare(analysis.text))
        if not document.has_annotation("DEP"):
            return []
        root = next((token for token in document if token.dep_ == "ROOT"), None)
        if root is None:
            return []
        conjuncts = [
            token for token in root.conjuncts
            if token.pos_ in {"VERB", "AUX"}
        ]
        subjects = [
            child for child in root.children
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
        ]
        if len(conjuncts) != 1 or not subjects:
            return []
        conjunct = conjuncts[0]
        cut = min(token.i for token in conjunct.subtree)
        left = re.sub(
            r"(?:,\s*)?\b(?:and|or)\s*$",
            "",
            document[:cut].text.strip(),
            flags=re.I,
        ).strip()
        subject = _safe_subtree_text(subjects[0]).strip()
        right = _safe_subtree_text(conjunct).strip()
        if not left or not subject or not right:
            return []
        return emit(
            f"{left}. {subject} {right}",
            "Structural",
            "COORDINATED_PREDICATES",
            "Split coordinated predicates and repeated only the source subject.",
            .80,
        )
    except Exception:
        return []


@rule("semicolon_split", {"Structural"}, {"LONG_MULTI_CLAUSE"}, .65)
def tr_semicolon(analysis, profile, user_terms):
    parts = [item.strip() for item in bare(analysis.text).split(";") if item.strip()]
    if len(parts) != 2 or not all(len(item.split()) >= 3 for item in parts):
        return []
    return emit(
        f"{parts[0]}. {parts[1]}",
        "Structural",
        "LONG_MULTI_CLAUSE",
        "Split two propositions already delimited by a semicolon.",
        .65,
    )


# =====================================================================
# GENERATOR
# =====================================================================

class Generator:
    def applicable(self, analysis, lane):
        detected = set(getattr(analysis, "families", []) or [])
        return sorted(
            [
                registered
                for registered in RULES
                if lane in registered.lanes
                and bool(registered.families & detected)
            ],
            key=lambda registered: -registered.priority,
        )

    def lane(self, analysis, profile, user_terms, lane, trace):
        output = []
        seen = {surface_key(analysis.text)}
        for registered in self.applicable(analysis, lane):
            try:
                generated = registered.fn(analysis, profile, user_terms) or []
            except Exception as exc:
                trace["rule_errors"].append(
                    {
                        "rule": registered.name,
                        "lane": lane,
                        "error": str(exc),
                    }
                )
                continue
            for candidate in generated:
                if candidate is None or candidate.lane != lane:
                    continue
                key = surface_key(candidate.text)
                if key and key not in seen:
                    seen.add(key)
                    output.append(candidate)
        if lane == "Minimal" and getattr(CFG, "mlm", False):
            output.extend(self.lexical(analysis, profile, user_terms, trace))
        return output[:16]

    def generate(self, analysis, profile, user_terms, mode):
        if mode not in {"Minimal", "Balanced", "Structural"}:
            mode = "Balanced"
        trace = {
            "detected_families": list(getattr(analysis, "families", []) or []),
            "applicable": {},
            "generated_by_lane": {},
            "rule_errors": [],
            "candidate_traces": [],
        }
        pools = {}
        for lane in ["Minimal", "Balanced", "Structural"]:
            trace["applicable"][lane] = [
                registered.name
                for registered in self.applicable(analysis, lane)
            ]
            pools[lane] = self.lane(
                analysis,
                profile,
                user_terms,
                lane,
                trace,
            )
            trace["generated_by_lane"][lane] = len(pools[lane])

        order = {
            "Minimal": ["Minimal"],
            "Balanced": ["Balanced", "Minimal", "Structural"],
            "Structural": ["Structural", "Balanced", "Minimal"],
        }[mode]

        output = []
        seen = set()
        for lane in order:
            for candidate in pools[lane]:
                key = surface_key(candidate.text)
                if key and key not in seen:
                    seen.add(key)
                    output.append(candidate)

        # Bounded depth-2 composition. This invokes registered functions
        # directly and never recurses through Generator.generate.
        if mode in {"Balanced", "Structural"}:
            composed = []
            for base in list(output[:8]):
                try:
                    second_analysis = Analyzer().analyze(
                        base.text,
                        profile,
                        user_terms,
                    )
                except Exception:
                    continue
                next_lanes = ["Minimal"] if mode == "Balanced" else ["Balanced", "Minimal"]
                for next_lane in next_lanes:
                    for registered in self.applicable(second_analysis, next_lane)[:4]:
                        try:
                            second_candidates = registered.fn(
                                second_analysis,
                                profile,
                                user_terms,
                            ) or []
                        except Exception:
                            continue
                        for second in second_candidates[:1]:
                            key = surface_key(second.text)
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            second.lane = mode
                            second.family = f"{base.family}+{second.family}"
                            continuation = second.reason[:1].lower() + second.reason[1:]
                            second.reason = f"{base.reason} Then {continuation}"
                            second.priority = (base.priority + second.priority) / 2.0
                            composed.append(second)
            output.extend(composed[:8])

        trace["routes"] = [
            name
            for lane in order
            for name in trace["applicable"].get(lane, [])
        ]
        trace["generated"] = len(output)
        return output, trace

    def lexical(self, analysis, profile, user_terms, trace=None):
        optional_fn = globals().get("optional")
        if not callable(optional_fn):
            return []
        pipeline = optional_fn("mlm")
        if not pipeline:
            return []
        if "wn" not in globals() or "getInflection" not in globals():
            return []
        try:
            document = nlp()(analysis.text)
        except Exception as exc:
            if trace is not None:
                trace["rule_errors"].append(
                    {
                        "rule": "contextual_lexical",
                        "lane": "Minimal",
                        "error": str(exc),
                    }
                )
            return []

        protected = {
            str(term).lower()
            for term in (
                set(getattr(profile, "immutable", set()) or set())
                | set(user_terms or set())
            )
        }
        output = []
        seen = {surface_key(analysis.text)}
        for token in document:
            if token.pos_ not in {"ADJ", "ADV", "VERB"}:
                continue
            if token.is_stop or not token.is_alpha:
                continue
            if token.lemma_.lower() in protected:
                continue
            try:
                allowed = {
                    lemma.name().replace("_", " ").lower()
                    for synset in wn.synsets(token.lemma_)[:4]
                    for lemma in synset.lemmas()
                }
                mask = pipeline.tokenizer.mask_token
                predictions = pipeline(
                    analysis.text[:token.idx]
                    + mask
                    + analysis.text[token.idx + len(token.text):],
                    top_k=12,
                )
            except Exception:
                continue
            for prediction in predictions:
                word = str(prediction.get("token_str", "")).strip().lower()
                if not word.isalpha() or word == token.lower_ or word not in allowed:
                    continue
                try:
                    inflected = getInflection(word, tag=token.tag_)
                    replacement = inflected[0] if inflected else word
                except Exception:
                    replacement = word
                revised = (
                    analysis.text[:token.idx]
                    + replacement
                    + analysis.text[token.idx + len(token.text):]
                )
                candidate = K(
                    revised,
                    "Minimal",
                    "CONTEXTUAL_LEXICAL",
                    "Contextual lexical substitution supported by WordNet and the optional masked-language model.",
                    .45,
                )
                key = surface_key(candidate.text)
                if key and key not in seen:
                    seen.add(key)
                    output.append(candidate)
                break
            if len(output) >= 3:
                break
        return output


# Base registry invariant: 25 unique names before the v4.1 upgrade.
_BASE_RULE_NAMES = [registered.name for registered in RULES]
_BASE_DUPLICATES = sorted(
    name
    for name, count in Counter(_BASE_RULE_NAMES).items()
    if count > 1
)
if _BASE_DUPLICATES:
    raise RuntimeError(
        "Standalone Cell 5 base registry has duplicate rules: "
        + ", ".join(_BASE_DUPLICATES)
    )
if len(RULES) != 25:
    raise RuntimeError(
        "Standalone Cell 5 base registry expected 25 rules but found "
        + str(len(RULES))
    )

print("=" * 78)
print("CELL 5 BASE UNIVERSAL TRANSFORMATION ENGINE LOADED")
print("=" * 78)
print("Base rules:", len(RULES))
print("Base duplicate rule names:", _BASE_DUPLICATES)
print("Generator: defined")
print("=" * 78)

#@title 5. UNIVERSAL REPLACEMENT v4.1 — production primary-clause upgrade
# REPLACE the earlier Cell 5 v4 upgrade block with this block.
# Keep the original Cell 5 v3 registry/generator above it.
#
# Design goals:
#   - deterministic and non-generative;
#   - preserve enumerations verbatim whenever possible;
#   - prefer one-sentence recasts to avoid false enumeration loss;
#   - use spaCy dependency/noun-chunk evidence when available;
#   - use bounded surface fallbacks when dependency parsing is unavailable;
#   - never create a candidate merely to force novelty;
#   - replace same-name rules without increasing the registry beyond 30.

from collections import Counter
import re


# =====================================================================
# 0. SHARED PRODUCTION HELPERS
# =====================================================================

_V41_PARTICIPLE_TO_BASE = {
    "spanning": "span",
    "covering": "cover",
    "including": "include",
    "containing": "contain",
    "involving": "involve",
}

_V41_IRREGULAR_3SG = {
    "have": "has",
    "do": "does",
    "go": "goes",
}


def _v41_distinct(source, candidate):
    return bool(candidate and surface_key(source) != surface_key(candidate))


def _v41_balanced_parentheses(text):
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in str(text or ""):
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    return not stack


def _v41_sentence_doc(text):
    try:
        doc = nlp()(str(text or ""))
        if len(doc):
            return doc
    except Exception:
        pass
    return None


def _v41_subtree_span(token):
    items = list(token.subtree)
    if not items:
        return None
    start = min(item.i for item in items)
    end = max(item.i for item in items) + 1
    return token.doc[start:end]


def _v41_noun_chunk_for_token(doc, token):
    try:
        chunks = list(doc.noun_chunks)
    except Exception:
        chunks = []

    containing = [chunk for chunk in chunks if chunk.start <= token.i < chunk.end]
    if not containing:
        return None

    # Prefer the narrowest chunk containing the syntactic head.
    return min(containing, key=lambda chunk: chunk.end - chunk.start)


def _v41_present_verb(base, singular):
    base = str(base or "").lower()
    if not singular:
        return base
    if base in _V41_IRREGULAR_3SG:
        return _V41_IRREGULAR_3SG[base]
    if re.search(r"(?:s|x|z|ch|sh|o)$", base):
        return base + "es"
    if re.search(r"[^aeiou]y$", base):
        return base[:-1] + "ies"
    return base + "s"


def _v41_token_is_singular(token):
    try:
        number = token.morph.get("Number")
        if "Sing" in number:
            return True
        if "Plur" in number:
            return False
    except Exception:
        pass

    # Conservative surface fallback.
    return not token.text.lower().endswith("s")


def _v41_list_preserved(source_fragment, candidate):
    """Require list-bearing source fragments to survive literally."""
    fragment = re.sub(r"\s+", " ", str(source_fragment or "")).strip(" .")
    candidate_norm = re.sub(r"\s+", " ", str(candidate or ""))
    return not fragment or fragment.lower() in candidate_norm.lower()


def _v41_emit_if_safe(source, text, lane, family, reason, priority):
    candidate = clean(text)
    if not _v41_distinct(source, candidate):
        return []
    if not _v41_balanced_parentheses(candidate):
        return []
    return [K(candidate, lane, family, reason, priority)]


# =====================================================================
# 26. FINITE PREDICATE + WITHOUT COMPLEMENT LIST
# =====================================================================

@rule(
    "without_complement_list",
    {"Balanced", "Structural"},
    {
        "FINITE_COMPLEMENT_LIST",
        "COORDINATED_OBJECTS_OR_ENUMERATION",
        "ENUMERATION",
        "LONG_MULTI_CLAUSE",
    },
    .92,
)
def tr_without_complement_list(analysis, profile, user_terms):
    source = bare(analysis.text)

    match = re.match(
        r"(?is)^(?P<main>.+?)\s+without\s+"
        r"(?P<missing>.+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+?)$",
        source,
    )
    if not match:
        return []

    main = match["main"].strip()
    missing = match["missing"].strip()
    outputs = []

    # Candidate A keeps the exact polarity marker and the complete list.
    text_a = f"Without {missing}, {lower_initial(main)}"
    if _v41_list_preserved(missing, text_a):
        outputs.extend(_v41_emit_if_safe(
            source,
            text_a,
            "Balanced",
            "FINITE_COMPLEMENT_LIST",
            "Fronted the existing without-complement while preserving the exact polarity marker and complete list.",
            .92,
        ))

    # Candidate B preserves attachment in sentence-final position while
    # creating a distinct clause boundary. It is useful when a validator is
    # sensitive to fronted negative adjuncts.
    text_b = f"{main}; this occurs without {missing}"
    if _v41_list_preserved(missing, text_b):
        outputs.extend(_v41_emit_if_safe(
            source,
            text_b,
            "Structural",
            "FINITE_COMPLEMENT_LIST",
            "Separated the main proposition from its existing without-complement without changing negative scope.",
            .86,
        ))

    return outputs


# =====================================================================
# 27. PROPOSAL FRAME + APPOSITIVE DEFINITION
# =====================================================================

@rule(
    "proposal_appositive_definition",
    {"Balanced", "Structural"},
    {
        "PROPOSAL_FRAME",
        "APPOSITIVE_DEFINITION",
        "COORDINATED_OBJECTS_OR_ENUMERATION",
        "ENUMERATION",
        "LONG_MULTI_CLAUSE",
    },
    .98,
)
def tr_proposal_appositive_definition(analysis, profile, user_terms):
    source = bare(analysis.text)

    match = re.match(
        r"(?is)^(?P<subject>We|I|The\s+(?:study|paper|article|report|team|authors?))\s+"
        r"(?P<verb>propose|proposes|introduce|introduces|present|presents)\s+"
        r"(?P<name>.+?)\s*,\s*"
        r"(?P<article>an?|the)\s+"
        r"(?P<definition>.+)$",
        source,
    )
    if not match:
        return []

    name = match["name"].strip()
    definition = match["definition"].strip()
    article = match["article"].lower()

    if not _v41_balanced_parentheses(name):
        return []

    outputs = []

    # Keep the name and full appositive enumeration in one sentence so
    # whole-candidate enumeration comparison remains stable.
    text_a = (
        f"{name} is {article} {definition} proposed by "
        f"{lower_initial(match['subject'])}"
    )
    if _v41_list_preserved(definition, text_a):
        outputs.extend(_v41_emit_if_safe(
            source,
            text_a,
            "Balanced",
            "APPOSITIVE_DEFINITION",
            "Converted the proposal-plus-appositive frame into one definitional sentence while preserving the complete definition.",
            .98,
        ))

    # A second one-sentence option retains the source's proposal predicate.
    text_b = (
        f"{match['subject']} {match['verb']} {name} as "
        f"{article} {definition}"
    )
    if _v41_list_preserved(definition, text_b):
        outputs.extend(_v41_emit_if_safe(
            source,
            text_b,
            "Structural",
            "PROPOSAL_FRAME",
            "Recast the appositive as an explicit as-complement while preserving the name and complete definition.",
            .94,
        ))

    return outputs


# =====================================================================
# 28. PARTICIPIAL POSTMODIFIER
# =====================================================================

@rule(
    "participial_postmodifier",
    {"Balanced", "Structural"},
    {
        "PARTICIPIAL_POSTMODIFIER",
        "COORDINATED_OBJECTS_OR_ENUMERATION",
        "ENUMERATION",
        "NOMINALIZATION",
    },
    .94,
)
def tr_participial_postmodifier(analysis, profile, user_terms):
    source = bare(analysis.text)
    doc = _v41_sentence_doc(source)
    outputs = []

    if doc is not None and doc.has_annotation("DEP"):
        targets = [
            token
            for token in doc
            if token.text.lower() in _V41_PARTICIPLE_TO_BASE
            and token.pos_ in {"VERB", "ADJ"}
        ]

        for participle in targets:
            # In common parses, the participial modifier is attached to the
            # noun it modifies through acl/relcl or an equivalent relation.
            head = participle.head
            if head is participle or head.pos_ not in {"NOUN", "PROPN", "PRON"}:
                continue

            chunk = _v41_noun_chunk_for_token(doc, head)
            if chunk is None:
                chunk_start = head.left_edge.i
                chunk_end = head.i + 1
            else:
                chunk_start = chunk.start
                chunk_end = chunk.end

            if participle.i < chunk_end:
                continue

            # The complement is the participle subtree excluding the
            # participle token itself. This preserves enumerated material.
            subtree = _v41_subtree_span(participle)
            if subtree is None:
                continue

            complement_start = participle.i + 1
            complement_end = subtree.end
            if complement_start >= complement_end:
                continue

            complement = doc[complement_start:complement_end].text.strip()
            if not complement:
                continue

            base = _V41_PARTICIPLE_TO_BASE[participle.text.lower()]
            finite = _v41_present_verb(base, _v41_token_is_singular(head))

            prefix = doc[:participle.i].text.rstrip()
            suffix = doc[complement_end:].text.lstrip()
            candidate = f"{prefix} that {finite} {complement}"
            if suffix:
                candidate += " " + suffix

            if _v41_list_preserved(complement, candidate):
                outputs.extend(_v41_emit_if_safe(
                    source,
                    candidate,
                    "Balanced",
                    "PARTICIPIAL_POSTMODIFIER",
                    "Converted an attested participial noun modifier into a finite relative clause while preserving its head and complement.",
                    .94,
                ))

        if outputs:
            return outputs

    # Bounded fallback for environments without a dependency parse.
    fallback = re.match(
        r"(?is)^(?P<prefix>.+?\b(?:to|for|with|among|across)\s+)"
        r"(?P<head>(?:[A-Za-z][\w-]*\s+){0,5}[A-Za-z][\w-]*)\s+"
        r"(?P<participle>spanning|covering|including|containing|involving)\s+"
        r"(?P<items>.+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+?)$",
        source,
    )
    if not fallback:
        return []

    head = fallback["head"].strip()
    final_noun = head.split()[-1]
    singular = not final_noun.lower().endswith("s")
    base = _V41_PARTICIPLE_TO_BASE[fallback["participle"].lower()]
    finite = _v41_present_verb(base, singular)
    candidate = (
        f"{fallback['prefix']}{head} that {finite} {fallback['items']}"
    )

    if not _v41_list_preserved(fallback["items"], candidate):
        return []

    return _v41_emit_if_safe(
        source,
        candidate,
        "Balanced",
        "PARTICIPIAL_POSTMODIFIER",
        "Converted a bounded participial postmodifier into a finite relative clause while preserving its noun phrase and complete list.",
        .88,
    )


# =====================================================================
# 29. FINITE VERB + COMPLEX OBJECT LIST
# =====================================================================

@rule(
    "finite_object_list_frame",
    {"Balanced", "Structural"},
    {
        "FINITE_OBJECT_LIST_FRAME",
        "COORDINATED_OBJECTS_OR_ENUMERATION",
        "ENUMERATION",
        "LONG_MULTI_CLAUSE",
    },
    .88,
)
def tr_finite_object_list_frame(analysis, profile, user_terms):
    source = bare(analysis.text)

    match = re.match(
        r"(?is)^(?P<subject>We|I|They|The\s+[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,4})\s+"
        r"(?P<verb>pre-register|preregister|document|record|evaluate|assess|review|"
        r"analyze|compare|test|measure|include|identify|specify|report)\s+"
        r"(?P<items>.+?,\s*.+?(?:,\s*)?(?:and|or)\s+.+?)$",
        source,
    )
    if not match:
        return []

    nominal = {
        "pre-register": "preregistration",
        "preregister": "preregistration",
        "document": "documentation",
        "record": "record",
        "evaluate": "evaluation",
        "assess": "assessment",
        "review": "review",
        "analyze": "analysis",
        "compare": "comparison",
        "test": "testing",
        "measure": "measurement",
        "include": "scope",
        "identify": "identification",
        "specify": "specification",
        "report": "report",
    }.get(match["verb"].lower())

    if not nominal:
        return []

    subject = match["subject"].strip()
    possessive = {"we": "Our", "i": "My", "they": "Their"}.get(subject.lower())

    if possessive:
        candidate = f"{possessive} {nominal} covers {match['items']}"
    else:
        candidate = (
            f"The {nominal} conducted by {lower_initial(subject)} "
            f"covers {match['items']}"
        )

    if not _v41_list_preserved(match["items"], candidate):
        return []

    return _v41_emit_if_safe(
        source,
        candidate,
        "Balanced",
        "FINITE_OBJECT_LIST_FRAME",
        "Recast the finite predicate as an event frame while preserving every listed object verbatim.",
        .88,
    )


# =====================================================================
# 30. CONTINGENCY / DEPENDENCY RELATION
# =====================================================================

@rule(
    "contingency_relation",
    {"Minimal", "Balanced", "Structural"},
    {
        "CONTINGENCY_RELATION",
        "COORDINATED_OBJECTS_OR_ENUMERATION",
        "ENUMERATION",
        "NOMINALIZATION",
    },
    .96,
)
def tr_contingency_relation(analysis, profile, user_terms):
    source = bare(analysis.text)

    match = re.match(
        r"(?is)^(?P<subject>.+?)\s+"
        r"(?P<copula>is|are|was|were)\s+contingent\s+on\s+"
        r"(?P<basis>.+)$",
        source,
    )
    if not match:
        return []

    verb = {
        "is": "depends",
        "are": "depend",
        "was": "depended",
        "were": "depended",
    }[match["copula"].lower()]

    return _v41_emit_if_safe(
        source,
        f"{match['subject']} {verb} on {match['basis']}",
        "Minimal",
        "CONTINGENCY_RELATION",
        "Recast the contingent-on relation as an equivalent dependency relation.",
        .96,
    )


# =====================================================================
# 31. SAFER COORDINATED-PREDICATE SPLIT
# =====================================================================

@rule(
    "coordinated_predicate_split",
    {"Structural"},
    {"COORDINATED_PREDICATES"},
    .86,
)
def tr_coord_safe(analysis, profile, user_terms):
    try:
        document = nlp()(bare(analysis.text))
        if not document.has_annotation("DEP"):
            return []

        root = next((token for token in document if token.dep_ == "ROOT"), None)
        if root is None or root.pos_ not in {"VERB", "AUX"}:
            return []

        subjects = [
            child for child in root.children
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
        ]
        conjuncts = [
            token for token in root.conjuncts
            if token.pos_ in {"VERB", "AUX"}
            and token.dep_ == "conj"
            and token.i > root.i
        ]
        if len(subjects) != 1 or len(conjuncts) != 1:
            return []

        conjunct = conjuncts[0]
        between = document[root.i + 1:conjunct.i].text
        if not re.search(r"(?:,\s*)?\b(?:and|or)\b", between, re.I):
            return []

        subject = sub(subjects[0]).strip()
        right = sub(conjunct).strip()
        cut = min(token.i for token in conjunct.subtree)
        left = re.sub(
            r"(?:,\s*)?\b(?:and|or)\s*$",
            "",
            document[:cut].text.strip(),
            flags=re.I,
        ).strip()

        if not left or not subject or not right:
            return []
        if len(left.split()) < 4:
            return []
        if re.match(r"(?is)^(?:propose|introduce|present)\b", right):
            return []

        return _v41_emit_if_safe(
            bare(analysis.text),
            f"{left}. {subject} {right}",
            "Structural",
            "COORDINATED_PREDICATES",
            "Split two dependency-verified coordinated predicates and repeated only the attested subject.",
            .86,
        )
    except Exception:
        return []


# =====================================================================
# V4.1 SELF-TESTS AND REGISTRY CHECK
# =====================================================================

_V41_REQUIRED_RULES = {
    "without_complement_list",
    "proposal_appositive_definition",
    "participial_postmodifier",
    "finite_object_list_frame",
    "contingency_relation",
    "coordinated_predicate_split",
}

_v41_names = [registered.name for registered in RULES]
_v41_missing = sorted(_V41_REQUIRED_RULES - set(_v41_names))
_v41_duplicates = sorted(
    name for name, count in Counter(_v41_names).items() if count > 1
)

if _v41_missing:
    raise RuntimeError("Cell 5 v4.1 missing rules: " + ", ".join(_v41_missing))
if _v41_duplicates:
    raise RuntimeError("Cell 5 v4.1 duplicate rules: " + ", ".join(_v41_duplicates))
if len(RULES) != 30:
    raise RuntimeError(
        "Cell 5 v4.1 expected 30 rules but found " + str(len(RULES))
    )

# Parser-independent invariant tests.
if not _v41_balanced_parentheses("Framework (ALCEF-RSF)"):
    raise RuntimeError("Cell 5 v4.1 parenthesis invariant failed.")
if _v41_balanced_parentheses("Framework (ALCEF-RSF"):
    raise RuntimeError("Cell 5 v4.1 unbalanced-parenthesis test failed.")
if not _v41_list_preserved("A, B, and C", "The scope covers A, B, and C."):
    raise RuntimeError("Cell 5 v4.1 list-preservation invariant failed.")
if _v41_present_verb("span", True) != "spans":
    raise RuntimeError("Cell 5 v4.1 singular conjugation test failed.")
if _v41_present_verb("span", False) != "span":
    raise RuntimeError("Cell 5 v4.1 plural conjugation test failed.")

print("=" * 78)
print("CELL 5 UNIVERSAL TRANSFORMATION ENGINE v4.1 LOADED")
print("=" * 78)
print("Dependency-aware noun-head extraction: active")
print("Bounded surface fallback: active")
print("Literal list-preservation checks: active")
print("One-sentence proposal/appositive recasts: active")
print("Polarity-marker preservation candidates: active")
print("Finite relative-clause postmodifier recast: active")
print("Safer coordinated-predicate split: active")
print("Missing v4.1 rules:", _v41_missing)
print("Duplicate rule names:", _v41_duplicates)
print("Registered transformation rules:", len(RULES))
print("Expected rule count: 30")
print("Invariant self-tests: PASSED")
print("=" * 78)
