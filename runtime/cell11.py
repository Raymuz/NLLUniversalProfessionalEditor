#@title 7. VALIDATION / RANKING / ENGINE v4.3 — confidence-gated Unicode enumeration judge
# Complete replacement for Cell 7.
# Compatible with Cell 5 v4.1, Cell 5A v3.4, Cell 6 v5.1.
# Accepts arbitrary Unicode text safely; linguistic routing quality still depends
# on the loaded spaCy model and available annotations.
# Run Cell 7A and Cell 7B again after this cell.

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

EXPECTED_CELL7_RULE_COUNT = 30

_REQUIRED_CELL7 = [
    "STATE", "CFG", "RULES", "Analyzer", "Generator", "Scanner",
    "Candidate", "SentenceResult", "DocumentResult", "PROFILES",
    "protect", "optional", "nlp", "sub", "sent_split", "surface_key",
    "realization_issues",
]
_missing_cell7 = [name for name in _REQUIRED_CELL7 if name not in globals()]
if _missing_cell7:
    raise RuntimeError(
        "Cell 7 cannot run. Missing: " + ", ".join(_missing_cell7)
        + ". Run Cells 1–6 v5.1 in order first."
    )
if len(RULES) != EXPECTED_CELL7_RULE_COUNT:
    raise RuntimeError(
        f"Cell 7 expected {EXPECTED_CELL7_RULE_COUNT} rules but found {len(RULES)}."
    )

# Backward compatibility for older tests/helpers.
_surface_key = surface_key

NEG = {"not", "no", "never", "without", "neither", "nor"}
HEDGE = {
    "may", "might", "could", "can", "possibly", "potentially",
    "suggest", "suggests", "indicate", "indicates", "likely",
    "approximately", "about", "must", "should", "would", "will",
}


def _ensure_state():
    if not isinstance(STATE, dict):
        raise RuntimeError("STATE must be a dictionary.")
    STATE.setdefault("warnings", [])
    STATE.setdefault("start", None)
    return STATE


def sr(a, b):
    return SequenceMatcher(
        None,
        re.sub(r"\s+", " ", str(a).lower()).strip(),
        re.sub(r"\s+", " ", str(b).lower()).strip(),
    ).ratio()


def token_change(a, b):
    A = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", str(a).lower())
    B = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", str(b).lower())
    return 1 - SequenceMatcher(None, A, B).ratio()


def nli_pair(a, b):
    _ensure_state()
    bundle = optional("nli")
    if not bundle:
        return None
    try:
        import torch
        tok, model = bundle
        x = tok(a, b, return_tensors="pt", truncation=True, max_length=512)
        dev = next(model.parameters()).device
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.no_grad():
            prob = torch.softmax(model(**x).logits[0], -1).cpu().numpy()
        labels = {
            str(model.config.id2label[i]).lower(): float(prob[i])
            for i in range(len(prob))
        }
        return {
            "entailment": max([v for k, v in labels.items() if "entail" in k] or [0]),
            "contradiction": max([v for k, v in labels.items() if "contrad" in k] or [0]),
        }
    except Exception as exc:
        STATE["warnings"].append(f"NLI scoring failed: {exc}")
        return None


def stsim(a, b):
    _ensure_state()
    model = optional("sts")
    if not model:
        return None
    try:
        from sentence_transformers.util import cos_sim
        emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(cos_sim(emb[0], emb[1]).item())
    except Exception as exc:
        STATE["warnings"].append(f"STS scoring failed: {exc}")
        return None


# =====================================================================
# RELATION / INVENTORY HELPERS
# =====================================================================

def _safe_doc(text):
    try:
        return nlp()(str(text or ""))
    except Exception:
        return None


def _neg_inventory(text, doc=None):
    """Compare polarity markers by lexical class, not dependency owner."""
    lo = str(text or "").lower()
    return Counter(
        token for token in ["not", "no", "never", "without", "neither", "nor"]
        if re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", lo)
    )


def _neg_scope_debug(doc):
    if doc is None or not getattr(doc, "has_annotation", lambda *_: False)("DEP"):
        return []
    return sorted(
        (x.lower_, x.head.lemma_.lower())
        for x in doc
        if x.dep_ == "neg" or x.lower_ in {"never", "without", "neither", "nor"}
    )


def _modals(doc, text=""):
    if doc is not None:
        try:
            return Counter(
                x.lemma_.lower()
                for x in doc
                if x.lemma_.lower() in HEDGE or x.tag_ == "MD"
            )
        except Exception:
            pass
    return Counter(re.findall(
        r"\b(?:may|might|can|could|must|should|would|will|possibly|potentially|likely)\b",
        str(text or "").lower(),
    ))


def _purpose_contrast_preserved(original, candidate):
    match = re.search(
        r"(?is)\bnot\s+to\s+(?P<x>.+?)\s+but\s+to\s+(?P<y>.+?)[.]?$",
        original,
    )
    if not match:
        return False
    x = surface_key(match["x"])
    y = surface_key(match["y"])
    z = surface_key(candidate)
    return (
        x in z and y in z
        and (
            "rather than" in candidate.lower()
            or ("does not" in candidate.lower() and "instead" in candidate.lower())
        )
    )


def _unicode_normalize(text):
    """Canonical Unicode normalization for comparison, never for output."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    translations = str.maketrans({
        "–": "-", "—": "-", "−": "-", "‑": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "\u00a0": " ",
    })
    return re.sub(r"\s+", " ", value.translate(translations)).strip().casefold()


def _unicode_tokens(text):
    """
    Tokenize letters/numbers across Unicode scripts without ASCII loss.

    Punctuation, including every dash/apostrophe form, is treated as a
    comparison boundary. This makes these equivalent for inventory checks:
      re-examination == re examination == re–examination
      matched-data   == matched data
    The original candidate text is never modified.
    """
    value = _unicode_normalize(text)
    output = []
    current = []

    for char in value:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "M"}:
            current.append(char)
        elif current:
            token = "".join(current)
            if token:
                output.append(token)
            current = []

    if current:
        token = "".join(current)
        if token:
            output.append(token)

    return output


def _noun_chunk_for_token(doc, token):
    try:
        chunks = [chunk for chunk in doc.noun_chunks if chunk.start <= token.i < chunk.end]
    except Exception:
        chunks = []
    return min(chunks, key=lambda chunk: chunk.end - chunk.start) if chunks else None


def _clean_member_text(text):
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n,;:.()[]{}")
    value = re.sub(r"^(?:and|or|nor)\s+", "", value, flags=re.I)
    return value.strip()


def _dependency_enumeration_groups(doc):
    """Recover coordinated members from dependency heads/noun chunks."""
    if doc is None or not doc.has_annotation("DEP"):
        return []

    groups = []
    seen = set()
    for token in doc:
        conjuncts = [child for child in token.children if child.dep_ == "conj"]
        if not conjuncts:
            continue

        members = [token] + sorted(conjuncts, key=lambda item: item.i)
        # Include chained conjuncts.
        cursor = 0
        while cursor < len(members):
            for child in members[cursor].children:
                if child.dep_ == "conj" and child not in members:
                    members.append(child)
            cursor += 1
        members = sorted(set(members), key=lambda item: item.i)

        texts = []
        for member in members:
            chunk = _noun_chunk_for_token(doc, member)
            if chunk is not None:
                text = chunk.text
            else:
                # Conservative local subtree: exclude other conjunct branches.
                blocked = {item.i for other in members if other is not member for item in other.subtree}
                indices = [
                    item.i for item in member.subtree
                    if item.i not in blocked and item.dep_ not in {"cc", "punct"}
                ]
                if indices:
                    text = doc[min(indices):max(indices) + 1].text
                else:
                    text = member.text
            text = _clean_member_text(text)
            if text:
                texts.append(text)

        texts = list(dict.fromkeys(texts))
        key = tuple(_unicode_normalize(item) for item in texts)
        if len(texts) >= 3 and key not in seen:
            seen.add(key)
            groups.append(texts)

    return groups


def _analysis_enumeration_groups(analysis):
    """Accept analyzer groups only when each item is a plausible member."""
    groups = []
    for group in list(getattr(analysis, "enumerations", []) or []):
        cleaned = [_clean_member_text(item) for item in group]
        cleaned = [item for item in cleaned if item]
        # Reject boundary-corrupted pseudo-members containing multiple commas.
        if len(cleaned) >= 3 and all(item.count(",") <= 1 for item in cleaned):
            groups.append(cleaned)
    return groups


def _top_level_segments(text):
    """Split on top-level commas while respecting brackets and Unicode text."""
    segments = []
    current = []
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    for char in str(text or ""):
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers and stack and stack[-1] == char:
            stack.pop()
        if char == "," and not stack:
            segments.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        segments.append("".join(current).strip())
    return segments


def _surface_list_groups(text):
    """Conservative fallback; never cuts through hyphenated compounds."""
    value = str(text or "").strip().rstrip(".?!")
    segments = _top_level_segments(value)
    if len(segments) < 3:
        return []

    # Locate the final coordinator in the final segment.
    final = segments[-1]
    match = re.search(r"\s+(?:and|or|nor)\s+(.+)$", final, re.I | re.S)
    if not match:
        return []
    last_member = _clean_member_text(match.group(1))
    pre_final = _clean_member_text(final[:match.start()])

    # Find the most plausible list start. Prefer explicit introducers.
    introducer = re.compile(
        r"\b(?:using|including|involving|such as|comprising|consisting of|covers?|"
        r"evaluated|assessed|measured|reviewed|identified|pre-register(?:ed)?|preregister(?:ed)?)\s+",
        re.I,
    )
    first_index = 0
    first_text = segments[0]
    matches = list(introducer.finditer(first_text))
    if matches:
        first_member = _clean_member_text(first_text[matches[-1].end():])
    else:
        # Without an introducer, avoid guessing a list start from the full clause.
        return []

    middle = [_clean_member_text(item) for item in segments[1:-1]]
    members = [first_member] + middle
    if pre_final:
        members.append(pre_final)
    if last_member:
        members.append(last_member)
    members = [item for item in members if item]
    return [members] if len(members) >= 3 else []


def _plausible_enumeration_member(text):
    """
    Reject boundary-corrupted pseudo-members without weakening true omission checks.

    A valid member may contain Unicode letters, numbers, spaces, apostrophes,
    and dash variants, but should not contain clause punctuation, a full
    sentence lead-in, or an excessive token span.
    """
    value = _clean_member_text(text)
    tokens = _unicode_tokens(value)

    if not tokens or len(tokens) > 12:
        return False
    if any(mark in value for mark in [",", ";", ":", "."]):
        return False
    if re.match(
        r"(?is)^(?:we|i|they|he|she|it|the\s+(?:study|paper|article|report|"
        r"system|framework|team|model))\s+"
        r"(?:propose|proposes|introduce|introduces|present|presents|"
        r"evaluate|evaluates|assess|assesses|pre-register|preregister)\b",
        value,
    ):
        return False

    return True


def _enumeration_groups(analysis, source_text, source_doc):
    """
    Build only high-confidence list groups.

    Sources are tried independently instead of accepting the first nonempty
    extractor. Candidate groups are filtered for plausible member boundaries,
    verified against the source, de-duplicated, and ranked by compactness.
    """
    candidate_groups = []
    candidate_groups.extend(_dependency_enumeration_groups(source_doc))
    candidate_groups.extend(_analysis_enumeration_groups(analysis))
    candidate_groups.extend(_surface_list_groups(source_text))

    source_tokens = _unicode_tokens(source_text)
    accepted = []
    seen = set()

    for group in candidate_groups:
        stable_group = []
        for raw_item in group:
            item = _clean_member_text(raw_item)
            if not _plausible_enumeration_member(item):
                continue

            item_tokens = _unicode_tokens(item)
            width = len(item_tokens)
            appears_in_source = any(
                source_tokens[index:index + width] == item_tokens
                for index in range(len(source_tokens) - width + 1)
            )
            if appears_in_source:
                stable_group.append(item)

        # Preserve order and remove duplicate members.
        unique_group = []
        member_seen = set()
        for item in stable_group:
            key = tuple(_unicode_tokens(item))
            if key and key not in member_seen:
                member_seen.add(key)
                unique_group.append(item)

        normalized = tuple(tuple(_unicode_tokens(item)) for item in unique_group)
        if len(normalized) >= 3 and normalized not in seen:
            seen.add(normalized)
            accepted.append(unique_group)

    # Remove groups that are strict noisy supersets of a cleaner group.
    compact = []
    token_sets = [
        {tuple(_unicode_tokens(item)) for item in group}
        for group in accepted
    ]
    for index, group in enumerate(accepted):
        current = token_sets[index]
        noisier = any(
            other < current
            for other_index, other in enumerate(token_sets)
            if other_index != index and len(other) >= 3
        )
        if not noisier:
            compact.append(group)

    return compact


def _member_present(member, candidate):
    """Require normalized member tokens in order, independent of punctuation."""
    member_tokens = _unicode_tokens(member)
    candidate_tokens = _unicode_tokens(candidate)
    if not member_tokens:
        return True
    width = len(member_tokens)
    return any(
        candidate_tokens[index:index + width] == member_tokens
        for index in range(len(candidate_tokens) - width + 1)
    )


def _relation_signature(text):
    lo = str(text or "").lower()
    classes = {
        "cause": {"because", "therefore", "thus", "consequently", "hence", "as a result", "for this reason", "accordingly"},
        "concession": {"although", "though", "despite", "nevertheless", "however", "even so"},
        "condition": {"if", "unless", "whether"},
        "simultaneous": {"while", "at the same time"},
        "purpose": {"so that", "intended to", "designed to"},
        "absence": {"without"},
        "dependency": {"contingent on", "depend on", "depends on", "depended on"},
    }
    return {
        key
        for key, words in classes.items()
        if any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", lo) for word in words)
    }


def _triples(doc):
    triples = []
    if doc is None or not doc.has_annotation("DEP"):
        return triples
    for verb in doc:
        subjects = [x for x in verb.children if x.dep_ in {"nsubj", "nsubjpass"}]
        objects = [x for x in verb.children if x.dep_ in {"obj", "dobj", "attr"}]
        triples += [
            (sub(subject).lower(), verb.lemma_.lower(), sub(obj).lower())
            for subject in subjects
            for obj in objects
        ]
    return triples


# =====================================================================
# VALIDATOR
# =====================================================================

class Validator:
    def validate(self, original, candidate, analysis, profile, user_terms, mode):
        _ensure_state()
        revised = candidate.text
        failures = []
        reasons = []
        checks = {}

        def reject(code, why):
            if code not in failures:
                failures.append(code)
            reasons.append({"code": code, "reason": why})

        # -------------------------------------------------------------
        # Exact protected inventories
        # -------------------------------------------------------------
        hard_terms = set(getattr(profile, "immutable", set()) or set()) | set(user_terms or set())
        for kind, code in [
            ("citation", "TERMINOLOGY_REJECTION"),
            ("quantity", "QUANTITY_REJECTION"),
            ("url", "TERMINOLOGY_REJECTION"),
            ("equation", "TERMINOLOGY_REJECTION"),
            ("code", "TERMINOLOGY_REJECTION"),
        ]:
            source_inventory = Counter(
                item[1].lower()
                for item in protect(original, hard_terms)
                if item[0] == kind
            )
            candidate_inventory = Counter(
                item[1].lower()
                for item in protect(revised, hard_terms)
                if item[0] == kind
            )
            checks[kind] = source_inventory == candidate_inventory
            if source_inventory != candidate_inventory:
                reject(
                    code,
                    f"{kind} inventory changed: {dict(source_inventory)} -> {dict(candidate_inventory)}",
                )

        missing_terms = [
            term
            for term in hard_terms
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", original, re.I)
            and not re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", revised, re.I)
        ]
        if missing_terms:
            reject(
                "TERMINOLOGY_REJECTION",
                "Missing immutable/user terms: " + ", ".join(sorted(missing_terms)),
            )

        # -------------------------------------------------------------
        # Realization gate
        # -------------------------------------------------------------
        realization = list(realization_issues(revised) or [])
        checks["realization_issues"] = realization
        if realization:
            reject(
                "GRAMMAR_REJECTION",
                "Candidate failed realization checks: " + " | ".join(realization),
            )

        source_doc = _safe_doc(original)
        candidate_doc = _safe_doc(revised)

        # -------------------------------------------------------------
        # Polarity: compare marker inventory; retain dependency scope only
        # as diagnostics. Reordering 'without' is therefore allowed.
        # -------------------------------------------------------------
        source_neg = _neg_inventory(original, source_doc)
        candidate_neg = _neg_inventory(revised, candidate_doc)
        checks["negation_inventory"] = (dict(source_neg), dict(candidate_neg))
        checks["negation_scope_debug"] = (
            _neg_scope_debug(source_doc),
            _neg_scope_debug(candidate_doc),
        )

        polarity_preserved = source_neg == candidate_neg
        if (
            not polarity_preserved
            and candidate.family.startswith("NEGATED_PURPOSE_CONTRAST")
            and _purpose_contrast_preserved(original, revised)
        ):
            polarity_preserved = True

        if not polarity_preserved:
            reject(
                "POLARITY_REJECTION",
                f"Polarity-marker inventory changed: {dict(source_neg)} -> {dict(candidate_neg)}",
            )

        # -------------------------------------------------------------
        # Modality / hedging
        # -------------------------------------------------------------
        source_modals = _modals(source_doc, original)
        candidate_modals = _modals(candidate_doc, revised)
        checks["modality"] = (dict(source_modals), dict(candidate_modals))
        if source_modals != candidate_modals:
            reject(
                "MODALITY_REJECTION",
                f"Modality/hedging changed: {dict(source_modals)} -> {dict(candidate_modals)}",
            )

        # -------------------------------------------------------------
        # Enumeration preservation across the WHOLE candidate text.
        # A split/reordered candidate is allowed if every source list member
        # survives in token order somewhere in the complete candidate.
        # -------------------------------------------------------------
        enumeration_groups = _enumeration_groups(analysis, original, source_doc)
        missing_by_group = []
        for group_index, group in enumerate(enumeration_groups, start=1):
            missing = [member for member in group if not _member_present(member, revised)]
            if missing:
                missing_by_group.append({"group": group_index, "missing": missing})

        checks["enumeration_groups"] = enumeration_groups
        checks["enumeration_missing"] = missing_by_group
        if missing_by_group:
            detail = " | ".join(
                f"group {entry['group']}: " + ", ".join(entry["missing"])
                for entry in missing_by_group
            )
            reject(
                "ENUMERATION_REJECTION",
                "Enumeration members missing: " + detail,
            )

        # -------------------------------------------------------------
        # Subject/object orientation
        # -------------------------------------------------------------
        source_triples = _triples(source_doc)
        candidate_triples = _triples(candidate_doc)
        checks["source_triples"] = source_triples
        checks["candidate_triples"] = candidate_triples
        for subject, verb, obj in source_triples:
            if subject != obj and any(
                cs == obj and cq == subject and cv == verb
                for cs, cv, cq in candidate_triples
            ):
                reject(
                    "SEMANTIC_REJECTION",
                    f"Subject/object orientation reversed for '{verb}'.",
                )

        if not source_triples and not candidate_triples:
            simple = (
                r"(?is)^\s*(?P<s>[A-Z][A-Za-z0-9_-]*)\s+"
                r"(?P<v>\w+(?:ed|s))\s+(?P<o>[A-Z][A-Za-z0-9_-]*)"
            )
            source_match = re.match(simple, original)
            candidate_match = re.match(simple, revised)
            if (
                source_match and candidate_match
                and source_match["s"].lower() == candidate_match["o"].lower()
                and source_match["o"].lower() == candidate_match["s"].lower()
                and source_match["v"].lower() == candidate_match["v"].lower()
            ):
                reject(
                    "SEMANTIC_REJECTION",
                    f"Subject/object orientation reversed for '{source_match['v']}'.",
                )

        # -------------------------------------------------------------
        # Discourse relations
        # -------------------------------------------------------------
        source_relations = _relation_signature(original)
        candidate_relations = _relation_signature(revised)
        checks["relations"] = (sorted(source_relations), sorted(candidate_relations))
        if "concession" in candidate_relations - source_relations:
            reject(
                "SEMANTIC_REJECTION",
                f"Candidate introduced concession: {sorted(source_relations)} -> {sorted(candidate_relations)}",
            )
        if "cause" in source_relations and not ({"cause", "purpose"} & candidate_relations):
            reject(
                "SEMANTIC_REJECTION",
                f"Causal/result relation disappeared: {sorted(source_relations)} -> {sorted(candidate_relations)}",
            )
        if "absence" in source_relations and "absence" not in candidate_relations:
            reject(
                "POLARITY_REJECTION",
                "The source's 'without' relation disappeared from the candidate.",
            )
        if "dependency" in source_relations and "dependency" not in candidate_relations:
            reject(
                "SEMANTIC_REJECTION",
                "The source dependency/contingency relation disappeared.",
            )

        # -------------------------------------------------------------
        # Optional soft semantic signals
        # -------------------------------------------------------------
        sts = stsim(original, revised)
        forward = nli_pair(original, revised)
        reverse = nli_pair(revised, original)
        checks.update(sts=sts, nli_forward=forward, nli_reverse=reverse)

        if sts is not None and sts < CFG.min_sts:
            checks["sts_warning"] = f"Low soft STS signal: {sts:.3f}"

        if forward and reverse:
            if max(forward["contradiction"], reverse["contradiction"]) > .55:
                reject("SEMANTIC_REJECTION", "Bidirectional NLI found contradiction.")
            elif min(forward["entailment"], reverse["entailment"]) < CFG.min_nli:
                checks["nli_warning"] = (
                    f"Weak entailment {forward['entailment']:.3f}/{reverse['entailment']:.3f}; "
                    "manual review advised."
                )

        # -------------------------------------------------------------
        # Rewrite magnitude
        # -------------------------------------------------------------
        surface = sr(original, revised)
        change = token_change(original, revised)
        checks.update(surface=surface, change=change, missing_terms=missing_terms)
        minimum_change = {"Minimal": .035, "Balanced": .065, "Structural": .09}[mode]

        if surface_key(original) == surface_key(revised):
            reject("INSUFFICIENT_REWRITE", "Candidate is identical to the source.")
        elif change < minimum_change:
            reject(
                "INSUFFICIENT_REWRITE",
                f"Token change {change:.3f} is below {mode} threshold {minimum_change:.3f}.",
            )

        # -------------------------------------------------------------
        # Optional local LanguageTool comparison
        # -------------------------------------------------------------
        grammar_tool = optional("lt") if getattr(CFG, "lt", False) else None
        if grammar_tool:
            try:
                delta = len(grammar_tool.check(revised)) - len(grammar_tool.check(original))
                checks["grammar_delta"] = delta
                if delta > 2:
                    reject(
                        "GRAMMAR_REJECTION",
                        f"Candidate introduced {delta} additional LanguageTool findings.",
                    )
            except Exception as exc:
                checks["grammar_warning"] = str(exc)

        checks.update(
            failures=failures,
            rejection_reasons=reasons,
            accepted=not failures,
        )
        return not failures, failures, checks


# =====================================================================
# RANKER
# =====================================================================

class Ranker:
    def rank(self, original, candidates, analysis, profile, user_terms, mode):
        good = []
        bad = []
        target = {"Minimal": .12, "Balanced": .24, "Structural": .38}[mode]

        for candidate in candidates:
            ok, failures, checks = Validator().validate(
                original, candidate, analysis, profile, user_terms, mode
            )
            change = checks.get("change", token_change(original, candidate.text))
            fit = max(0, 1 - abs(change - target) / max(target, .10))
            sts = checks.get("sts")
            semantic = .78 if sts is None else max(0, min(1, sts))
            nli_scores = [
                item["entailment"]
                for item in [checks.get("nli_forward"), checks.get("nli_reverse")]
                if item
            ]
            if nli_scores:
                semantic = .55 * semantic + .45 * min(nli_scores)

            lane_fit = (
                1.0 if candidate.lane == mode
                else .82 if (mode, candidate.lane) in {
                    ("Balanced", "Minimal"),
                    ("Balanced", "Structural"),
                    ("Structural", "Balanced"),
                }
                else .65
            )
            grammar = max(.6, 1 - .1 * max(0, checks.get("grammar_delta", 0)))
            candidate.score = (
                100 * (
                    .43 * semantic
                    + .22 * fit
                    + .15 * lane_fit
                    + .12 * grammar
                    + .08 * candidate.priority
                )
                if ok else -1e9
            )
            candidate.checks = checks
            (good if ok else bad).append(candidate)

        good.sort(key=lambda item: item.score, reverse=True)
        bad.sort(key=lambda item: (
            len(item.checks.get("failures", [])),
            -item.checks.get("change", 0),
        ))
        return good, bad


# =====================================================================
# ENGINE
# =====================================================================

class Engine:
    def __init__(self):
        self.an = Analyzer()
        self.gen = Generator()
        self.sc = Scanner()
        self.rank = Ranker()

    def one(self, sentence, profile, user_terms, mode):
        _ensure_state()
        analysis = self.an.analyze(sentence, profile, user_terms)

        if analysis.families == ["PARSE_FAILURE"]:
            return SentenceResult(
                sentence, sentence, "PARSE_FAILURE", analysis.families, None,
                "Dependency parse unavailable and no surface-safe family matched.",
                [], {}, {"detected": analysis.families},
            )

        candidates, trace = self.gen.generate(
            analysis, profile, user_terms, mode
        )

        if not candidates:
            code = (
                "NO_PATTERN_COVERAGE"
                if not trace.get("routes")
                else "CANDIDATE_GENERATION_FAILURE"
            )
            return SentenceResult(
                sentence, sentence, code, analysis.families, None,
                "No transformation produced a distinct candidate for the detected structures.",
                [], {}, trace,
            )

        good, bad = self.rank.rank(
            sentence, candidates, analysis, profile, user_terms, mode
        )

        trace["candidate_traces"] = [
            {
                "text": item.text,
                "lane": item.lane,
                "family": item.family,
                "reason": item.reason,
                "accepted": item.checks.get("accepted"),
                "failures": item.checks.get("failures", []),
                "rejection_reasons": item.checks.get("rejection_reasons", []),
                "change": item.checks.get("change"),
                "sts": item.checks.get("sts"),
                "nli_forward": item.checks.get("nli_forward"),
                "nli_reverse": item.checks.get("nli_reverse"),
                "score": item.score,
            }
            for item in good + bad
        ]

        if not good:
            counts = Counter(
                code
                for item in bad
                for code in item.checks.get("failures", [])
            )
            precedence = [
                "TERMINOLOGY_REJECTION",
                "POLARITY_REJECTION",
                "MODALITY_REJECTION",
                "QUANTITY_REJECTION",
                "ENUMERATION_REJECTION",
                "GRAMMAR_REJECTION",
                "SEMANTIC_REJECTION",
                "INSUFFICIENT_REWRITE",
            ]
            status = next((code for code in precedence if counts[code]), "RANKING_EMPTY")
            return SentenceResult(
                sentence, sentence, status, analysis.families, None,
                "Candidates were generated, but none passed all hard safeguards and rewrite gates.",
                [item.text for item in bad[:3]],
                bad[0].checks if bad else {},
                trace | {"rejections": dict(counts)},
            )

        selected = good[0]
        trace["selected"] = {
            "text": selected.text,
            "lane": selected.lane,
            "family": selected.family,
            "score": selected.score,
        }
        return SentenceResult(
            sentence, selected.text, "SUCCESS", analysis.families,
            selected.family, selected.reason,
            [item.text for item in good[1:4]],
            selected.checks, trace,
        )

    def process(self, text, workflow, mode, discipline, terms_text, progress=None):
        STATE.clear()
        STATE.update(start=time.time(), warnings=[])
        profile, evidence = self.route(text, discipline)
        user_terms = {
            item.strip()
            for item in re.split(r"[,;\n]", terms_text or "")
            if item.strip()
        }
        findings, score, band, features = self.sc.scan(text)

        if workflow == "Scan Only":
            return DocumentResult(
                text, text, workflow, mode, profile.name, evidence,
                score, band, findings, [], STATE["warnings"],
                {"features": features},
            )

        paragraphs = [
            item.strip()
            for item in re.split(r"\n\s*\n", text)
            if item.strip()
        ]
        total = sum(len(sent_split(item)) for item in paragraphs)
        done = 0
        results = []
        output_paragraphs = []

        for block_start in range(0, len(paragraphs), CFG.batch):
            for paragraph in paragraphs[block_start:block_start + CFG.batch]:
                output_sentences = []
                for sentence in sent_split(paragraph):
                    targeted = (
                        workflow == "Edit / Paraphrase Only"
                        or any(finding.sentence == len(results) for finding in findings)
                    )
                    if targeted:
                        item = self.one(sentence, profile, user_terms, mode)
                    else:
                        item = SentenceResult(
                            sentence, sentence, "NOT_TARGETED", [], None,
                            "No scanner finding targeted this sentence.",
                            [], {}, {},
                        )
                    results.append(item)
                    output_sentences.append(item.revised)
                    done += 1
                    if progress:
                        progress(
                            done / max(total, 1),
                            desc=f"Sentence {done}/{total}",
                        )
                output_paragraphs.append(" ".join(output_sentences))

        return DocumentResult(
            text,
            "\n\n".join(output_paragraphs),
            workflow,
            mode,
            profile.name,
            evidence,
            score,
            band,
            findings,
            results,
            STATE["warnings"],
            {
                "sentences": total,
                "success": sum(item.status == "SUCCESS" for item in results),
                "unchanged": sum(item.original == item.revised for item in results),
                "seconds": time.time() - STATE["start"],
            },
        )

    def route(self, text, requested):
        if requested in PROFILES:
            return PROFILES[requested], ["manual"]

        lo = text.lower()
        scores = {
            name: sum(
                bool(re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", lo))
                for term in profile.keywords | profile.phrases
            )
            for name, profile in PROFILES.items()
        }
        winner = max(scores, key=scores.get)
        winner = winner if scores[winner] else "General Academic"
        evidence = [
            term for term in PROFILES[winner].keywords
            if term.lower() in lo
        ][:10]
        return PROFILES[winner], evidence


ENGINE = Engine()


# =====================================================================
# REPORT WRITER
# =====================================================================

from docx import Document


def report(result):
    folder = Path(CFG.out) / ("NLL_" + time.strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=True)

    revised_path = folder / "revised.txt"
    diagnostics_path = folder / "diagnostics.json"
    revised_path.write_text(result.revised, encoding="utf8")
    diagnostics_path.write_text(
        json.dumps(asdict(result), indent=2, default=str),
        encoding="utf8",
    )

    document = Document()
    document.add_heading("NLL Revised Document", 0)
    document.add_paragraph(result.revised)
    document.add_heading("Review", 1)
    document.add_paragraph(
        f"Writing Pattern Score: {result.pattern_score:.0f}/100 ({result.band}). "
        "The score describes observable structural and stylistic patterns only."
    )

    for index, item in enumerate(result.sentences, 1):
        document.add_heading(f"Sentence {index}: {item.status}", 2)
        document.add_paragraph("Original: " + item.original)
        document.add_paragraph("Revised: " + item.revised)
        document.add_paragraph("Reason: " + item.reason)
        document.add_paragraph("Detected structures: " + ", ".join(item.families))
        document.add_paragraph(
            "Diagnostics: " + json.dumps(item.diagnostics, default=str)[:4000]
        )

    doc_path = folder / "revised_and_review.docx"
    document.save(doc_path)
    return [str(doc_path), str(revised_path), str(diagnostics_path)]


# =====================================================================
# SELF-TESTS
# =====================================================================

_ensure_state()

# 1. Polarity movement: reordering 'without' must not be rejected simply
# because its dependency head changed.
_test_profile = next(iter(PROFILES.values()))
_test_analyzer = Analyzer()
_pol_source = (
    "Systems train separate models without explicit rules for assessment, "
    "correction, and graduation."
)
_pol_candidate = Candidate(
    "Without explicit rules for assessment, correction, and graduation, "
    "systems train separate models.",
    "Balanced",
    "FINITE_COMPLEMENT_LIST",
    "validator self-test",
    .9,
)
_pol_analysis = _test_analyzer.analyze(_pol_source, _test_profile, set())
_pol_ok, _pol_fail, _ = Validator().validate(
    _pol_source, _pol_candidate, _pol_analysis, _test_profile, set(), "Balanced"
)
if "POLARITY_REJECTION" in _pol_fail:
    raise RuntimeError(
        "Cell 7 v4.3 failed: polarity-preserving 'without' movement was rejected."
    )

# 2. Enumeration must survive when the candidate reorganizes/splits text.
_enum_source = "We propose Framework X, a protocol consisting of contracts, examinations, and promotion."
_enum_candidate = Candidate(
    "Framework X is a protocol consisting of contracts, examinations, and promotion proposed by us.",
    "Balanced",
    "APPOSITIVE_DEFINITION",
    "validator self-test",
    .9,
)
_enum_analysis = _test_analyzer.analyze(_enum_source, _test_profile, set())
_, _enum_fail, _ = Validator().validate(
    _enum_source, _enum_candidate, _enum_analysis, _test_profile, set(), "Balanced"
)
if "ENUMERATION_REJECTION" in _enum_fail:
    raise RuntimeError(
        "Cell 7 v4.3 failed: preserved enumeration was rejected."
    )

# 3. Malformed realization must fail at the judge even if it somehow bypasses 5A.
_bad_candidate = Candidate(
    "Protocol to learners spanning classification, drafting, and planning.",
    "Structural",
    "PARTICIPIAL_POSTMODIFIER",
    "validator self-test",
    .9,
)
_bad_analysis = _test_analyzer.analyze(
    "The testbed applies the protocol to learners spanning classification, drafting, and planning.",
    _test_profile,
    set(),
)
_, _bad_fail, _ = Validator().validate(
    "The testbed applies the protocol to learners spanning classification, drafting, and planning.",
    _bad_candidate,
    _bad_analysis,
    _test_profile,
    set(),
    "Structural",
)
if "GRAMMAR_REJECTION" not in _bad_fail:
    raise RuntimeError(
        "Cell 7 v4.3 failed: malformed participial fragment was not rejected."
    )

# 4. Appositive before the true list must not become a pseudo-member.
_long_enum_source = (
    "We propose Framework X (FX), a protocol consisting of native model contracts, "
    "historical contracts, protected examinations, peer study, and promotion."
)
_long_enum_candidate = Candidate(
    "Framework X (FX) is a protocol consisting of native model contracts, "
    "historical contracts, protected examinations, peer study, and promotion proposed by us.",
    "Balanced", "APPOSITIVE_DEFINITION", "validator self-test", .9,
)
_long_enum_analysis = _test_analyzer.analyze(_long_enum_source, _test_profile, set())
_, _long_enum_fail, _ = Validator().validate(
    _long_enum_source, _long_enum_candidate, _long_enum_analysis,
    _test_profile, set(), "Balanced"
)
if "ENUMERATION_REJECTION" in _long_enum_fail:
    raise RuntimeError(
        "Cell 7 v4.3 failed: appositive boundary corrupted enumeration extraction."
    )

# 5. Hyphenated list members must remain intact during comparison.
_hyphen_source = (
    "We pre-register baseline questions, disjoint re-examination data, matched-data "
    "baselines, ablations, and independently held-out testing."
)
_hyphen_candidate = Candidate(
    "Our preregistration covers baseline questions, disjoint re-examination data, "
    "matched-data baselines, ablations, and independently held-out testing.",
    "Balanced", "FINITE_OBJECT_LIST_FRAME", "validator self-test", .9,
)
_hyphen_analysis = _test_analyzer.analyze(_hyphen_source, _test_profile, set())
_, _hyphen_fail, _ = Validator().validate(
    _hyphen_source, _hyphen_candidate, _hyphen_analysis,
    _test_profile, set(), "Balanced"
)
if "ENUMERATION_REJECTION" in _hyphen_fail:
    raise RuntimeError(
        "Cell 7 v4.3 failed: hyphenated list member was split or lost."
    )

# 6. Unicode punctuation variants must compare as the same content tokens.
if _unicode_tokens("re-examination data") != _unicode_tokens("re examination data"):
    raise RuntimeError(
        "Cell 7 v4.3 failed: ASCII hyphen token normalization mismatch."
    )
if _unicode_tokens("matched–data") != _unicode_tokens("matched-data"):
    raise RuntimeError(
        "Cell 7 v4.3 failed: Unicode dash token normalization mismatch."
    )
if _unicode_tokens("ALCEF–RSF") != _unicode_tokens("ALCEF-RSF"):
    raise RuntimeError(
        "Cell 7 v4.3 failed: acronym dash normalization mismatch."
    )

# 7. Boundary-corrupted pseudo-members must be rejected, while legitimate
# multiword and hyphenated members remain eligible.
if _plausible_enumeration_member(
    "Framework X (FX), a protocol consisting of native contracts, historical contracts"
):
    raise RuntimeError(
        "Cell 7 v4.3 failed: appositive pseudo-member passed confidence gate."
    )
if not _plausible_enumeration_member("disjoint re-examination data"):
    raise RuntimeError(
        "Cell 7 v4.3 failed: valid hyphenated member failed confidence gate."
    )
if not _plausible_enumeration_member("independently held-out testing"):
    raise RuntimeError(
        "Cell 7 v4.3 failed: valid multiword member failed confidence gate."
    )

print("=" * 78)
print("CELL 7 VALIDATION / RANKING / ENGINE v4.3 LOADED")
print("=" * 78)
print("Validator: OK")
print("Ranker: OK")
print("Engine: OK")
print("ENGINE instance: OK")
print("Polarity-inventory validation: active")
print("Dependency-aware Unicode enumeration validation: active")
print("Appositive/list-boundary protection: active")
print("Hyphen/dash-insensitive content comparison: active")
print("Parser-artifact group filtering: active")
print("Enumeration-member confidence gate: active")
print("Noisy-superset group suppression: active")
print("Cell 5A realization gate rechecked by judge: active")
print("Optional STS/NLI signals: soft unless contradiction is detected")
print("Runtime warning-state safety: active")
print("Validator self-tests: PASSED")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_CELL7_RULE_COUNT)
print("=" * 78)
