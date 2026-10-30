"""Original V5 notebook code assembled in original execution order. No LLM rewrite substitute."""
from __future__ import annotations

# ===== SOURCE NOTEBOOK CELL 2 =====
#@title 2. Runtime feature configuration (change before running the remaining cells)
# Core paraphrasing is local and deterministic. Optional analytical models download once and are cached.
ENABLE_STANZA = False       # constituency cross-check
ENABLE_STS = True           # semantic similarity (soft feature)
ENABLE_NLI = True           # bidirectional entailment gate
ENABLE_MLM = False          # masked-token lexical candidate filter only
ENABLE_LANGUAGE_TOOL = False
ENABLE_ONLINE_OEWN = False  # cached Open English WordNet fallback
DEFAULT_PARAGRAPH_BATCH = 20
print("Configuration saved. Optional models load lazily when selected in the UI.")


# ===== SOURCE NOTEBOOK CELL 3 =====
#@title 3. Contracts, resources, disciplines, and document protection

import os,re,json,html,time,traceback,statistics
from pathlib import Path
import uuid
from dataclasses import dataclass,field,asdict
from collections import Counter
from difflib import SequenceMatcher
import numpy as np,pandas as pd,spacy
from nltk.corpus import wordnet as wn
from lemminflect import getInflection
DISCLAIMER="Observable style-pattern index only; not proof/probability of AI authorship and no detector-bypass guarantee."
CODES={"NO_PATTERN_COVERAGE","PARSE_FAILURE","CANDIDATE_GENERATION_FAILURE","REALIZATION_FAILURE","SEMANTIC_REJECTION","TERMINOLOGY_REJECTION","POLARITY_REJECTION","MODALITY_REJECTION","QUANTITY_REJECTION","ENUMERATION_REJECTION","GRAMMAR_REJECTION","RANKING_EMPTY","INSUFFICIENT_REWRITE","SUCCESS","NOT_TARGETED"}
@dataclass
class Config:
    stanza:bool=False;sts:bool=True;nli:bool=True;mlm:bool=False;lt:bool=False;online:bool=False;batch:int=20;near_copy:float=.965;min_sts:float=.72;min_nli:float=.58;out:str="/tmp/nll_outputs";cache:str="/tmp/nll_cache"
@dataclass
class Profile:name:str;keywords:set[str];phrases:set[str];immutable:set[str]=field(default_factory=set);concepts:set[str]=field(default_factory=set)
@dataclass
class Prop:predicate:str;subject:str="";objects:list=field(default_factory=list);complements:list=field(default_factory=list);polarity:str="positive";modals:list=field(default_factory=list);voice:str="active"
@dataclass
class Analysis:text:str;families:list;props:list;relations:list;enumerations:list;protected:list;notes:list=field(default_factory=list);constituency:str|None=None
@dataclass
class Finding:category:str;passage:str;start:int;end:int;sentence:int;severity:float;explanation:str
@dataclass
class Candidate:text:str;lane:str;family:str;reason:str;priority:float=.5;checks:dict=field(default_factory=dict);score:float=-999
@dataclass
class SentenceResult:original:str;revised:str;status:str;families:list;family:str|None;reason:str;alternatives:list;validation:dict;diagnostics:dict
@dataclass
class DocumentResult:original:str;revised:str;workflow:str;mode:str;discipline:str;routing:list;pattern_score:float;band:str;findings:list;sentences:list;warnings:list;runtime:dict
CFG=Config();MODELS={};STATE={};Path(CFG.out).mkdir(parents=True,exist_ok=True);Path(CFG.cache).mkdir(parents=True,exist_ok=True)
def S(x):return {z.strip() for z in x.split("|") if z.strip()}
PROFILES={
"General Academic":Profile("General Academic",S("study|research|analysis|result|method|evidence|sample|finding"),S("this study|the findings")),
"Computing / Cybersecurity":Profile("Computing / Cybersecurity",S("algorithm|classifier|malware|adversarial|dataset|network|feature|robustness|encryption|cloud"),S("machine learning|feature space|problem space|threat model|static analysis"),S("XGBoost|Linear SVM|DNN|Windows PE|Android|APK|API"),S("adversarial robustness|perturbation constraint")),
"Engineering":Profile("Engineering",S("design|system|load|stress|material|control|circuit|failure|efficiency"),S("finite element|control system"),S("CAD|FEA|HVAC")),
"Medicine / Health":Profile("Medicine / Health",S("patient|clinical|treatment|diagnosis|disease|therapy|trial|cohort|outcome"),S("randomized controlled trial|adverse event"),S("DNA|RNA|MRI|CT|ICD")),
"Biology":Profile("Biology",S("cell|gene|protein|organism|species|genome|enzyme|tissue|mutation"),S("gene expression|cell signaling"),S("DNA|RNA|PCR")),
"Chemistry":Profile("Chemistry",S("compound|reaction|molecule|catalyst|solvent|synthesis|concentration|ion"),S("reaction rate|chemical equilibrium"),S("pH|NMR|HPLC")),
"Physics":Profile("Physics",S("energy|force|particle|field|wave|quantum|velocity|mass|gravity"),S("boundary condition|quantum state")),
"Mathematics / Statistics":Profile("Mathematics / Statistics",S("theorem|proof|equation|distribution|variance|regression|probability|estimator|matrix"),S("null hypothesis|confidence interval"),S("ANOVA|OLS|MSE")),
"Social Sciences":Profile("Social Sciences",S("society|community|policy|culture|institution|inequality|survey|interview|qualitative"),S("social context|thematic analysis")),
"Psychology":Profile("Psychology",S("behavior|cognition|emotion|participant|memory|attention|motivation|construct"),S("cognitive load|self efficacy"),S("DSM")),
"Education":Profile("Education",S("student|teacher|learning|curriculum|instruction|classroom|assessment|school"),S("learning outcome|instructional strategy")),
"Business / Economics":Profile("Business / Economics",S("market|firm|revenue|cost|consumer|economy|finance|investment|management"),S("return on investment|market share"),S("ROI|GDP")),
"Law / Policy":Profile("Law / Policy",S("law|legal|court|statute|regulation|policy|rights|liability|compliance"),S("case law|due process"),S("FAR|CFR")),
"Humanities":Profile("Humanities",S("text|narrative|history|philosophy|literature|discourse|interpretation|rhetoric"),S("historical context|critical theory")),
"Environmental Science":Profile("Environmental Science",S("climate|ecosystem|pollution|emission|biodiversity|habitat|sustainability|carbon"),S("climate change|greenhouse gas"),S("CO2|LCA"))}
def nlp():
    if "spacy" not in MODELS:
        try:MODELS["spacy"]=spacy.load("en_core_web_sm")
        except Exception as exc:raise RuntimeError("spaCy en_core_web_sm dependency parser is required") from exc
    return MODELS["spacy"]
def optional(kind):
    if kind=="sts" and CFG.sts:
        try:
            if kind not in MODELS:
                from sentence_transformers import SentenceTransformer
                MODELS[kind]=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:STATE["warnings"].append(f"STS unavailable: {e}");MODELS[kind]=None
    if kind=="nli" and CFG.nli:
        try:
            if kind not in MODELS:
                from transformers import AutoTokenizer,AutoModelForSequenceClassification
                mid="cross-encoder/nli-deberta-v3-small";m=AutoModelForSequenceClassification.from_pretrained(mid);m.eval();MODELS[kind]=(AutoTokenizer.from_pretrained(mid),m)
        except Exception as e:STATE["warnings"].append(f"NLI unavailable: {e}");MODELS[kind]=None
    if kind=="mlm" and CFG.mlm:
        try:
            if kind not in MODELS:
                import torch
                from transformers import pipeline
                MODELS[kind]=pipeline("fill-mask",model="distilroberta-base",device=0 if torch.cuda.is_available() else -1)
        except Exception as e:STATE["warnings"].append(f"MLM unavailable: {e}");MODELS[kind]=None
    if kind=="lt" and CFG.lt:
        try:
            if kind not in MODELS:
                import language_tool_python
                MODELS[kind]=language_tool_python.LanguageTool("en-US")
        except Exception as e:STATE["warnings"].append(f"LanguageTool unavailable: {e}");MODELS[kind]=None
    return MODELS.get(kind)
PATS={
    "citation":re.compile(r"\([^()\n]{0,90}(?:19|20)\d{2}[a-z]?[^()\n]{0,50}\)|\[[0-9,;\-\s]+\]|[A-Z][A-Za-z'\-]+ et al\. \((?:19|20)\d{2}[a-z]?\)"),
    "url":re.compile(r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+",re.I),
    "equation":re.compile(r"\$[^$]+\$"),
    "code":re.compile(r"`[^`]+`|```[\s\S]+?```"),
    "quantity":re.compile(r"(?<!\w)[<>≈~±]?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*(?:%|mg|kg|mL|mm|cm|km|Hz|GHz|°C|MPa|kW|days?|years?)?\b",re.I)
}
def protect(text,terms):
    out=[]
    for k,p in PATS.items():out += [(k,m.group(),m.start(),m.end()) for m in p.finditer(text)]
    for t in terms:out += [("term",m.group(),m.start(),m.end()) for m in re.finditer(r"(?<!\w)"+re.escape(t)+r"(?!\w)",text,re.I)]
    return sorted(set(out),key=lambda x:(x[2],x[3],x[0]))
def sent_split(text):return [s.text.strip() for s in nlp()(text).sents if s.text.strip()] or [text.strip()]
def read_document(path):
    if not path:return "",{"source":"textbox"}
    p=Path(path);e=p.suffix.lower()
    if e==".txt":return p.read_text(encoding="utf8",errors="replace"),{"type":"txt"}
    if e==".docx":
        from docx import Document
        d=Document(str(p));return "\n\n".join(x.text for x in d.paragraphs if x.text.strip()),{"type":"docx"}
    if e==".pdf":
        import pymupdf
        d=pymupdf.open(str(p));return "\n\n".join(x.get_text("text") for x in d),{"type":"pdf","pages":len(d)}
    raise ValueError("Use TXT, DOCX, or text-based PDF.")


# ===== SOURCE NOTEBOOK CELL 4 =====
#@title 4. Grammar/proposition intermediate representation and transparent scanner
MODALS={"may","might","can","could","must","should","would","will","shall"}
def sub(t):
    x=list(t.subtree);return t.doc[x[0].i:x[-1].i+1].text
class Analyzer:
    def analyze(self,s,p,u):
        d=nlp()(s);terms=p.immutable|p.concepts|u
        if not d.has_annotation("DEP"):return Analysis(s,["PARSE_FAILURE"],[],[],[],protect(s,terms),["dependency parse unavailable"])
        fam=set();props=[];rels=[];enums=[];preds=[]
        for r in [t for t in d if t.dep_=="ROOT"]:preds += [r]+[x for x in r.conjuncts if x.pos_ in {"VERB","AUX"}]
        for v in preds:
            ss=[x for x in v.children if x.dep_ in {"nsubj","nsubjpass","csubj","csubjpass"}]
            if not ss and v.dep_=="conj":ss=[x for x in v.head.children if x.dep_ in {"nsubj","nsubjpass","csubj"}]
            oo=[x for x in v.children if x.dep_ in {"obj","dobj","attr","oprd","dative"}];cc=[x for x in v.children if x.dep_ in {"ccomp","xcomp","advcl","acl","relcl"}];aux=[x.lemma_.lower() for x in v.children if x.dep_ in {"aux","auxpass"}]
            props.append(Prop(v.lemma_.lower(),sub(ss[0]) if ss else "",[sub(x) for x in oo],[sub(x) for x in cc],"negative" if any(x.dep_=="neg" for x in v.children) else "positive",[x for x in aux if x in MODALS],"passive" if any(x.dep_ in {"nsubjpass","auxpass"} for x in v.children) else "active"))
        lo=s.lower();tests={"COORDINATED_PREDICATES":any(t.dep_=="conj" and t.pos_=="VERB" for t in d),"COORDINATED_OBJECTS_OR_ENUMERATION":any(t.dep_=="conj" and t.pos_ in {"NOUN","PROPN"} for t in d),"NEGATED_PURPOSE_CONTRAST":bool(re.search(r"(?:purpose|aim|objective|goal).*is\s+not\s+to.+but\s+to",lo)),"EVIDENTIAL_COMPLEMENT":bool(re.search(r"(?:provide|provides|provided|offer|offers)\s+(?:clear\s+)?evidence\s+that",lo)),"WHETHER_COMPLEMENT":"whether" in lo,"WHEREBY_CLAUSE":"whereby" in lo,"PURPOSE_RESULT_CLAUSE":"so that" in lo,"WHILE_PARTICIPIAL":bool(re.search(r"while\s+\w+ing",lo)),"PURPOSE_INFINITIVE":bool(re.search(r"(?:intended|designed|used)\s+to",lo)),"REDUCED_RELATIVE_OR_NP_MODIFIER":bool(re.search(r"(?:obtained|derived|collected|trained|based|intended)\s+(?:from|on|to)",lo)),"FRONTED_ADJUNCT":bool(re.match(r"^(?:for|in|within|under|during|among|across)[^,]{1,80},",lo)),"WHEN_CLAUSE":bool(re.search(r"when",lo)),"PARENTHETICAL_EMPHASIS":bool(re.search(r"by (?:itself|themselves)|alone",lo)),"NOMINALIZATION":any(t.pos_=="NOUN" and t.suffix_.lower() in {"tion","sion","ment","ance","ence","ity"} for t in d),"METHOD_RELATION":bool(re.search(r"through.+(?:ing|tion|sion|ment)",lo)),"PASSIVE":any(x.voice=="passive" for x in props)}
        if d and d[0].tag_=="VBG":fam|={"GERUND_SUBJECT","PARTICIPIAL_FRONTING"}
        fam|={k for k,v in tests.items() if v};markers={"cause":{"because","since","therefore"},"purpose":{"so that","intended to"},"contrast":{"but","although","whereas","while"},"condition":{"if","unless","whether"},"temporal":{"when","while","before","after"}}
        for typ,ms in markers.items():
            for x in ms:
                if re.search(r"(?<!\w)"+re.escape(x)+r"(?!\w)",lo):rels.append({"type":typ,"marker":x})
        seen=set()
        for t in d:
            xs=[t]+list(t.conjuncts)
            if len(xs)>=3 and t.i not in seen:enums.append([sub(x) for x in xs]);seen|={x.i for x in xs}
        const=None
        if CFG.stanza:
            try:
                if "stanza" not in MODELS:
                    import stanza
                    MODELS["stanza"]=stanza.Pipeline("en",processors="tokenize,pos,lemma,depparse,constituency",use_gpu=False,verbose=False)
                const=str(MODELS["stanza"](s).sentences[0].constituency)
            except Exception as e:STATE["warnings"].append(f"Stanza unavailable: {e}")
        return Analysis(s,sorted(fam),props,rels,enums,protect(s,terms),[],const)
FORMULAIC=[r"it is important to note that",r"in today's rapidly evolving",r"delve(?:s|d)? into",r"multifaceted",r"pivotal role",r"seamlessly",r"in conclusion",r"furthermore",r"moreover",r"this highlights"]
class Scanner:
    def scan(self,text):
        ss=sent_split(text);fs=[];off=0;lens=[];starts=[]
        for i,s in enumerate(ss):
            st=max(off,text.find(s,off));off=st+len(s);d=nlp()(s);w=[t for t in d if t.is_alpha];lens.append(len(w));starts.append(" ".join(t.lemma_.lower() for t in w[:3]))
            for p in FORMULAIC:
                for z in re.finditer(p,s,re.I):fs.append(Finding("formulaic wording",z.group(),st+z.start(),st+z.end(),i,.65,"Formulaic academic wording; context may still justify it."))
            if len(w)>38:fs.append(Finding("sentence complexity",s,st,st+len(s),i,.6,"Long multi-clause sentence may be dense."))
        rep={x for x,n in Counter(starts).items() if x and n>=3}
        for i,(s,o) in enumerate(zip(ss,starts)):
            if o in rep:fs.append(Finding("repetitive opening",s[:80],text.find(s),text.find(s)+min(80,len(s)),i,.55,"Repeated sentence opening."))
        mean=statistics.mean(lens) if lens else 0;cv=statistics.pstdev(lens)/mean if len(lens)>1 and mean else 0;score=min(100,len(fs)*7+max(0,.28-cv)*40);return fs,float(score),"Low" if score<30 else "Moderate" if score<60 else "High",{"sentence_length_cv":cv,"findings":len(fs)}

# ===== SOURCE NOTEBOOK CELL 6 =====
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


# ===== SOURCE NOTEBOOK CELL 7 =====
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


# ===== SOURCE NOTEBOOK CELL 8 =====
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


# ===== SOURCE NOTEBOOK CELL 9 =====
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


# ===== SOURCE NOTEBOOK CELL 11 =====
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


# ===== SOURCE NOTEBOOK CELL 12 =====
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


# ===== SOURCE NOTEBOOK CELL 13 =====
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


# ===== SOURCE NOTEBOOK CELL 17 =====
#@title 9. PROFESSIONAL GRADIO UI v5.1 — FULL 30-RULE DIAGNOSTICS
# Complete replacement for Cell 9.
# Compatible with Cell 5 v4.1, Cell 5A v3.4, Cell 6 v5.1,
# Cell 7 v4, Cell 7A v2, Cell 7B v2, Cell 8 v4, and Cell 8A v2.

import json
import html
import re
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from dataclasses import asdict, is_dataclass

import pandas as pd
import gradio as gr


# =====================================================================
# 1. PREFLIGHT
# =====================================================================

EXPECTED_RULE_COUNT = 30

_REQUIRED_CELL9 = [
    "ENGINE", "Engine", "Analyzer", "Generator", "Validator", "Ranker",
    "RULES", "PROFILES", "realization_issues",
]

_missing_cell9 = [name for name in _REQUIRED_CELL9 if name not in globals()]
if _missing_cell9:
    raise RuntimeError(
        "Cell 9 cannot run. Missing: " + ", ".join(_missing_cell9)
        + ". Run Cells 1–8A in order first."
    )

if len(RULES) != EXPECTED_RULE_COUNT:
    raise RuntimeError(
        f"Cell 9 expected {EXPECTED_RULE_COUNT} registered transformation "
        f"rules, but found {len(RULES)}. Rerun Cells 5, 5A, 6, 7, 7A, "
        "and 7B in order."
    )

MUST_PARAPHRASE_ACTIVE = (
    getattr(Engine.one, "__name__", "") == "_one_with_must_paraphrase_v2"
)

if not MUST_PARAPHRASE_ACTIVE:
    raise RuntimeError(
        "Cell 9 requires the Cell 7B v2 must-paraphrase orchestrator. "
        "Run Cell 7B v2 before Cell 9."
    )


# =====================================================================
# 2. DETERMINISTIC SETTINGS
# =====================================================================

def _set_cfg_flag(name, value):
    if "CFG" in globals():
        try:
            setattr(CFG, name, bool(value))
        except Exception:
            pass


def _apply_ui_settings(use_stanza, use_language_tool):
    # No generative/model-assisted candidate production in this UI.
    _set_cfg_flag("mlm", False)
    _set_cfg_flag("sts", False)
    _set_cfg_flag("nli", False)
    _set_cfg_flag("online", False)

    # Optional read-only analysis resources.
    _set_cfg_flag("stanza", use_stanza)
    _set_cfg_flag("lt", use_language_tool)


# =====================================================================
# 3. GENERAL HELPERS
# =====================================================================

def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _file_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    return getattr(uploaded_file, "name", None) or str(uploaded_file)


def _read_input_text(typed_text, uploaded_file):
    typed_text = str(typed_text or "")
    if typed_text.strip():
        return typed_text.strip(), "typed_text"

    path_value = _file_path(uploaded_file)
    if not path_value:
        raise gr.Error("Enter text or upload a TXT, DOCX, or text-based PDF.")

    path = Path(path_value)
    if not path.exists():
        raise gr.Error("The uploaded file could not be found.")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace").strip(), path.name

    if suffix == ".docx":
        from docx import Document
        document = Document(str(path))
        text = "\n\n".join(p.text for p in document.paragraphs).strip()
        return text, path.name

    if suffix == ".pdf":
        try:
            import fitz
        except Exception as exc:
            raise gr.Error("PyMuPDF is unavailable: " + str(exc))
        pdf = fitz.open(str(path))
        try:
            text = "\n\n".join(page.get_text("text") for page in pdf).strip()
        finally:
            pdf.close()
        if not text:
            raise gr.Error(
                "The PDF contains no extractable text. Scanned PDFs require OCR."
            )
        return text, path.name

    raise gr.Error("Unsupported file type. Use TXT, DOCX, or text-based PDF.")


def _trace(item):
    value = getattr(item, "diagnostics", None)
    if value is None:
        value = getattr(item, "trace", None)
    return dict(value or {})


def _checks(item):
    return dict(getattr(item, "checks", None) or {})


def _changed(item):
    return str(getattr(item, "original", "")) != str(getattr(item, "revised", ""))


# =====================================================================
# 4. DIAGNOSTIC TABLES
# =====================================================================

def _sentence_rows(result):
    rows = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        selected = trace.get("selected") or {}
        generated = trace.get("generated_by_lane", {}) or {}
        must = trace.get("must_paraphrase", {}) or {}
        attempts = must.get("attempts", []) or []
        rows.append({
            "Sentence": number,
            "Status": getattr(item, "status", ""),
            "Changed": _changed(item),
            "Requested mode": must.get("requested_mode", getattr(result, "mode", "")),
            "Accepted mode": must.get("accepted_mode", ""),
            "Attempts": must.get("attempt_count", len(attempts)),
            "Must-paraphrase outcome": must.get("outcome", ""),
            "Detected families": " | ".join(list(getattr(item, "families", []) or [])),
            "Selected family": getattr(item, "family", None) or selected.get("family") or "",
            "Selected lane": selected.get("lane", ""),
            "Selected score": selected.get("score"),
            "Minimal candidates": generated.get("Minimal", 0),
            "Balanced candidates": generated.get("Balanced", 0),
            "Structural candidates": generated.get("Structural", 0),
            "Before final filter": trace.get("generated_before_final_filter"),
            "After final filter": trace.get("generated_after_final_filter"),
            "Original": getattr(item, "original", ""),
            "Revised": getattr(item, "revised", ""),
            "Reason": getattr(item, "reason", ""),
        })
    return rows


def _routing_rows(result):
    rows = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        applicable = trace.get("applicable", {}) or {}
        generated = trace.get("generated_by_lane", {}) or {}
        rows.append({
            "Sentence": number,
            "Status": getattr(item, "status", ""),
            "Detected families": " | ".join(
                trace.get("detected_families", [])
                or list(getattr(item, "families", []) or [])
            ),
            "Minimal rules": " | ".join(applicable.get("Minimal", []) or []),
            "Balanced rules": " | ".join(applicable.get("Balanced", []) or []),
            "Structural rules": " | ".join(applicable.get("Structural", []) or []),
            "Minimal generated": generated.get("Minimal", 0),
            "Balanced generated": generated.get("Balanced", 0),
            "Structural generated": generated.get("Structural", 0),
            "Before final filter": trace.get("generated_before_final_filter"),
            "After final filter": trace.get("generated_after_final_filter"),
            "Rule errors": json.dumps(_json_safe(trace.get("rule_errors", [])), ensure_ascii=False),
            "Terminal rejections": json.dumps(_json_safe(trace.get("rejections", {})), ensure_ascii=False),
        })
    return rows


def _realization_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        for rejection_number, rejection in enumerate(
            list(_trace(item).get("realization_rejections", []) or []), 1
        ):
            rows.append({
                "Sentence": sentence_number,
                "Rejection": rejection_number,
                "Stage": rejection.get("stage", ""),
                "Lane": rejection.get("lane", ""),
                "Family/rule": rejection.get("rule", ""),
                "Issues": " | ".join(rejection.get("issues", []) or []),
                "Rejected text": rejection.get("text", ""),
            })
    if not rows:
        rows.append({
            "Sentence": 0, "Rejection": 0, "Stage": "", "Lane": "",
            "Family/rule": "", "Issues": "None", "Rejected text": "",
        })
    return rows


def _candidate_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        candidates = list(_trace(item).get("candidate_traces", []) or [])
        if not candidates:
            rows.append({
                "Sentence": sentence_number,
                "Candidate": 0,
                "Lane": "",
                "Family": "",
                "Accepted": False,
                "Score": None,
                "Change": None,
                "Failures": "",
                "Rejection reasons": "No candidate trace available",
                "Candidate text": "",
            })
            continue
        for candidate_number, candidate in enumerate(candidates, 1):
            rows.append({
                "Sentence": sentence_number,
                "Candidate": candidate_number,
                "Lane": candidate.get("lane", ""),
                "Family": candidate.get("family", ""),
                "Accepted": candidate.get("accepted", False),
                "Score": candidate.get("score"),
                "Change": candidate.get("change"),
                "Failures": " | ".join(candidate.get("failures", []) or []),
                "Rejection reasons": json.dumps(
                    _json_safe(candidate.get("rejection_reasons", [])),
                    ensure_ascii=False,
                ),
                "Candidate text": candidate.get("text", ""),
            })
    return rows


def _attempt_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        must = _trace(item).get("must_paraphrase", {}) or {}
        attempts = must.get("attempts", []) or []
        if not attempts:
            rows.append({
                "Sentence": sentence_number, "Attempt": 0, "Mode": "",
                "Status": "", "Changed": False, "Selected family": "",
                "Failures": "", "Realization issues": "",
                "Generated after filter": None, "Revised": "",
            })
            continue
        for attempt_number, attempt in enumerate(attempts, 1):
            rows.append({
                "Sentence": sentence_number,
                "Attempt": attempt_number,
                "Mode": attempt.get("attempted_mode", ""),
                "Status": attempt.get("status", ""),
                "Changed": attempt.get("changed", False),
                "Selected family": attempt.get("selected_family", ""),
                "Failures": " | ".join(attempt.get("failures", []) or []),
                "Realization issues": " | ".join(attempt.get("realization_issues", []) or []),
                "Generated after filter": attempt.get("generated_after_final_filter"),
                "Revised": attempt.get("revised", ""),
            })
    return rows


def _finding_rows(result):
    rows = []
    for finding in list(getattr(result, "findings", []) or []):
        safe = _json_safe(finding)
        rows.append(safe if isinstance(safe, dict) else {"Finding": safe})
    return rows


# =====================================================================
# 5. SUMMARY AND FULL PAYLOAD
# =====================================================================

def _summary_markdown(result):
    sentences = list(getattr(result, "sentences", []) or [])
    total = len(sentences)
    successes = sum(getattr(item, "status", "") == "SUCCESS" for item in sentences)
    changed = sum(_changed(item) for item in sentences)
    statuses = Counter(getattr(item, "status", "UNKNOWN") for item in sentences)
    attempts = sum(
        int((_trace(item).get("must_paraphrase", {}) or {}).get("attempt_count", 0) or 0)
        for item in sentences
    )
    preserved = sum(
        (_trace(item).get("must_paraphrase", {}) or {}).get("outcome")
        == "ORIGINAL_PRESERVED_AFTER_EXHAUSTIVE_SAFE_SEARCH"
        for item in sentences
    )
    status_text = ", ".join(f"{name}: {count}" for name, count in sorted(statuses.items()))
    warnings = list(getattr(result, "warnings", []) or [])
    warning_text = "<br>".join(html.escape(str(x)) for x in warnings) if warnings else "None"

    return f"""
### Processing summary

- **Discipline:** {getattr(result, 'discipline', '')}
- **Routing evidence:** {', '.join(map(str, getattr(result, 'evidence', []) or [])) or 'default profile'}
- **Writing Pattern Score:** {getattr(result, 'pattern_score', '')} — **{getattr(result, 'band', '')}**
- **Registered transformation rules:** {len(RULES)}
- **Must-paraphrase orchestrator:** {'Active' if MUST_PARAPHRASE_ACTIVE else 'Inactive'}
- **Sentences processed:** {total}
- **Successful selections:** {successes}
- **Changed sentences:** {changed}
- **Unchanged sentences:** {total - changed}
- **Total bounded attempts:** {attempts}
- **Preserved after exhaustive safe search:** {preserved}
- **Terminal statuses:** {status_text or 'None'}
- **Warnings:** {warning_text}

> The Writing Pattern Score describes observable structural and stylistic characteristics only. It does not identify authorship.
"""


def _full_payload(result, source_name):
    sentences = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        sentences.append({
            "sentence": number,
            "status": getattr(item, "status", ""),
            "original": getattr(item, "original", ""),
            "revised": getattr(item, "revised", ""),
            "families": _json_safe(getattr(item, "families", [])),
            "selected_family": getattr(item, "family", None),
            "reason": getattr(item, "reason", ""),
            "checks": _json_safe(_checks(item)),
            "trace": _json_safe(_trace(item)),
        })

    return {
        "source_name": source_name,
        "workflow": getattr(result, "workflow", ""),
        "mode": getattr(result, "mode", ""),
        "discipline": getattr(result, "discipline", ""),
        "routing_evidence": _json_safe(getattr(result, "evidence", [])),
        "writing_pattern_score": getattr(result, "pattern_score", None),
        "writing_pattern_band": getattr(result, "band", ""),
        "warnings": _json_safe(getattr(result, "warnings", [])),
        "stats": _json_safe(getattr(result, "stats", {})),
        "registered_rule_count": len(RULES),
        "must_paraphrase_active": MUST_PARAPHRASE_ACTIVE,
        "must_paraphrase_wrapper": getattr(Engine.one, "__name__", ""),
        "rewrite_configuration": {
            "deterministic_rules": True,
            "masked_language_model": False,
            "sts_model": False,
            "nli_model": False,
            "online_lexicon": False,
            "language_tool_optional": bool(getattr(globals().get("CFG", object()), "lt", False)),
        },
        "registered_rules": [
            {
                "name": registered.name,
                "lanes": sorted(registered.lanes),
                "families": sorted(registered.families),
                "priority": registered.priority,
            }
            for registered in RULES
        ],
        "sentences": sentences,
    }


# =====================================================================
# 6. REPORTS
# =====================================================================

def _write_reports(result, source_name, payload):
    folder = Path(tempfile.mkdtemp(prefix="nll_review_"))
    stem = Path(source_name or "nll_input").stem
    token = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    txt_path = folder / f"{stem}_{token}_revised.txt"
    json_path = folder / f"{stem}_{token}_diagnostics.json"
    docx_path = folder / f"{stem}_{token}_review.docx"
    html_path = folder / f"{stem}_{token}_review.html"

    revised = str(getattr(result, "revised", ""))
    txt_path.write_text(revised, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    from docx import Document
    document = Document()
    document.add_heading("NLL Universal Professional Editor — Review", level=0)
    document.add_paragraph("Discipline: " + str(getattr(result, "discipline", "")))
    document.add_paragraph(
        "Writing Pattern Score: " + str(getattr(result, "pattern_score", ""))
        + " — " + str(getattr(result, "band", ""))
    )
    document.add_paragraph(
        "The Writing Pattern Score describes observable structural and stylistic "
        "characteristics only. It does not identify authorship."
    )
    document.add_heading("Revised text", level=1)
    for paragraph in re.split(r"\n\s*\n", revised):
        document.add_paragraph(paragraph)

    document.add_heading("Sentence review", level=1)
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        document.add_heading(f"Sentence {number}: {getattr(item, 'status', '')}", level=2)
        document.add_paragraph("Original: " + str(getattr(item, "original", "")))
        document.add_paragraph("Revised: " + str(getattr(item, "revised", "")))
        document.add_paragraph("Reason: " + str(getattr(item, "reason", "")))
        document.add_paragraph(
            "Must-paraphrase: "
            + json.dumps(_json_safe(trace.get("must_paraphrase", {})), ensure_ascii=False)
        )
    document.save(str(docx_path))

    rows = _sentence_rows(result)
    body = "".join(
        "<tr>"
        f"<td>{row['Sentence']}</td>"
        f"<td>{html.escape(str(row['Status']))}</td>"
        f"<td>{html.escape(str(row['Accepted mode']))}</td>"
        f"<td>{html.escape(str(row['Detected families']))}</td>"
        f"<td>{html.escape(str(row['Original']))}</td>"
        f"<td>{html.escape(str(row['Revised']))}</td>"
        f"<td>{html.escape(str(row['Reason']))}</td>"
        "</tr>"
        for row in rows
    )
    html_path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>NLL Review</title><style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:30px auto;color:#172033}}
h1,h2{{color:#123b66}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top;text-align:left}}
th{{background:#eaf2fb}}pre{{white-space:pre-wrap;background:#f8fafc;padding:12px}}
</style></head><body>
<h1>NLL Universal Professional Editor — Review</h1>
<p><strong>Rules:</strong> {len(RULES)} | <strong>Must paraphrase:</strong> {MUST_PARAPHRASE_ACTIVE}</p>
<h2>Revised text</h2><pre>{html.escape(revised)}</pre>
<h2>Sentence review</h2><table><thead><tr><th>#</th><th>Status</th><th>Mode</th>
<th>Families</th><th>Original</th><th>Revised</th><th>Reason</th></tr></thead>
<tbody>{body}</tbody></table>
<h2>Full diagnostics</h2><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
</body></html>""",
        encoding="utf-8",
    )

    return [str(txt_path), str(docx_path), str(json_path), str(html_path)]


# =====================================================================
# 7. MAIN CALLBACK
# =====================================================================

def run_ui_with_full_diagnostics(
    typed_text,
    uploaded_file,
    workflow,
    mode,
    discipline,
    protected_terms,
    use_stanza,
    use_language_tool,
    progress=gr.Progress(),
):
    source_text, source_name = _read_input_text(typed_text, uploaded_file)
    if not source_text.strip():
        raise gr.Error("No readable text was provided.")

    _apply_ui_settings(use_stanza, use_language_tool)
    progress(0.02, desc="Preparing request")

    try:
        result = ENGINE.process(
            source_text, workflow, mode, discipline, protected_terms, progress
        )
    except TypeError:
        result = ENGINE.process(
            text=source_text,
            workflow=workflow,
            mode=mode,
            discipline=discipline,
            terms_text=protected_terms,
            progress=progress,
        )

    progress(0.86, desc="Building diagnostics")
    payload = _full_payload(result, source_name)
    files = _write_reports(result, source_name, payload)
    progress(1.0, desc="Complete")

    return (
        getattr(result, "revised", source_text),
        _summary_markdown(result),
        pd.DataFrame(_sentence_rows(result)),
        pd.DataFrame(_routing_rows(result)),
        pd.DataFrame(_attempt_rows(result)),
        pd.DataFrame(_realization_rows(result)),
        pd.DataFrame(_candidate_rows(result)),
        pd.DataFrame(_finding_rows(result)),
        payload,
        files,
    )


# =====================================================================
# 8. PROFESSIONAL GRADIO UI
# =====================================================================

CELL9_CSS = """
.gradio-container { max-width: 1650px !important; }
#nll-title { text-align:center; margin-bottom:4px; color:#123b66; }
#nll-subtitle { text-align:center; color:#596779; margin-bottom:16px; }
.nll-panel { border:1px solid #d5dee8; border-radius:14px; padding:12px; background:#fbfdff; }
.nll-note { background:#fff8dc; border-left:4px solid #d7a100; padding:8px 12px; }
.nll-ok { background:#e8f6ee; border-left:4px solid #238636; padding:8px 12px; }
"""


with gr.Blocks(css=CELL9_CSS, title="NLL Universal Professional Editor") as NLL_APP:
    gr.Markdown("# NLL Universal Professional Editor", elem_id="nll-title")
    gr.Markdown(
        "Deterministic, non-generative rewriting with must-paraphrase orchestration, "
        "meaning safeguards, and complete sentence/candidate diagnostics.",
        elem_id="nll-subtitle",
    )

    gr.Markdown(
        f"**System ready:** {len(RULES)} rules | Must-paraphrase active | "
        "Cell 5A realization gate active | Cell 7 judge active",
        elem_classes="nll-ok",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="nll-panel"):
            gr.Markdown("## Input and controls")
            input_text = gr.Textbox(
                label="Text",
                lines=20,
                placeholder="Paste academic, contract, policy, technical, business, or professional text.",
            )
            input_file = gr.File(
                label="Optional file upload",
                file_types=[".txt", ".docx", ".pdf"],
                type="filepath",
            )
            workflow_control = gr.Radio(
                choices=["Scan Only", "Scan & Neutralize", "Edit / Paraphrase Only"],
                value="Edit / Paraphrase Only",
                label="Workflow",
            )
            mode_control = gr.Radio(
                choices=["Minimal", "Balanced", "Structural"],
                value="Balanced",
                label="Preferred rewrite mode",
            )
            discipline_control = gr.Dropdown(
                choices=["Auto Detect"] + list(PROFILES.keys()),
                value="Auto Detect",
                label="Discipline",
            )
            protected_terms_control = gr.Textbox(
                label="Immutable/user-protected terms",
                placeholder="Comma, semicolon, or line-separated terms, e.g., WSC, TxDOT, ALCEF-RSF",
                lines=3,
            )
            with gr.Accordion("Optional non-generative analysis resources", open=False):
                stanza_control = gr.Checkbox(
                    value=bool(getattr(globals().get("CFG", object()), "stanza", False)),
                    label="Stanza parsing support",
                )
                language_tool_control = gr.Checkbox(
                    value=bool(getattr(globals().get("CFG", object()), "lt", False)),
                    label="Local LanguageTool comparison",
                )
            analyze_button = gr.Button("Analyze / Revise", variant="primary", size="lg")
            gr.Markdown(
                "The selected mode is attempted first. If no safe change survives, "
                "the other modes are searched. The original is retained only after "
                "all safe deterministic options are exhausted.",
                elem_classes="nll-note",
            )

        with gr.Column(scale=1, elem_classes="nll-panel"):
            gr.Markdown("## Revised output and review")
            revised_output = gr.Textbox(label="Revised output", lines=20)
            summary_output = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Sentence comparison"):
                    sentence_table = gr.Dataframe(
                        label="Original, revised, mode, family, and judge outcome",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Routing diagnostics"):
                    routing_table = gr.Dataframe(
                        label="Detected families, applicable rules, and generated candidates",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Must-paraphrase attempts"):
                    attempt_table = gr.Dataframe(
                        label="Requested-mode-first judge and retry attempts",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Realization rejections"):
                    realization_table = gr.Dataframe(
                        label="Candidates rejected by Cell 5A",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Candidate diagnostics"):
                    candidate_table = gr.Dataframe(
                        label="Validated candidates, scores, and rejection reasons",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Writing-pattern findings"):
                    findings_table = gr.Dataframe(
                        label="Observable structural and stylistic findings",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Full JSON diagnostics"):
                    diagnostic_json = gr.JSON(label="Complete diagnostic payload")
                with gr.Tab("Downloads"):
                    download_output = gr.File(
                        label="Download revised text and review reports",
                        file_count="multiple",
                    )

    inputs = [
        input_text, input_file, workflow_control, mode_control,
        discipline_control, protected_terms_control,
        stanza_control, language_tool_control,
    ]
    outputs = [
        revised_output, summary_output, sentence_table, routing_table,
        attempt_table, realization_table, candidate_table, findings_table,
        diagnostic_json, download_output,
    ]

    analyze_button.click(
        fn=run_ui_with_full_diagnostics,
        inputs=inputs,
        outputs=outputs,
        show_progress="full",
        concurrency_limit=1,
    )

NLL_APP.queue(default_concurrency_limit=1, max_size=8)

print("=" * 76)
print("CELL 9 PROFESSIONAL GRADIO UI v5.1 LOADED")
print("=" * 76)
print("Aligned pipeline: 5 v4.1 -> 5A v3.4 -> 6 v5.1 -> 7 v4 -> 7A v2 -> 7B v2 -> 8 v4 -> 8A v2")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_RULE_COUNT)
print("Must-paraphrase orchestrator: active")
print("Cell 5A realization diagnostics: active")
print("Sentence comparison: active")
print("Routing diagnostics: active")
print("Must-paraphrase attempt diagnostics: active")
print("Candidate diagnostics: active")
print("Writing Pattern Score terminology: active")
print("LLM/MLM candidate generation: disabled")
print("STS/NLI scoring: disabled")
print("Online lexical fallback: disabled")
print("TXT/DOCX/JSON/HTML downloads: active")
print("=" * 76)

# Gradio is mounted by the production ASGI app; do not launch here.