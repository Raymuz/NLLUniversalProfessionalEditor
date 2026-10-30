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
