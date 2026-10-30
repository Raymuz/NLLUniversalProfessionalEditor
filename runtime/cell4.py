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