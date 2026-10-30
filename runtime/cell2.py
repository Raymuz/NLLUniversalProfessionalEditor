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
