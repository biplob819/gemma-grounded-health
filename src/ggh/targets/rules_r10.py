"""TGT-R10 as the first v2 build used it (commit e8b7b48, OWNER D-43), frozen.

The rule was tightened after the first per-sentence audit (OWNER D-46); the tightened rule is
TGT-R12 in `rules.py`. This module keeps the first version byte for byte so that examples
stamped TGT-R10 still rebuild exactly (PRV-R1). Never edit it: a change here would change
those targets. Its hash is part of `make.rules_hash`.
"""

from __future__ import annotations

import re

_LEAD = (  # sentence start, with an optional opener before the aim phrasing
    r"^\W*(?:\d+\)\s*)?"  # "(1) To evaluate..."
    r"(?:(?:therefore|thus|hence|accordingly|consequently|here(?:in)?|now|so),?\s+)?"
    r"(?:(?:in|for)\s+(?:this|the\s+(?:present|current)|our)\s+(?:[\w-]+,?\s+){0,3}?"
    r"(?:study|analysis|report|paper|work|trial|investigation|review|survey|article),?\s+)?"
    r"(?:here(?:\s*in)?,?\s+)?"
)
_CLAUSE = (  # sentence start, or a clause start ("Since X is unclear, we investigated...")
    r"(?:" + _LEAD + r"|[,;:]\s+(?:(?:therefore|thus|hence|here(?:in)?|so),?\s+)?)"
)
_STUDY = (  # "this study", "the present prospective cohort study", "this multicenter RCT"
    r"(?:this|the\s+(?:present|current)|our)\s+(?:[\w-]+,?\s+){0,3}?"
    r"(?:study|analysis|report|paper|work|trial|rct|investigation|review|project|survey|"
    r"experiment|research)"
)
_ADV = (  # "we have now examined", "we therefore aimed"
    r"(?:(?:have|therefore|thus|also|further|then|next|now|here|first|specifically|"
    r"retrospectively|prospectively)\s+){0,2}"
)
_WE = r"(?:we|the\s+authors)\s+"
AIM_PATTERNS = (  # aim phrasing: an aim unless the sentence also says what was found
    _LEAD + r"(?:in\s+order\s+)?to\s+(?:investigate|determine|evaluate|assess|examine|"
    r"explore|compare|study|test|identify|analy[sz]e|describe|estimate|establish|"
    r"characteri[sz]e|clarify|elucidate|measure|quantify|ascertain|verify|validate|define|"
    r"understand|address|confirm|document|report|review|present)\b",
    _CLAUSE + r"(?:(?:the|our|a|one|another)\s+)?(?:(?:primary|main|secondary|first|second|"
    r"overall|specific|principal|major|study|research|present|current|ultimate|key)\s+){0,2}"
    r"(?:aims?|purposes?|objectives?|goals?|intent|intention)\s+(?:of|was|were|is|are)\b",
    r"\b(?:aims?|purposes?|objectives?|goals?)\s+of\s+" + _STUDY + r"\b",
    r"\b(?:it\s+(?:is|was)|we|authors)\s+(?:\w+\s+)?hypothesi[sz]ed\b",
    r"\btest(?:ed|s)?\s+(?:the|our|a|this|these|two)\s+(?:\w+\s+)?hypothes[ie]s\b",
    r"\b(?:our|the|a)\s+(?:\w+\s+)?hypothesis\s+(?:is|was)\s+that\b",
    r"\bquestions?\s+addressed\b",
    _CLAUSE + _WE + _ADV + r"(?:conducted|undertook|performed|did|designed|carried\s+out|"
    r"initiated)\s+(?:this|the\s+present|the\s+current|a|an)\s+(?:[\w-]+,?\s+){0,4}?"
    r"(?:study|analysis|trial|survey|review|investigation|experiment)\s+(?:in\s+order\s+)?to\b",
    _CLAUSE + _WE + _ADV + r"(?:hypothesi[sz]e|aimed|aim|sought|seek|set\s+out|wished|"
    r"wanted|intended)\b",
    _CLAUSE + _STUDY + r"\s+(?:was\s+|is\s+)?(?:aimed|aims|sought|seeks|designed|"
    r"undertaken|initiated|intends|intended|attempts|attempted|set\s+out|"
    r"focus(?:es|ed)\s+on|to\s+\w+|(?:conducted|performed|carried\s+out|done)\s+"
    r"(?:in\s+order\s+)?to)\b",
)
DESCRIPTION_PATTERNS = (  # study description: an aim only when no result is reported
    _CLAUSE + _WE + _ADV + r"(?:investigated?|examined?|evaluated?|assessed?|studied|"
    r"study|explored?|tested?)\b",
    _CLAUSE + _STUDY + r"\s+(?:investigated?|investigates|examined?|examines|evaluated?|"
    r"evaluates|assessed?|assesses|explored?|explores|describes|described|presents|"
    r"compared?|compares)\b",
    r"\b(?:was|were)\s+(?:\w+ly\s+)?(?:studied|investigated|examined|explored)\b",
    r"\b(?:study|trial|analysis|investigation)\s+(?:was\s+)?(?:(?:conducted|performed|"
    r"undertaken|designed|carried\s+out)\s+)?(?:in\s+order\s+)?to\s+(?:investigate|determine|"
    r"evaluate|assess|examine|explore|test|compare|clarify|elucidate|identify)\b",
)
_FOUND = re.compile(  # a result reported in so many words ("we found", "results showed")
    r"\b(?:we|and|but|they|authors|it|was|were|is|are|has|have|had)\s+(?:also\s+|further\s+|"
    r"then\s+|not\s+)?(?:found|showed|shown|revealed|demonstrated|observed)\b|"
    r"\b(?:results?|findings?|data|analys[ie]s)\s+(?:\w+\s+)?(?:showed|revealed|"
    r"demonstrated|indicated|suggested)\b",
    re.I,
)
_STATED = re.compile(  # a result stated as fact; read outside any "whether"/"if" clause
    r"\bp\s*[<>=\u2264\u2265]|\d\s*%|\b(?:odds|hazard|risk)\s+ratios?\b|"
    r"\b(?:is|are|was|were|be|been)\s+(?:not\s+|significantly\s+|strongly\s+|"
    r"independently\s+|positively\s+|negatively\s+|inversely\s+)*(?:associated|correlated|"
    r"related|linked|higher|lower|greater|smaller|increased|decreased|reduced|elevated|"
    r"improved|unchanged|similar|different|comparable|superior|inferior|effective|"
    r"predictive)\b|\bsignificant(?:ly)?\b",
    re.I,
)
_OPEN_CLAUSE = re.compile(r"\b(?:whether|if)\b.*", re.I | re.S)
_AIM = re.compile("|".join(f"(?:{p})" for p in AIM_PATTERNS), re.I)
_DESCRIPTION = re.compile("|".join(f"(?:{p})" for p in DESCRIPTION_PATTERNS), re.I)


def reports_result(sentence: str) -> bool:
    """A result in so many words ("we found"), or one stated as fact outside a "whether" or
    "if" clause ("X was associated with Y", "p < 0.05", "12%")."""
    return bool(_FOUND.search(sentence) or _STATED.search(_OPEN_CLAUSE.sub("", sentence)))


def is_aim_sentence(sentence: str) -> bool:
    """TGT-R10: an aim, objective, hypothesis or study-description sentence."""
    if _AIM.search(sentence):  # aim wording: only a result in so many words overrides it
        return _FOUND.search(sentence) is None
    if _DESCRIPTION.search(sentence):
        return not reports_result(sentence)
    return False
