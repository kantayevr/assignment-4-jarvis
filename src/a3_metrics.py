"""Scoring helpers carried over from Assignment 3."""
import math, re

def normalize(tag):return tag.strip().lower().removeprefix('v')
def extract_tag(answer):
    found={normalize(x) for x in re.findall(r'(?<![\w.])v?\d+\.\d+\.\d+(?:[-+][0-9a-zA-Z.-]+)?(?!\w|\.\d)',answer)}
    return next(iter(found)) if len(found)==1 else None

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]
def mcnemar_exact(b,c):
    n=b+c
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/(2**n))
def score(row,gold):
    tag=extract_tag(row['answer']);correct=tag==normalize(gold['answer_tag'])
    matching=[x for x in row['retrieved'] if x['repository']==row['repository']]
    supported=tag is not None and any(normalize(x['tag'])==tag for x in matching)
    gold_available=any(normalize(x['tag'])==normalize(gold['answer_tag']) for x in matching)
    abstain=tag is None and row['answer'].strip().upper()=='UNKNOWN'
    if correct:cat='correct'
    elif abstain:cat='abstention'
    elif tag is None:cat='unparseable_or_ambiguous'
    elif gold['changed'] and tag==normalize(gold['old_tag']):cat='outdated'
    elif not supported:cat='unsupported_tag'
    else:cat='other_incorrect'
    return {'id':row['id'],'repository':row['repository'],'condition':row['condition'],
        'answer':row['answer'],'extracted_tag':tag or '', 'gold_tag':gold['answer_tag'],'old_tag':gold['old_tag'],
        'changed':gold['changed'],'correct':correct,'supported_by_target_record':supported,
        'unsupported_tag':tag is not None and not supported,'gold_in_retrieved_context':gold_available,
        'target_record_retrieved':bool(matching),'error_category':cat,
        'input_tokens':row['input_tokens'],'output_tokens':row['output_tokens'],'elapsed_seconds':row['elapsed_seconds']}
