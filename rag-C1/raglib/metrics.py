from typing import List, Set
import numpy as np

def precision_at_k(pred_ids: List[str], rel_ids: Set[str], k: int) -> float:
    top = pred_ids[:k];  return 0.0 if k == 0 else sum(d in rel_ids for d in top)/k

def recall_at_k(pred_ids: List[str], rel_ids: Set[str], k: int) -> float:
    top = pred_ids[:k];  return 0.0 if not rel_ids else sum(d in rel_ids for d in top)/len(rel_ids)

def ndcg_at_k(pred_ids: List[str], rel_ids: Set[str], k: int) -> float:
    def dcg(lst): return sum((1.0 if d in rel_ids else 0.0)/np.log2(i+2) for i,d in enumerate(lst[:k]))
    idcg = sum(1.0/np.log2(i+2) for i in range(min(k, len(rel_ids))))
    return 0.0 if idcg == 0 else dcg(pred_ids)/idcg

def mrr(pred_ids: List[str], rel_ids: Set[str]) -> float:
    for i,d in enumerate(pred_ids,1):
        if d in rel_ids: return 1.0/i
    return 0.0
