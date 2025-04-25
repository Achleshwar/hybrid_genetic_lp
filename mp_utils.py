from multiprocessing import Pool
from hashlib import sha256
from ga_utils import construct_G
from lp_utils import compute_hm

def evaluate_one_candidate(args):
    P, m = args
    G = construct_G(P)

    try:
        hm = compute_hm(G, m)
    except Exception as e:
        print('Error in compute_hm:', e)
        hm = float('inf')
    
    return (P, hm)

def rerank_with_exact_lp_parallel(population, m, top_k=5, num_workers=4):
    top_candidates = population[:top_k]
    top_candidates = population[:3] + population[5:8] # for diversification
    with Pool(num_workers) as pool:
        results = pool.map(evaluate_one_candidate, [(P, m) for P in top_candidates])
    results.sort(key=lambda x: x[1])
    return results

def hash_matrix(G):
    return sha256(G.tobytes()).hexdigest()

def compute_hm_cached(G, m, 
                      hm_cache:dict):
    key = (hash_matrix(G), m)
    if key in hm_cache:
        return hm_cache[key]
    val = compute_hm(G, m)
    hm_cache[key] = val
    return val