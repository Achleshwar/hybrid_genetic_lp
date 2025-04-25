import json
import os
import numpy as np

def load_candidates(json_path, target_code):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # sort candidates on the basis of their m-heights
    data.sort(key=lambda x: x["lp_m_height"])
    candidates = []
    for entry in data:
        if entry["code"] == target_code:
            P = np.array(entry["matrix"])
            candidates.append(P)

    return candidates

def log_outputs(n, k, m,
                top_candidates,
                top_m_heights,
                top_sampled_m_heights,
                generations):

    output_list = []  # list of dicts, one per candidate

    for i in range(len(top_candidates)):
        candidate_dict = {
            "rank": i + 1,
            "n": n,
            "k": k,
            "m": m,
            "top_candidates": top_candidates[i].tolist(),
            "top_m_heights": float(top_m_heights[i]),
            "top_sampled_m_heights": float(top_sampled_m_heights[i]),
            "generations": generations
        }
        output_list.append(candidate_dict)

    # Save the list to a JSON file
    output_dir = "./outputs/"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"output_{n}_{k}_{m}.json")
    with open(output_path, 'w') as f:
        json.dump(output_list, f, indent=4)
