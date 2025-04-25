# Hybrid Genetic Algorithm

## How to run
```bash
python main.py --job_path <path-to-txt-file>
```

### Example of a job text file
```text
(9,4,2)
(9,4,3)
(9,4,4)
```

## Directory Structure
- `mp_utils.py` provides multiprocessing utils
- `ga_utils.py` provides genetic algorithm utils such as `mutate`, `crossover`, etc.
- `lp_utils.py` provides LP utils such as `compute_hm` based on LP
- `utils.py` provides logging and file reading utils
- `main.py` is what we need to run

## Logging
Outputs should be saved in `./outputs/` directory as `{n}_{k}_{m}.json` file. Each file has following structure:

```json
[
  {
    "rank": 1,
    "n": 9,
    "k": 6,
    "m": 3,
    "top_candidates": [[...], [...], ...],
    "top_m_heights": 2.98,
    "top_sampled_m_heights": 2.79,
    "generations": 100
  },
  {
    "rank": 2,
    ...
  },
  ...
]
```

## Verbose Interpretation
Below is a sample run of our hybrid genetic + LP-based refinement framework for minimizing m-height of analog codes:

<pre> ```
Initial best m-height: 2.997564706099727
Initial best LP m-height: 3.0116279069767415
Gen 1: Best heurisitc m-height = 2.9929
Gen 2: Best heurisitc m-height = 2.9621
Gen 3: Best heurisitc m-height = 2.9621
Gen 4: Best heurisitc m-height = 2.9621
Gen 5: Best heurisitc m-height = 2.9621
Gen 6: Best heurisitc m-height = 2.9621
Gen 7: Best heurisitc m-height = 2.9621
Gen 8: Best heurisitc m-height = 2.9621
Gen 9: Best heurisitc m-height = 2.9621
Gen 10: Best heurisitc m-height = 2.9621
🔍 LP-refinement step
🔁 LP-Refinement Start: Best LP m-height = 3.0000
Gen 1: Best LP m-height = 3.0000 
    LP-refined h_m = 3.0000 

Best candidate P matrix:
[[86 86 86 86 86]
[86 85 87 87 85]
[87 86 87 86 87]
[87 85 87 86 86]] 
Best m-height achieved: 3.0 ``` </pre>