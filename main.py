import numpy as np
import argparse
from tqdm import tqdm
import time
from scipy.optimize import differential_evolution

from utils import load_candidates, log_outputs
from ga_utils import generate_random_P, construct_G, tournament_selection, crossover, mutate
from lp_utils import solve_lp, compute_hm, lp_guided_refinement
from mp_utils import compute_hm_cached, rerank_with_exact_lp_parallel

# --- m-height evaluation using differential evolution ---
def new_m_height(G, m, de_maxiter=50, de_repeats=5):
    """
    Compute a heuristic for the m-height of the code given generator matrix G.
    Uses differential evolution over weight vectors to minimize the ratio:
      ratio = sorted_abs(xG)[0] / sorted_abs(xG)[m].
    Lower is better.
    """
    k, n = G.shape

    def objective(x, G, m):
        x = np.array(x)
        xG = np.dot(x, G)
        sorted_abs = np.sort(np.abs(xG))[::-1]
        if m >= len(sorted_abs) or sorted_abs[m] == 0:
            return 1e6  # Penalize invalid cases
        ratio = sorted_abs[0] / sorted_abs[m]
        return 1.0/ratio

    bounds = [(-1, 1)] * k
    best_ratio = 0.0
    for _ in range(de_repeats):
        result = differential_evolution(objective, bounds, args=(G, m),
                                        maxiter=de_maxiter, disp=False)
        ratio = 1.0 / result.fun if result.fun != 0 else 1e6
        # ratio = -result.fun if result.fun != 0 else 1e6
        if ratio > best_ratio:
            best_ratio = ratio

    return best_ratio

# --- Hybrid Genetic Algorithm ---

def genetic_algorithm(k, n, m, 
                      pop_size=20, 
                      fast_generations=50,
                      lp_every = 10,
                      stall_generations=20,
                      mutation_rate=0.1,
                      tournament_size=3,
                      top_k_final=5, 
                      num_workers=4,
                      verbose=True, 
                      **kwargs):
    """
    Run a genetic algorithm to search for the optimal P (with integer values)
    such that the m-height of the systematic generator matrix G = [I | P] is minimized.
    Uses tournament selection for parent choice.
    """
    n_minus_k = n - k
    # Initialize population of candidate P matrices
    best_P_candidates = kwargs.get("best_P_candidates", None) # should be a list
    if best_P_candidates is not None:
        population = best_P_candidates
        current_pop_size = len(population)
        if current_pop_size < pop_size:
            # Fill the rest of the population with random candidates
            population += [generate_random_P(k, n_minus_k) for _ in range(pop_size - current_pop_size)]
    else:
        population = [generate_random_P(k, n_minus_k) for _ in range(pop_size)]

    fitness = [new_m_height(construct_G(P), m) for P in population]
    best_fitness = min(fitness)
    best_P = population[np.argmin(fitness)]
    no_improve = 0

    # Initial evaluation using LP
    lp_fitness = [compute_hm(construct_G(P), m) for P in population]
    best_lp_fitness = min(lp_fitness)
    best_lp_P = population[np.argmin(lp_fitness)]

    if verbose:
        print("Initial best m-height:", best_fitness)
        print("Initial best LP m-height:", best_lp_fitness)

    for gen in range(fast_generations):

        # ========== GA with heuristic ==========
        new_population = []
        # Generate offspring until we have a full new population
        while len(new_population) < pop_size:
            # Use tournament selection to pick parents
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        new_fitness = [new_m_height(construct_G(P), m) for P in new_population]
        # Combine old and new populations and select the best pop_size individuals
        combined_population = population + new_population
        combined_fitness = fitness + new_fitness
        sorted_indices = np.argsort(combined_fitness)
        population = [combined_population[i] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]
        current_best = fitness[0]
        print(f"Gen {gen+1}: Best heurisitc m-height = {current_best:.4f}")

        if current_best < best_fitness:
            best_fitness = current_best
            best_P = population[0]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stall_generations:
            print("No improvement for", stall_generations, "generations. Stopping early.")
            break

        # ========== LP refinement ==========
        if (gen+1) % lp_every == 0:
            if verbose:
                print("🔍 LP-refinement step")
            population, fitness, refined_P, refined_hm = lp_guided_refinement(
                top_P_list=population[:5],
                m=m,
                generations=1,
                pop_size=pop_size,
                mutation_rate=mutation_rate,
                verbose=True
            )

            if refined_hm < best_lp_fitness:
                best_lp_P, best_lp_fitness = refined_P, refined_hm
            if verbose:
                print(f"  LP-refined h_m = {refined_hm:.4f}")


    # final evaluation using LP
    best_results = rerank_with_exact_lp_parallel(population, m, top_k=top_k_final, num_workers=num_workers)
    # save top 5 candidates
    top_candidates = []
    top_m_heights = []
    top_sampled_m_heights = []
    top_candidates.append(best_lp_P)
    top_m_heights.append(best_lp_fitness)
    top_sampled_m_heights.append(new_m_height(construct_G(best_lp_P), m))
    for i, result in enumerate(best_results):
        if i==0:
            best_P_exact, best_hm_exact = result
        if i > 5:
            break
        
        top_candidates.append(result[0])
        top_m_heights.append(result[1])
        top_sampled_m_heights.append(new_m_height(construct_G(result[0]), m))

    # logging dict
    log_outputs(n,k,m,
                top_candidates,
                top_m_heights,
                top_sampled_m_heights,
                generations=fast_generations)

    return best_P_exact, best_hm_exact

if __name__ == "__main__":
    # # Example experiment: for n=9, k=4, choose m (e.g., m=2 or any value in {2, ..., n-k})
    # k = 4
    # n = 9
    # m = 2  # Adjust as needed

    # code = f"({n},{k},{m})"
    # p1_results_json = "./p1_results.json"
    # best_P_candidates = load_candidates(p1_results_json, target_code=code)

    # best_P, best_mheight = genetic_algorithm(k, n, m, 
    #                                         pop_size=20, 
    #                                         fast_generations=10,
    #                                         lp_every = 10,
    #                                         stall_generations=20,
    #                                         mutation_rate=0.1,
    #                                         tournament_size=3,
    #                                         top_k_final=5, 
    #                                         num_workers=4,
    #                                         verbose=True, 
    #                                         best_P_candidates=best_P_candidates)

    # print("\nBest candidate P matrix:")
    # print(best_P)
    # print("Best m-height achieved:", best_mheight)

    # target_parameters = [
    #     (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    #     (9, 5, 2), (9, 5, 3), (9, 5, 4),
    #     (9, 6, 2), (9, 6, 3),
    #     (10, 4, 2), (10, 4, 3), (10, 4, 4), (10, 4, 5), (10, 4, 6),
    #     (10, 5, 2), (10, 5, 3), (10, 5, 4), (10, 5, 5),
    #     (10, 6, 2), (10, 6, 3), (10, 6, 4),
    # ]

    parser = argparse.ArgumentParser(description='Run hybrid genetic algorithm')
    parser.add_argument('--job_path', type=str)

    args=parser.parse_args()
    job_path = args.job_path
    with open(job_path, 'r') as f:
        lines = f.readlines()
        target_parameters = [line.strip() for line in lines]

    for target_code in target_parameters:
        # n, k, m = target_code
        n, k, m = map(int, target_code.strip('()').split(','))
        code = f"{n}_{k}_{m}"
        p1_results_json = './p1_results.json'
        best_P_candidates = load_candidates(p1_results_json, target_code=target_code)

        print(f"Running genetic algorithm for n={n}, k={k}, m={m}")
        best_P, best_mheight = genetic_algorithm(k, n, m, 
                                            pop_size=20, 
                                            fast_generations=100,
                                            lp_every = 10,
                                            stall_generations=20,
                                            mutation_rate=0.1,
                                            tournament_size=3,
                                            top_k_final=5, 
                                            num_workers=4,
                                            verbose=True, 
                                            best_P_candidates=best_P_candidates)
        print(f"Best candidate P matrix for n={n}, k={k}, m={m}:")
        print(best_P)
        print(f"Best m-height achieved: {best_mheight}\n")

        # # save results
        # output_matrix = f"./tournament/matrix/G_{n}_{k}_{m}.txt"
        # output_height = f"./tournament/height/H_{n}_{k}_{m}.txt"
        # np.savetxt(output_matrix, best_P, fmt='%d')
        # with open(output_height, 'w') as f:
        #     f.write(str(best_mheight))