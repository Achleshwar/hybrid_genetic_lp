from itertools import combinations, product
from scipy.optimize import linprog
import numpy as np

from ga_utils import generate_random_P, construct_G, tournament_selection, crossover, mutate

def solve_lp(G, a, b, X, psi, m):
    k, n = G.shape

    # Constraints for j in Y
    Y = [j for j in range(n) if j not in X and j != a and j != b]

    # Tau
    tau = [0]*n
    tau[0] = a
    for j in range(1, m):
        tau[j] = X[j-1]
    tau[m] = b
    for j in range(n-m-1):
        tau[m+1+j] = Y[j]

    # Tau Inverse
    tau_inv = [0]*n
    for i in range(n):
        tau_inv[tau[i]]=i

    # Objective function: maximize sum(s[0] * g[i,a] * u[i])
    c = -np.array([psi[0] * G[i,a] for i in range(k)])  # Negated for minimization

    # Constraints
    A_ub = []
    b_ub = []

    for j in X:
        # Make sure tau_inv[j] does not exceed psi index bounds
        if tau_inv[j] >= m:
            continue  # skip this j, since it refers to a psi index that doesn't exist
        s_tau_j = psi[tau_inv[j]]
        A_ub.append([s_tau_j * G[i,j] - psi[0] * G[i,a] for i in range(k)])
        b_ub.append(0)

        A_ub.append([-s_tau_j * G[i,j] for i in range(k)])
        b_ub.append(-1)

    # Constraints for column b
    A_eq = [[G[i,b] for i in range(k)]]
    b_eq = [1]

    for j in Y:
        A_ub.append([G[i,j] for i in range(k)])
        b_ub.append(1)

        A_ub.append([-G[i,j] for i in range(k)])
        b_ub.append(1)

    # Solve LP using linprog
    res = linprog(c=c,
                  A_ub=np.array(A_ub),
                  b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq),
                  b_eq=np.array(b_eq),
                  bounds = [(None, None)]*k,
                  method='highs')

    if res.status == 3:
        return float('inf')
    if res.success:
        return -res.fun  # Negate because we minimized negative objective
    else:
        return 0 # Infeasible
def compute_hm(G, m):
    """
    G: 2-D numpy array
    m: int
    """
    k, n = G.shape

    if m == 0:
        return 1.0  # h_0(C) is always 1

    max_hm = 0

    # Iterate over all possible tuples (a, b, X, psi)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue

            # Generate all subsets X of size m-1 from [n] excluding {a, b}
            remaining_indices = [j for j in range(n) if j != a and j != b]
            for X in combinations(remaining_indices, m-1):

                # Generate all possible binary vectors psi of length m
                for psi in product([-1, 1], repeat=m):
                    # Solve LP for this tuple (a, b, X, psi)
                    z = solve_lp(G, a, b, X, psi, m)

                    # Update max_hm if necessary
                    max_hm = max(max_hm, z)

    return max_hm

def lp_guided_refinement(top_P_list, m, generations=20, pop_size=10, mutation_rate=0.1,
                         stall_generations=10, num_workers=4, verbose=True):
    """
    Given top P matrices from earlier heuristic GA, refine them using LP-evaluated GA.
    """
    k, n_minus_k = top_P_list[0].shape
    population = top_P_list.copy()
    population.append(top_P_list[0])  # Add the best candidate to the population

    while len(population) < pop_size:
        p1_idx, p2_idx = np.random.choice(len(top_P_list), size=2, replace=True)
        p1, p2 = top_P_list[p1_idx], top_P_list[p2_idx]
        child = mutate(crossover(p1, p2), mutation_rate)
        population.append(child)

    # Evaluate using exact LP
    fitness = [compute_hm(construct_G(P), m) for P in population]
    best_P = population[np.argmin(fitness)]
    best_fitness = min(fitness)
    no_improve = 0

    if verbose:
        print(f"🔁 LP-Refinement Start: Best LP m-height = {best_fitness:.4f}")

    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            p1_idx, p2_idx = np.random.choice(len(population), size=2, replace=True)
            p1, p2 = population[p1_idx], population[p2_idx]
            child = mutate(crossover(p1, p2), mutation_rate)
            new_population.append(child)

        new_fitness = [compute_hm(construct_G(P), m) for P in new_population]

        combined_population = population + new_population
        combined_fitness = fitness + new_fitness
        sorted_indices = np.argsort(combined_fitness)
        population = [combined_population[i] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]

        current_best = fitness[0]
        if verbose:
            print(f"Gen {gen+1:>2}: Best LP m-height = {current_best:.4f}")

        if current_best < best_fitness:
            best_fitness = current_best
            best_P = population[0]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stall_generations:
            if verbose:
                print(f"⏹️  No improvement for {stall_generations} gens. Stopping.")
            break

    return population, fitness, best_P, best_fitness