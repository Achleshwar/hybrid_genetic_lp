import os
import numpy as np


# --- Utility functions ---
def generate_random_P(k, n_minus_k, low=-100, high=100):
    """
    Generate a k x (n-k) matrix with random integers between low and high.
    Ensure that no column is all zero.
    """
    P = np.random.randint(low, high + 1, size=(k, n_minus_k))
    for j in range(n_minus_k):
        if np.all(P[:, j] == 0):
            i = np.random.randint(0, k)
            choices = [x for x in range(low, high + 1) if x != 0]
            P[i, j] = np.random.choice(choices)
    return P

def construct_G(P):
    """
    Construct the systematic generator matrix G = [I | P] from P.
    """
    k = P.shape[0]
    I = np.eye(k, dtype=int)
    G = np.hstack((I, P))
    return G.astype(float)

# --- Genetic operators for integer matrices ---

def mutate(P, mutation_rate, low=-100, high=100):
    """
    Mutate the candidate matrix P by adding a small random integer perturbation to some entries.
    This version uses vectorized operations for efficiency.
    """
    k, n_minus_k = P.shape
    # Create a mutation mask where True indicates mutation
    mutation_mask = np.random.rand(k, n_minus_k) < mutation_rate
    # Generate noise between -5 and 5
    noise = np.random.randint(-5, 5, size=(k, n_minus_k))
    P_new = P.copy() + mutation_mask * noise
    # Clip values to remain within [low, high]
    P_new = np.clip(P_new, low, high)
    # Ensure no column is all zero
    for j in range(n_minus_k):
        if np.all(P_new[:, j] == 0):
            i = np.random.randint(0, k)
            choices = [x for x in range(low, high + 1) if x != 0]
            P_new[i, j] = np.random.choice(choices)
    return P_new

def crossover(P1, P2):
    """
    Create a child matrix from two parents by randomly choosing each element from one of the parents.
    Uses vectorized selection for speed.
    """
    k, n_minus_k = P1.shape
    mask = np.random.rand(k, n_minus_k) < 0.5
    P_new = np.where(mask, P1, P2)
    return P_new

def tournament_selection(population, fitness, tournament_size=3):
    """
    Select one individual from the population using tournament selection.
    """
    pop_size = len(population)
    # Randomly pick tournament_size individuals
    indices = np.random.choice(pop_size, size=tournament_size, replace=False)
    tournament_fitness = [fitness[i] for i in indices]
    winner_index = indices[np.argmin(tournament_fitness)]
    return population[winner_index]