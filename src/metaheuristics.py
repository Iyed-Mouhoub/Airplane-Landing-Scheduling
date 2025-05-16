# metaheuristics.py

import random
import numpy as np
from typing import List, Tuple
from src.landing_scheduler import compute_penalty, check_separation

class SimulatedAnnealing:
    def __init__(self, data, T0=1000, alpha=0.98, iter_per_temp=100):
        self.data = data
        self.T = T0
        self.alpha = alpha
        self.iter_per_temp = iter_per_temp
        self.best_order = None
        self.best_penalty = float('inf')

    def _initial_solution(self) -> List[int]:
        return list(range(len(self.data['earliest'])))

    def _neighbor(self, order: List[int]) -> List[int]:
        i, j = random.sample(range(len(order)), 2)
        new_order = order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order

    def _decode_times(self, order: List[int]) -> np.ndarray:
        return np.array([self.data['target'][i] for i in order])

    def run(self) -> Tuple[List[int], float]:
        order = self._initial_solution()
        times = self._decode_times(order)
        current_penalty = compute_penalty(times, **self.data)
        self.best_order, self.best_penalty = order, current_penalty

        while self.T > 1:
            for _ in range(self.iter_per_temp):
                candidate = self._neighbor(order)
                times_c = self._decode_times(candidate)
                if not check_separation(candidate, times_c):
                    continue
                p = compute_penalty(times_c, **self.data)
                delta = p - current_penalty
                if delta < 0 or random.random() < np.exp(-delta / self.T):
                    order, current_penalty = candidate, p
                    if p < self.best_penalty:
                        self.best_order, self.best_penalty = candidate, p
            self.T *= self.alpha

        return self.best_order, self.best_penalty


class GeneticAlgorithm:
    def __init__(self, data, pop_size=50, cx_rate=0.8, mut_rate=0.2, generations=100):
        self.data = data
        self.pop_size = pop_size
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.generations = generations

    def _init_pop(self) -> List[List[int]]:
        n = len(self.data['earliest'])
        return [random.sample(range(n), n) for _ in range(self.pop_size)]

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] < 0:
                child[i] = fill[idx]
                idx += 1
        return child

    def _mutate(self, individual: List[int]) -> None:
        if random.random() < self.mut_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def run(self) -> Tuple[List[int], float]:
        pop = self._init_pop()
        best, best_p = None, float('inf')

        for _ in range(self.generations):
            scored = []
            for ind in pop:
                times = np.array([self.data['target'][i] for i in ind])
                score = (float('inf') if not check_separation(ind, times)
                         else compute_penalty(times, **self.data))
                scored.append((ind, score))
                if score < best_p:
                    best, best_p = ind.copy(), score

            # Tournament selection
            new_pop = []
            for _ in range(self.pop_size):
                a, b = random.sample(scored, 2)
                winner = a if a[1] < b[1] else b
                new_pop.append(winner[0].copy())

            # Crossover & mutation
            pop = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = new_pop[i], new_pop[i+1]
                if random.random() < self.cx_rate:
                    c1 = self._crossover(p1, p2)
                    c2 = self._crossover(p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                self._mutate(c1)
                self._mutate(c2)
                pop.extend([c1, c2])

        return best, best_p


class VariableNeighborhoodSearch:
    def __init__(self, data, max_iter=1000):
        self.data = data
        self.max_iter = max_iter
        self.neighborhoods = [self._swap, self._insert, self._two_opt]

    def _decode_times(self, order: List[int]) -> np.ndarray:
        return np.array([self.data['target'][i] for i in order])

    def _swap(self, order: List[int]) -> List[int]:
        i, j = random.sample(range(len(order)), 2)
        order[i], order[j] = order[j], order[i]
        return order

    def _insert(self, order: List[int]) -> List[int]:
        i, j = random.sample(range(len(order)), 2)
        elem = order.pop(i)
        order.insert(j, elem)
        return order

    def _two_opt(self, order: List[int]) -> List[int]:
        i, j = sorted(random.sample(range(len(order)), 2))
        order[i:j] = reversed(order[i:j])
        return order

    def run(self) -> Tuple[List[int], float]:
        current = list(range(len(self.data['earliest'])))
        best = current.copy()
        best_p = compute_penalty(self._decode_times(best), **self.data)
        it = 0

        while it < self.max_iter:
            for neigh in self.neighborhoods:
                candidate = neigh(current.copy())
                times_c = self._decode_times(candidate)
                if not check_separation(candidate, times_c):
                    continue
                p = compute_penalty(times_c, **self.data)
                if p < best_p:
                    best, best_p = candidate.copy(), p
                    current = candidate.copy()
                    break
            it += 1

        return best, best_p
    