import random
import numpy as np
from typing import List, Tuple
from src.landing_scheduler import compute_penalty, repair_schedule


class SimulatedAnnealing:
    def __init__(self, data, T0=1000, alpha=0.98, iter_per_temp=100):
        self.data = data
        self.T0 = T0
        self.alpha = alpha
        self.iter_per_temp = iter_per_temp

    def _initial_solution(self) -> List[int]:
        return list(range(len(self.data['earliest'])))

    def _neighbor(self, order: List[int]) -> List[int]:
        i, j = random.sample(range(len(order)), 2)
        new_order = order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order

    def run(self) -> Tuple[List[int], float]:
        sep = 4.0
        # initial solution
        order = self._initial_solution()
        _, times = repair_schedule(order,
                                   self.data['earliest'],
                                   self.data['target'],
                                   separation=sep)
        current_pen = compute_penalty(times, **self.data)
        best_order, best_pen = order.copy(), current_pen
        T = self.T0

        while T > 1e-3:
            for _ in range(self.iter_per_temp):
                candidate = self._neighbor(order)
                # repair then score
                _, times_c = repair_schedule(candidate,
                                             self.data['earliest'],
                                             self.data['target'],
                                             separation=sep)
                p = compute_penalty(times_c, **self.data)
                # accept criterion
                if p < current_pen or random.random() < np.exp((current_pen - p) / T):
                    order = candidate
                    current_pen = p
                    if p < best_pen:
                        best_pen = p
                        best_order = candidate.copy()
            T *= self.alpha

        return best_order, best_pen


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
        sep = 4.0
        pop = self._init_pop()
        best_order, best_pen = None, float('inf')

        for _ in range(self.generations):
            scored = []
            for ind in pop:
                # repair then score
                _, times = repair_schedule(ind,
                                           self.data['earliest'],
                                           self.data['target'],
                                           separation=sep)
                score = compute_penalty(times, **self.data)
                scored.append((ind, score))
                if score < best_pen:
                    best_order, best_pen = ind.copy(), score

            # tournament selection
            new_pop = []
            for _ in range(self.pop_size):
                a, b = random.sample(scored, 2)
                winner = a if a[1] < b[1] else b
                new_pop.append(winner[0].copy())

            # crossover & mutation
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

        return best_order, best_pen


class VariableNeighborhoodSearch:
    def __init__(self, data, max_iter=1000):
        self.data = data
        self.max_iter = max_iter
        self.neighborhoods = [self._swap, self._insert, self._two_opt]

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
        sep = 4.0
        current = list(range(len(self.data['earliest'])))
        # repair initial and score
        _, times = repair_schedule(current,
                                   self.data['earliest'],
                                   self.data['target'],
                                   separation=sep)
        best, best_pen = current.copy(), compute_penalty(times, **self.data)
        it = 0

        while it < self.max_iter:
            for neigh in self.neighborhoods:
                candidate = neigh(current.copy())
                # repair then score
                _, times_c = repair_schedule(candidate,
                                             self.data['earliest'],
                                             self.data['target'],
                                             separation=sep)
                p = compute_penalty(times_c, **self.data)
                if p < best_pen:
                    best, best_pen = candidate.copy(), p
                    current = candidate.copy()
                    break
            it += 1

        return best, best_pen
