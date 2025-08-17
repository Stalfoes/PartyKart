# ROSTER = """
# Luke
# Rony
# Anton
# Paul
# Jensen
# Matei
# Alek
# Eric
# Michael
# """
# ROSTER = [p for p in ROSTER.split('\n') if len(p) > 0]

ROSTER = [
    1, 2, 3, 4, 5, 6, 7, 8
]
R = 7
SEED = 1
NUM_TRIALS = 50
NUM_OPTIMIZE_ITERATIONS = 10000
INITIAL_OPTIMIZE_TEMPERATURE = 1.0
OPTIMIZE_COOLING_RATE = 0.0025
NUM_SWAP_ITERATIONS = 1000000
INITIAL_SWAP_TEMPERATURE = 1.0
SWAP_COOLING_RATE = 0.0001

import pickle
from tqdm import tqdm
# from perms import combs
from perms import generate_all_pairs
from solver import find_best_races, order_races
# from helper import Block
# from graphics import pretty_plot_races, print_player_matrix


if __name__ == '__main__':
    for n in tqdm(range(5,10)):
        ROSTER = list(range(0,n))
        generate_all_pairs(ROSTER)
        for r in range(2,11):
            R = r
            # all_pairs = list(combs(ROSTER, 2))
            # solver.ALL_PAIRS = all_pairs
            # even.ALL_PAIRS = all_pairs
            # print(ROSTER, R)
            races, fitness = find_best_races(SEED, NUM_TRIALS, R, ROSTER, NUM_OPTIMIZE_ITERATIONS, INITIAL_OPTIMIZE_TEMPERATURE, OPTIMIZE_COOLING_RATE)
            # print(f"Fitness = {fitness}")
            # print_player_matrix(ROSTER, races)
            swapped_races, swap_fitness, _ = order_races(SEED + 1, races, ROSTER, NUM_SWAP_ITERATIONS, INITIAL_SWAP_TEMPERATURE, SWAP_COOLING_RATE)
            # print(f"Swap Fitness = {swap_fitness}")
            with open(f"/Users/kapeluck/Documents/Personal/RonyPartyKart/results/result_{n}_{r}.result", 'wb') as result_file:
                pickle.dump((swapped_races,fitness,swap_fitness))
            print(f'done {n=} {r=}')
    print('done!')