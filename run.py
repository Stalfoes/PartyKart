from typing import Optional, Union, Any
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# with open('config.yaml', 'r') as file:
#     config = yaml.load(file, Loader)





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
# # ROSTER = """
# # Alex
# # Anton
# # Paul
# # Jensen
# # Matei
# # Alek
# # Eric
# # Michael
# # """
# ROSTER = [p for p in ROSTER.split('\n') if len(p) > 0]

# ROSTER = [
#     1, 2, 3, 4, 5, 6, 7, 8
# ]
# R = 5
# SEED = 0
# NUM_TRIALS = 10
# NUM_OPTIMIZE_ITERATIONS = 25000
# INITIAL_OPTIMIZE_TEMPERATURE = 1.0
# OPTIMIZE_COOLING_RATE = 0.005
# NUM_SWAP_ITERATIONS = 500000
# INITIAL_SWAP_TEMPERATURE = 1.0
# SWAP_COOLING_RATE = 0.0001

from perms import combs
from helper import Block
import perms
import solver
import even
from solver import find_best_races, order_races
# from helper import Block
from simple_graphics import pretty_plot_races, print_player_matrix
# all_pairs = list(combs(ROSTER, 2))
# solver.ALL_PAIRS = all_pairs
# even.ALL_PAIRS = all_pairs

def run_with_config(config:dict[str,Any], do_printing:bool=True) -> list[Block]:
    print(config)
    ROSTER:list[Any] = config['roster']
    R:int = config['num_races_each']
    LEFT_PEOPLE:Optional[list[Any]] = config['people_that_left']
    if LEFT_PEOPLE is None:
        LEFT_PEOPLE = []
    PREVIOUS_RACES:Optional[list[Any]] = config['previous_races']
    if config['previous_races'] is not None:
        PREVIOUS_RACES = [race.split() for race in PREVIOUS_RACES]
    else:
        PREVIOUS_RACES = []
    SEED:int = config['algorithmic']['seed']
    NUM_TRIALS:int = config['algorithmic']['num_trials']
    NUM_OPTIMIZE_ITERATIONS:int = config['algorithmic']['num_optimize_iterations']
    INITIAL_OPTIMIZE_TEMPERATURE:float = config['algorithmic']['initial_optimize_temperature']
    OPTIMIZE_COOLING_RATE:float = config['algorithmic']['optimize_cooling_rate']
    NUM_SWAP_ITERATIONS:int = config['algorithmic']['num_swap_iterations']
    INITIAL_SWAP_TEMPERATURE:float = config['algorithmic']['initial_swap_temperature']
    SWAP_COOLING_RATE:float = config['algorithmic']['swap_cooling_rate']

    # races = [[1,2,3,4],[5,6,7,8],[9,10,1,2],[3,4,5,6],[7,8,9,10],[1,2,3,4],[5,6,7,8],[9,10,1,2],[3,4,5,6],[7,8,9,10]]
    # races = [Block(race) for race in races]
    # print(even.fitness(races, list(combs(list(range(1,11)), 2))))
    # quit()

    # perms.generate_all_pairs(ROSTER)
    races, fitness = find_best_races(SEED, NUM_TRIALS, R, ROSTER, LEFT_PEOPLE, PREVIOUS_RACES, NUM_OPTIMIZE_ITERATIONS, INITIAL_OPTIMIZE_TEMPERATURE, OPTIMIZE_COOLING_RATE)
    if do_printing:
        print(f"Fitness = {fitness}")
        print(f"{repr(races)=}")
        print_player_matrix(ROSTER, races)
    swapped_races, swap_fitness, _ = order_races(SEED + 1, races, ROSTER, NUM_SWAP_ITERATIONS, LEFT_PEOPLE, PREVIOUS_RACES, INITIAL_SWAP_TEMPERATURE, SWAP_COOLING_RATE)
    all_races = [Block(race) for race in PREVIOUS_RACES] + swapped_races
    if do_printing:
        print(f"{repr(swapped_races)=}")
        print(f"{repr(all_races)=}")
        print(f"Swap Fitness = {swap_fitness}")
    return [list(race) for race in all_races]


def load_config() -> dict[str,Any]:
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader)
    return config


if __name__ == '__main__':
    races = run_with_config(load_config(), True)
    with open('generated_races.pckl', 'wb') as file:
        pickle.dump(races, file)
    print('Saved!')
    pretty_plot_races(races)
