from typing import Any, Sequence
from copy import deepcopy
import jax
import jax.numpy as jnp
from tqdm import tqdm
import multiprocessing as mp

from helper import Block, RNGType
from even import fitness, count_pairs, PERFECT_FITNESS, spacing_fitness, PERFECT_SWAP_FITNESS
from perms import create_starting_blocks, combs
import perms
import deeper_config as cfg


# ALL_PAIRS:list[tuple[Any]] = None


class NoOpSwapException(ValueError):
    pass


def find_swap(games:list[Block], pairA:tuple[Any], pairB:tuple[Any], rng:RNGType) -> tuple[list[Block], RNGType]:
    # Check if the swap makes any sense
    common_values = set(pairA) & set(pairB)
    if len(common_values) != 1:
        raise NoOpSwapException(f"Pairs must have ONE value in common. {pairA=}, {pairB=}")
    # Get the two values we want to swap
    values_to_swap = (*(set(pairA) - common_values), *(set(pairB) - common_values))
    # find block with pairA in it
    pairA_block_indices = []
    for idx,block in enumerate(games):
        if block.containsall(pairA):
            # The swap might be legal and good
            pairA_block_indices.append(idx)
    if len(pairA_block_indices) == 0:
        # Something is wrong I think
        # There should be a pair that exists in a block
        # raise ValueError(f"Something is wrong. Searched for {pairA} but did not find any in blocks. pairA should be the more-common pair. Blocks: {games}")
        raise NoOpSwapException(f"Searched for {pairA} but did not find any in blocks.")
    # Find candidate blocks to throw values_to_swap[0] into instead of values_to_swap[1]
    blocks_with_B_value_but_neither_A = []
    for idx,block in enumerate(games):
        if values_to_swap[1] in block and not block.containsany(pairA):
            blocks_with_B_value_but_neither_A.append(idx)
    if len(blocks_with_B_value_but_neither_A) == 0:
        # Then it's not possible to affect anything meaningfully
        raise NoOpSwapException(f"Can't make a meaningful swap.")
    # Filter the pairA games to ones that don't have values_to_swap[0]
    pairA_block_indices = [i for i in pairA_block_indices if values_to_swap[1] not in games[i]]
    if len(pairA_block_indices) == 0:
        # Impossible to make the swap to affect anything meaningfully
        raise NoOpSwapException(f"Can't make a meaningful swap 2.")
    # Make the swap
    ret = deepcopy(games)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    i = jax.random.choice(rng1, jnp.asarray(pairA_block_indices))
    j = jax.random.choice(rng2, jnp.asarray(blocks_with_B_value_but_neither_A))
    ret[i].remove(values_to_swap[0])
    ret[i].add(values_to_swap[1])
    ret[j].remove(values_to_swap[1])
    ret[j].add(values_to_swap[0])
    return ret, rng


def swaps_and_differences(games:list[Block], all_pairs:list[tuple[Any,Any]], left_people:list[Any]) -> tuple[list[tuple[tuple[Any],tuple[Any]]], jnp.ndarray]:
    """Consider all the swaps we can make and provide their differences in how much one pair is counted over another
    """
    pair_counts = count_pairs(games, all_pairs) # returns all pairs (1,2),(2,3),(1,2)... as keys and the number of times we find them in the races as the values in a dictionary
    pair_differences = {(pair1,pair2):pair_counts[pair1] - pair_counts[pair2] for pair1 in all_pairs for pair2 in all_pairs}
    positive_pair_differences = {}
    for (pair1,pair2),diff in pair_differences.items():
        if Block(pair1).containsany(pair2) == False or Block(pair1).containsall(pair2):
            continue # not a valid swap! It's either two mutually exclusive pairs or the same pair
        if Block(pair1).containsany(left_people) or Block(pair2).containsany(left_people):
            continue # not a valid swap. Contains people that left. We don't care about these swaps, they won't be anything valid
        if diff >= 0:
            positive_pair_differences[(pair1,pair2)] = diff
        elif diff < 0:
            continue # we've already got the positive version
    keys = list(positive_pair_differences.keys())
    differences = jnp.asarray([positive_pair_differences[k] for k in keys])
    return keys, differences

# Generate a starting set of blocks that are valid and satisfy the `r` constraint. In this case, players=[1...9] and r=3
#       1234 5678 9123 4567 8912 3456 789
# evaluate the important metric, evenness, and go through a list of swaps to try, starting with the biggest difference in counts
#       most common (1,2), least common (1,5)
# try performing the swaps in order
#       candidates for A without 5:
#           1234 9123 8912
#       candidates for 5 without 1,2:
#           5678 4567 3456
#       take 1234 and replace 2 with 5 and take 5678 and replace 5 with 2
#       1345 2678 9123 4567 8912 3456 789
# did it improve fitness? keep it, and repeat from step 2
# if not, try the next swap

def optimize(starting_races:list[Block], rng:RNGType, max_iterations:int, all_pairs:list[tuple[Any,Any]], left_people:Sequence[Any], constant_races:list[list[Any]], initial_temperature:float, cooling_rate:float) -> tuple[list[Block], float, float]:
    """Fully optimize the races provided until nothing can be optimized anymore and then return the fitness too and the starting fitness
    """
    # print(f"{starting_races=}")
    def all_races(other_races:list[Block]) -> list[Block]:
        return constant_races + other_races 
    starting_fitness = fitness(all_races(starting_races), all_pairs)
    current_races = deepcopy(starting_races)
    current_fitness = starting_fitness
    for t in tqdm(range(1, max_iterations + 1), leave=False):
        temperature = initial_temperature / (1 + cooling_rate * t) # linear cooling
        # current_fitness = fitness(current_races) # I don't think we need to re-calculate this here. We have it already
        # Get all the possible swaps and the difference in counts between the pairs
        possible_swaps, pair_differences = swaps_and_differences(all_races(current_races), all_pairs, left_people)
        # print(f"{possible_swaps=}")
        # quit()
        # Get the swap selection probabilities and choose a swap
        probabilities = pair_differences / pair_differences.sum()
        rng, rng_choice = jax.random.split(rng)
        swap_index = jax.random.choice(rng_choice, jnp.arange(len(possible_swaps)), p=probabilities)
        pairA, pairB = possible_swaps[swap_index]
        try:
            new_races, rng = find_swap(current_races, pairA, pairB, rng)
            # print('made a swap')
        except NoOpSwapException as e:
            continue # ignore, cannot perform the swap. Move on
        except ValueError as e:
            raise e # this is an actual error we care about because it means we did something wrong
        new_fitness = fitness(all_races(new_races), all_pairs)
        # if new_fitness > -float('inf'):
            # print(f'doing better than -inf: {new_fitness=}')
        if new_fitness == PERFECT_FITNESS or abs(new_fitness - PERFECT_FITNESS) < cfg.FITNESS_EXACT_TOLERANCE:
            # We found a perfect solution. This is what we want. Exit here. No need to iterate more
            return new_races, new_fitness, starting_fitness
        delta_fitness = new_fitness - current_fitness
        if delta_fitness > 0:
            # take the swap. Greedy
            current_races = new_races
            current_fitness = new_fitness
        else:
            rng, rng_entropy = jax.random.split(rng)
            probability = jnp.exp(delta_fitness / temperature) if temperature > 0 else 0
            if jax.random.uniform(rng_entropy) < probability:
                # take the swap anyways if we have the entropy for it
                current_races = new_races
                current_fitness = new_fitness
    return current_races, current_fitness, starting_fitness


def perform_trial(queue:mp.Queue, rng:RNGType, roster:Sequence[Any], r:int, all_pairs:list[tuple[Any,Any]], left_people:Sequence[Any], constant_races:list[list[Any]], iterations:int, initial_temperature:float, cooling_rate:float) -> tuple[list[Block], float]:
    constant_races = [Block(race) for race in constant_races]
    # print(f" CONSTANT RACES = {constant_races=}")
    other_races, rng = create_starting_blocks(roster, r, left_people, constant_races, rng)
    # print(f" Other races = {other_races=}")
    other_races, race_fitness, _ = optimize(other_races, rng, iterations, all_pairs, left_people, constant_races, initial_temperature, cooling_rate)
    queue.put((other_races, race_fitness))


def find_best_races(seed:int, num_trials:int, r:int, roster:Sequence[Any], left_people:Sequence[Any], constant_races:list[list[Any]], optimize_iterations:int, initial_temperature:float, cooling_rate:float) -> tuple[list[Block], float]:
    # If we don't have someone in the roster that has already participated, just add them
    roster = list(roster)
    for race in constant_races:
        for person in race:
            if person not in roster:
                roster.append(person)
    # calculate all pairs of people in the roster
    all_pairs = list(combs(roster, 2, can_duplicate=True))
    # print(f"ALL PAIRS {all_pairs=}")
    # Run the parallel threads to perform a trial
    rng = jax.random.key(seed)
    rngs = jax.random.split(rng)
    queue = mp.Queue()
    processes:list[mp.Process] = []
    for i in range(num_trials):
        p = mp.Process(target=perform_trial, args=(queue, rngs[i], roster, r, all_pairs, left_people, constant_races, optimize_iterations, initial_temperature, cooling_rate))
        p.start()
        processes.append(p)
    # Wait for threads to finish and compare their performances. Find the best
    best_races = None
    best_fitness = -float('inf')
    for i in tqdm(range(num_trials)):
        processes[i].join()
        races, fitness = queue.get()
        if fitness == PERFECT_FITNESS or abs(fitness - PERFECT_FITNESS) < cfg.FITNESS_EXACT_TOLERANCE:
            # we can end here
            for j in range(i+1,num_trials):
                try:
                    processes[j].kill()
                except:
                    pass
            return races, fitness
        if fitness > best_fitness:
            best_races = races
            best_fitness = fitness
    if best_races is None:
        raise RuntimeError(f"best_races is somehow `None`. {best_fitness=}; Something went wrong in find_best_races.")
    return best_races, best_fitness

    # best_races, rng = create_starting_blocks(roster, r, rng)
    # rng, rng_optimize = jax.random.split(rng)
    # best_races, best_fitness, _ = optimize(best_races, rng_optimize, optimize_iterations, initial_temperature, cooling_rate)
    # for trial in tqdm(range(num_trials)):
    #     races, rng = create_starting_blocks(roster, r, rng)
    #     rng, rng_optimize = jax.random.split(rng)
    #     races, race_fitness, _ = optimize(races, rng_optimize, optimize_iterations, initial_temperature, cooling_rate)
    #     if race_fitness == PERFECT_FITNESS or abs(race_fitness - PERFECT_FITNESS) < cfg.FITNESS_EXACT_TOLERANCE:
    #         # The solution is perfect. We found what we wanted. End here!
    #         return races, race_fitness
    #     if race_fitness > best_fitness:
    #         best_races = races
    #         best_fitness = race_fitness
    # return best_races, best_fitness


def possible_race_swaps(races:list[Block]) -> list[tuple[int,int]]:
    """Output a list of possible swaps for the races. The swaps are labelled by indices to swap.
    """
    indices = list(range(len(races)))
    possible_swaps = combs(indices, 2)
    return list(possible_swaps)


def order_races(seed:int, races:list[Block], roster:Sequence[Any], num_iterations:int, left_people:Sequence[Any], constant_races:list[list[Any]], initial_temperature:float, cooling_rate:float) -> tuple[list[Block], float, float]:
    def reconstruct_races(ids:list[int]) -> list[Block]:
        return [races[idx] for idx in ids]
    def make_swap(ids:list[int], id1:int, id2:int) -> list[int]:
        # Make a swap of position for race id1 and race id2
        new_ids = deepcopy(ids)
        index1 = current_ordering.index(id1)
        index2 = current_ordering.index(id2)
        new_ids[index1],new_ids[index2] = new_ids[index2],new_ids[index1]
        return new_ids
    roster = list(roster)
    for race in constant_races:
        for person in race:
            if person not in roster:
                roster.append(person)
    constant_races = [Block(race) for race in constant_races]
    # print(f"{constant_races=}") # fine
    rng = jax.random.key(seed)
    starting_fitness = spacing_fitness(races, roster, left_people, constant_races)
    current_ordering = list(range(len(races)))
    current_fitness = starting_fitness
    possible_swaps = possible_race_swaps(races)
    for t in tqdm(range(1, num_iterations + 1)):
        temperature = initial_temperature / (1 + cooling_rate * t) # linear cooling
        # randomly choose a swap to select
        rng, rng_swap = jax.random.split(rng)
        swap_index = jax.random.choice(rng_swap, jnp.arange(len(possible_swaps)))
        id1, id2 = possible_swaps[swap_index]
        new_ordering = make_swap(current_ordering, id1, id2)
        new_fitness = spacing_fitness(reconstruct_races(new_ordering), roster, left_people, constant_races)
        if new_fitness == PERFECT_SWAP_FITNESS or abs(new_fitness - PERFECT_SWAP_FITNESS) < cfg.FITNESS_EXACT_TOLERANCE:
            # we found the best solution! no need to continue
            return reconstruct_races(new_ordering), new_fitness, starting_fitness
        delta_fitness = new_fitness - current_fitness
        if delta_fitness > 0:
            # take the swap. Greedy
            current_ordering = new_ordering
            current_fitness = new_fitness
        else:
            rng, rng_entropy = jax.random.split(rng)
            probability = jnp.exp(delta_fitness / temperature) if temperature > 0 else 0
            if jax.random.uniform(rng_entropy) < probability:
                # take the swap anyways if we have the entropy for it
                current_ordering = new_ordering
                current_fitness = new_fitness
    return reconstruct_races(current_ordering), current_fitness, starting_fitness
