from typing import Any, Sequence

import perms
from helper import Block


# ALL_PAIRS:list[tuple[Any]] = None
PERFECT_FITNESS = 0.0
PERFECT_SWAP_FITNESS = float('inf') # unknown!


def fitness(games:list[Block], all_pairs:list[tuple[Any,Any]]) -> float:
    """Same thing as evenness, except if there's an illegal block
    we make it negative infinity, since we absolutely don't want it.
    """
    for block in games:
        if block.is_invalid():
            # duplicate in the race!
            return -float('inf')
    return evenness(games, all_pairs)


def _variance(values:Sequence[Any]) -> tuple[Any, Any]:
    mean = sum(values) / len(values)
    variance = sum([(x - mean)**2 for x in values]) / len(values)
    return mean, variance


def evenness(games:list[Block], all_pairs:list[tuple[Any,Any]]) -> float:
    """Define some metric of evenness of a set of numbers.
    How close to equal are they?
    Maybe it's about minimizing the variance?
    """
    pair_counts = count_pairs(games, all_pairs)
    values = pair_counts.values()
    # mean = sum(values) / len(values)
    # variance = sum([(x - mean)**2 for x in values]) / len(values)
    mean, variance = _variance(values)
    return -variance
    # return -variance - 0.2 * (max(values) - min(values))


def count_pairs(games:list[Block], all_pairs:list[tuple[Any,Any]]) -> dict[tuple[Any],int]:
    counts = {pair: 0 for pair in all_pairs}
    for block in games:
        for pair in perms.combs(list(block), 2):
            counts[pair] += 1
    return counts


def spacing_fitness(games:list[Block], roster:Sequence[Any], left_people:list[Any], constant_races:list[Block]) -> float:
    """Provides a metric to maximize to increase spacing between all players.
    We are trying to minimize the variance in average spacings between players
    So, if player 1 has an average spacing of 2 between races but player 8 has an average of 3, the variance will be higher than
        if both players just had an average spacing of 2. We want to reduce that variance
    But also, maximizing the average spacings bof the players will try to spread them out as much as possible rather than making it fair
    """
    # Record what races a player plays in with the race indices
    player_appearances = {player: [] for player in roster if player not in left_people}
    # print(f"{player_appearances=}") # fine
    all_games = constant_races + games
    # print(f"{all_games=}")
    for block_idx, block in enumerate(all_games):
        for player in block:
            if player in left_people:
                continue
            player_appearances[player].append(block_idx)
    # Find the difference between the current race and the next-played race index for a player
    average_spacings = []
    for player,appearances in player_appearances.items():
        appearances.sort() # is this even necessary?
        if len(appearances) <= 1:
            # not necessary
            continue
        # find the differences between races in terms of time
        diffs = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
        avg_spacing = sum(diffs) / len(diffs)
        average_spacings.append(avg_spacing)
    mean_spacing, variance = _variance(average_spacings)
    return mean_spacing - variance
