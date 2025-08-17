from typing import Sequence, Any, Iterable
import jax
import jax.numpy as jnp

from helper import astuple, Block, RNGType
import deeper_config as cfg



# all_pairs:list[tuple[Any,Any]] = None
# def generate_all_pairs(roster:Sequence[Any]) -> None:
#     global all_pairs
#     all_pairs = list(combs(roster, 2))

# def get_all_pairs() -> list[tuple[Any,Any]]:        
#     return all_pairs


def combs(items:Iterable[Any], size:int, can_duplicate:bool=False) -> set[tuple[Any]]:
    if size == 1:
        return {(item,) for item in items}
    all_combs = set()
    for thing in items:
        if can_duplicate:
            remaining_things = {*items}
        else:
            remaining_things = {*items} - {thing}
        for sub_comb in combs(remaining_things, size - 1, can_duplicate):
            comb = astuple((thing, *sub_comb))
            all_combs.add(comb)
    return all_combs

def kartcombs(items):
    return combs(items, 4) | combs(items, 3)

def create_starting_blocks(roster:Sequence[Any], r:int, left_people:Sequence[Any], constant_races:list[list[Any]], rng:RNGType) -> tuple[list[Block], RNGType]:
    def generate_random_sequence(rng:RNGType) -> list[Any]:
        # Create a sequence of length r*len(roster) that is entirely random
        if len(left_people) == 0 and len(constant_races) == 0:
            def single(rng:RNGType) -> list[Any]:
                # create a sequence of length len(roster) where each player is only used once
                indices = jax.random.permutation(rng, jnp.arange(len(roster)), independent=True)  # shuffle indices
                return [roster[idx] for idx in indices]                     # convert indices back to player
            # Just generate random single shuffles and append all together
            total = []
            for _ in range(r):
                rng, _1 = jax.random.split(rng)
                total += single(rng)
            return total
        else:
            # Special case! We have some races that already happened and/or some people that left
            # Find what people are still here and need to be reshuffled
            need_to_randomize_roster = list(roster)
            # for person_that_left in left_people:
            #     need_to_randomize_roster.remove(person_that_left)
            # Count how many times they need to be in the remaining races
            people_and_races_needed_to_happen = {
                person: r for person in need_to_randomize_roster
            }
            for race in constant_races:
                for person in race:
                    # if person in left_people:
                        # continue
                    people_and_races_needed_to_happen[person] -= 1
            for person in left_people:
                people_and_races_needed_to_happen[person] = 0
            # Create a list of those people with names repeated the number of times they have to race
            need_to_randomize_roster = []
            for person,cnt in people_and_races_needed_to_happen.items():
                need_to_randomize_roster += [person] * cnt
            # print(f"{need_to_randomize_roster=}")
            # quit()
            # Shuffle them and return the shuffled
            indices = jax.random.permutation(rng, jnp.arange(len(need_to_randomize_roster)), independent=True)
            return [need_to_randomize_roster[idx] for idx in indices]
    def split_into_races(sequence:Sequence[Any]) -> list[list[Any]]:
        # split into groups of 4, being greedy. 3 only if necessary
        races = [list(sequence[i:i+cfg.BEST_RACE_SIZE]) for i in range(0,len(sequence),cfg.BEST_RACE_SIZE)]
        # float up racers so we have maximal groups of 4 but 3 if necessary
        while True:
            # check if all races are length 3 or 4
            all_good = all([len(block) in cfg.TOLERABLE_RACE_SIZES for block in races])
            if all_good:
                break
            # Not okay? Then we need to float one up from one race before it to the race that needs a person
            for idx in range(len(races)-1, -1, -1):
                block = races[idx]
                if len(block) not in cfg.TOLERABLE_RACE_SIZES:
                    floater = races[idx-1].pop()
                    races[idx].insert(0, floater)
                    break
        return races
    def races_all_legal(races:list[list[Any]]) -> bool:
        # check if all the races contain only unique players. No player repeats in a race
        if len(left_people) > 0 or len(constant_races) > 0:
            # if we're dealing with the special case, we want to check. Because otherwise we'll be stuck in like an
            #   infinite plane of just -inf fitness and never escape it. This is just some randomized initialization
            #   to hopefully escape it and hit some gradient of some kind
            return all([not Block(block).is_invalid() for block in races])
        else:
            # if we're dealing with the normal case, don't worry about it
            return True
    rng, rng1 = jax.random.split(rng)
    sequence = generate_random_sequence(rng1)   # Generate a random sequence of length len(roster) * r
    races = split_into_races(sequence)          # Split into races of size 4 or 3 (being greedy for 4)
    while not races_all_legal(races):           # Check if any race has someone racing against themselves
        # While any race is illegal, just create a new one randomly until we get a legal one
        rng, rng1 = jax.random.split(rng)
        sequence = generate_random_sequence(rng1)
        races = split_into_races(sequence)
    races = [Block(block) for block in races]   # Convert individual races to Block type
    return races, rng


if __name__ == '__main__':
    PEOPLE = list(range(1,15))
    KC = kartcombs(PEOPLE)
    print(len(KC))
    print(len(combs(PEOPLE,4)), len(combs(PEOPLE,3)))
    # print(' '.join(repr(c) for c in KC))
