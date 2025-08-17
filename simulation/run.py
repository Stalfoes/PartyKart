import jax
import jax.numpy as jnp
from tqdm import tqdm

RNGType = jax.typing.ArrayLike


TRUE_ELO = {
    'Luke'      :   1300,
    'Rony'      :    971,
    'Anton'     :   1115,
    'Paul'      :   1200,
    'Jensen'    :   1000,
    'Matei'     :    750,
    'Alek'      :    980,
    'Eric'      :   1429,
    'Michael'   :   1100,
}

def elo_prob_win(us:float, them:float) -> float: 
    return 1 / (1 + jnp.power(10, (them - us) / 480))

# print(elo_prob_win(10000, 1200))

# normal_dist = lambda x, stddev, mean: 1 / jnp.sqrt(2 * jnp.pi * jnp.square(stddev)) * jnp.exp(-(jnp.square(x - mean) / (2 * jnp.square(stddev))))
# confidence = lambda p_win: max(0,normal_dist(0.5,0.3,0.5) - normal_dist(p_win,0.3,0.5))

EXPECTED_PLACEMENT = sorted(TRUE_ELO.keys(), reverse=True, key=lambda k: TRUE_ELO[k])
# RESULT_CONFIDENCE = 

# print(EXPECTED_PLACEMENT)

brackets = [['Jensen', 'Rony', 'Anton', 'Michael'], ['Eric', 'Anton', 'Luke', 'Paul'], ['Alek', 'Eric', 'Anton', 'Michael'], ['Eric', 'Jensen', 'Matei', 'Alek'], ['Anton', 'Matei', 'Michael', 'Eric'], ['Jensen', 'Matei', 'Michael', 'Luke'], ['Luke', 'Alek', 'Rony'], ['Matei', 'Eric', 'Rony', 'Jensen'], ['Luke', 'Paul', 'Jensen', 'Michael'], ['Anton', 'Matei', 'Alek', 'Paul'], ['Alek', 'Michael', 'Paul', 'Rony'], ['Eric', 'Paul', 'Jensen', 'Rony'], ['Rony', 'Matei', 'Anton', 'Luke'], ['Jensen', 'Paul', 'Anton', 'Alek'], ['Paul', 'Matei', 'Michael', 'Luke'], ['Luke', 'Alek', 'Eric', 'Rony']]

def simulate_race(rng:RNGType, race:list[str]) -> list[str]:
    def who_wins(rng:RNGType, person1:str, person2:str) -> str:
        # check who wins between person1 and person2
        prob_1_win = elo_prob_win(TRUE_ELO[person1], TRUE_ELO[person2])
        if jax.random.uniform(rng) < prob_1_win:
            # person1 won
            return person1
        else:
            # person2 won
            return person2
    # print(f"{race=}")
    new_race = [race[0]] # just throw the dude in there
    for person in race[1:]: # for every other dude, figure out where he goes
        player_beat_everyone = True
        for i in range(len(new_race) - 1, -1, -1):
            opponent = new_race[i]
            rng, win_rng = jax.random.split(rng)
            winner = who_wins(win_rng, person, opponent)
            if winner == person:
                # then this person beat the opponent
                # keep climbing the ladder
                continue
            else:
                # we lost against against this person, stop here
                new_race.insert(i+1, person)
                player_beat_everyone = False
                break
        if player_beat_everyone:
            new_race.insert(0, person)
        
    # print(f"{new_race=}")
    return new_race

def give_points(race_placings:list[str]) -> dict[str,float]:
    if len(race_placings) == 4:
        return {
            person: points
            # for person,points in zip(race_placings, [24,16,8,0]) # 0.47%
            for person,points in zip(race_placings, [28,16,8,0]) # 0.77%
            # for person,points in zip(race_placings, [5,3,2,1]) # 0.76%
        }
    elif len(race_placings) == 3:
        return {
            person: points
            # for person,points in zip(race_placings, [20,12,4]) # 0.47%
            for person,points in zip(race_placings, [24,12,4]) # 0.77%
            # for person,points in zip(race_placings, [4,2,1]) # 0.76%
        }
    else:
        raise ValueError(f"Something went wrong, the placings is size {len(race_placings)}")

def simulate_all_races(rng:RNGType, races:list[str]) -> tuple[dict[str,float], list[list[str]]]:
    points = {person:0 for person in TRUE_ELO.keys()}
    all_results = []
    for race in races:
        rng, race_rng = jax.random.split(rng)
        race_results = simulate_race(race_rng, race)
        point_increases = give_points(race_results)
        for person,increase in point_increases.items():
            points[person] += increase
        all_results.append(race_results)
    return points, all_results

def simulate_N_times(rng:RNGType, trials:int) -> float:
    # find the accuracy of the point giving and the simulation I guess too
    score = 0
    for trial in tqdm(range(trials)):
        rng, trial_rng = jax.random.split(rng)
        points, results = simulate_all_races(trial_rng, brackets)
        # print(points)
        # print(results)
        # exit()
        ordering = sorted(points.keys(), reverse=True, key=lambda k: points[k])
        if ordering == EXPECTED_PLACEMENT:
            score += 1
    return score / trials

if __name__ == '__main__':
    SEED = 0
    N_TRIALS = 10000
    score = simulate_N_times(jax.random.key(SEED), N_TRIALS)
    print(f"Score = {score*100}%")