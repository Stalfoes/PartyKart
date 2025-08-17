from typing import Any
import pygame
from copy import deepcopy

from .leaderboard import Leaderboard
from .next_race import RaceDisplay
from .spinner import Spinner


class Controller:
    def __init__(self, race_display:RaceDisplay, leaderboard:Leaderboard, spinner:Spinner):
        self.race_display = race_display
        self.leaderboard = leaderboard
        self.spinner = spinner
        self.state_stack = []
    def reorder_keyboard_press(self, person_index:int, key:int):
        # Reorder the people
        new_index = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}[key]
        self._reorder(person_index, new_index)
    def _reorder(self, person_index:int, new_index:int) -> None:
        current = self.race_display.get_current()
        person = current.pop(person_index)
        current.insert(new_index, person)
    def next_race(self) -> None:
        last_state = self.get_state()
        people_in_race = self.race_display.get_current()
        if len(people_in_race) == 4:
            points_update = {person:points for person,points in zip(people_in_race, [24,16,8,0])}
        else:
            points_update = {person:points for person,points in zip(people_in_race, [20,12,4])}
        self.leaderboard.update(points_update)
        self.race_display.advance()
        self.state_stack.append(last_state)
    def back(self) -> None:
        if len(self.state_stack) == 0:
            return
        last_state = self.state_stack.pop()
        self.set_state(last_state)
    def clear_back_stack(self) -> None:
        self.state_stack.clear()
    def get_previous_races_string(self) -> str:
        previous_races = self.race_display.races[:self.race_display.index]
        previous_races = [' '.join(str(person) for person in race) for race in previous_races] 
        return '\n'.join(f'- {race}' for race in previous_races)
    def get_state(self) -> dict[str,Any]:
        return {
            'race_display': self.race_display.get_state(),
            'leaderboard': self.leaderboard.get_state(),
        }
    def set_state(self, state:dict[str,Any]):
        self.race_display.set_state(state['race_display'])
        self.leaderboard.set_state(state['leaderboard'])
    