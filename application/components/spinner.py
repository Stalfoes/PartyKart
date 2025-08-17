import jax
import jax.numpy as jnp
from typing import Any
import pygame


ACTIVITIES_FILE = '/Users/kapeluck/Documents/Personal/RonyPartyKart/application/activities/activities.txt'


class Spinner:
    def __init__(self, rng:Any, num_races:int):
        self.num_races = num_races
        self.activities, self.modifiers = self._load()
        self.sequence = self._generate_sequence(rng, num_races)
        self.play_timer:int = None
        self.current_idx:int = None
    def _load(self) -> tuple[list[str], list[str]]:
        activities = []
        modifiers = []
        with open(ACTIVITIES_FILE, 'r') as file:
            filling_activities = True
            for raw_line in file.readline():
                line = raw_line.strip()
                if line.startswith('#'):
                    continue
                if len(line) == 0:
                    if filling_activities == False:
                        raise ValueError(f"I don't know what to do with an extra blank line. Exiting loading the activities file in Spinner.")
                    filling_activities = False; continue
                if filling_activities:
                    activities.append(line)
                else:
                    modifiers.append(line)
        return activities, modifiers
    def _generate_sequence(self, rng:Any, length:int) -> list[tuple[str,str]]:
        shuffled_activities = []
        for i in range(jnp.ceil(length / len(self.activities))):
            rng, activities_rng = jax.random.split(rng)
            activities_indices = jax.random.permutation(activities_rng, jnp.arange(len(self.activities)), independent=True)
            shuffled_activities += [self.activities[idx] for idx in activities_indices]
        shuffled_modifiers = []
        for i in range(jnp.ceil(length / len(self.modifiers))):
            rng, modifiers_rng = jax.random.split(rng)
            modifiers_indices = jax.random.permutation(modifiers_rng, jnp.arange(len(self.modifiers)), independent=True)
            shuffled_modifiers += [self.modifiers[idx] for idx in modifiers_indices]
        return [(activity,modifier) for activity,modifier in zip(shuffled_activities, shuffled_modifiers)]
    def __getitem__(self, idx:int) -> tuple[str,str]:
        if self.sequence is None:
            raise ValueError(f"Sequence is uninialized.")
        return self.sequence[idx]
    def play(self, idx:int) -> None:
        self.play_timer = 0
        self.current_idx = idx
    def hide(self) -> None:
        self.play_timer = None
        self.current_idx = None
    def as_surface(self, size:tuple[int,int]) -> tuple[pygame.Surface, bool]:
        text_size = 20
        surface = pygame.Surface(size, pygame.SRCALPHA)
        font = pygame.font.Font(None, text_size)
        if self.play_timer is None:
            return surface, False
        