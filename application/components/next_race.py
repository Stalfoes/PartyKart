import pygame
from typing import Any, Optional
from copy import deepcopy

from . import graphics_helper as graphics

COLOUR = (67, 139, 181)

class RaceDisplay:
    def __init__(self, races:list[list[str]]):
        self.races = races
        self.index = 0
    def advance(self) -> None:
        if self.index >= len(self.races):
            return
        self.index += 1
    def back(self) -> None:
        if self.index <= -1:
            return
        self.index -= 1
    def get_current(self) -> list[str]:
        return self.races[self.index]
    def get_mouse_over_player_index(self, mouse_position:tuple[int,int], our_position:tuple[int,int], size:tuple[int,int]) -> Optional[int]:
        position = (mouse_position[0] - our_position[0], mouse_position[1] - our_position[1])
        # Check if the mouse is even over us and our space at all
        if position[0] < 0 or position[1] < 0 or position[0] >= size[0] or position[1] >= size[1]:
            # If not, move on
            return None
        # Check if we're even displaying a race
        if self.index < 0 or self.index >= len(self.races):
            return None
        # If yes, then check the squares
        num_people = len(self.races[self.index]) if self.index > 0 and self.index < len(self.races) else 4
        rect_height = int(0.9 * (size[1] // (4 + 1)))
        for idx,person in enumerate(self.races[self.index]):
            # draw the rectangle around the name
            y_pos = (idx + 1) * (size[1] // (num_people + 1))
            rect = pygame.Rect(0, y_pos, size[0]*0.95, rect_height)
            if rect.collidepoint(position):
                return idx
        return None
    def as_surface(self, size:tuple[int,int]) -> tuple[pygame.Surface, bool]:
        # Get sizes and calculations
        num_people = len(self.races[self.index]) if self.index > 0 and self.index < len(self.races) else 4
        text_size = int(0.9 * min(size[0], size[1]) // 4)
        rect_height = int(0.9 * (size[1] // (4 + 1)))
        little_spacer = rect_height * 0.22
        # Create surface
        surface = pygame.Surface(size, pygame.SRCALPHA)
        font = pygame.font.Font('/Users/kapeluck/Documents/Personal/RonyPartyKart/application/fonts/MarioNett.ttf', text_size)
        # text = font.render("Current Race", True, (10, 10, 10))
        text = graphics.outline_text_render("Current Race", font, (36, 175, 255), (10,10,10), 5)
        textpos = text.get_rect()
        textpos.centerx = surface.get_rect().centerx
        surface.blit(text, textpos)
        if self.index < 0 or self.index >= len(self.races):
            return surface, False
        for idx,person in enumerate(self.races[self.index]):
            # draw the rectangle around the name
            y_pos = (idx + 1) * (size[1] // (num_people + 1))
            rect = pygame.Rect(0, y_pos, size[0]*0.95, rect_height)
            pygame.draw.rect(surface, COLOUR, rect, border_radius=20)
            pygame.draw.rect(surface, (0,0,0), rect, width=4, border_radius=20)
            # draw the text on the rectangle
            # draw the name
            # text = font.render(f"{person}", True, (10, 10, 10))
            text = graphics.outline_text_render(f"{person}", font, (255,255,255), (10,10,10), 3)
            textpos = text.get_rect()
            textpos.centerx = rect.centerx
            surface.blit(text, (textpos.x, little_spacer + (idx + 1) * (size[1] // (num_people + 1))))
        return surface, False
    def get_state(self) -> dict[str,Any]:
        return {
            'races': deepcopy(self.races),
            'index': self.index,
        }
    def set_state(self, state:dict[str,Any]) -> None:
        self.races = state['races']
        self.index = state['index']