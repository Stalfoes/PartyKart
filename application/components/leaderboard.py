import pygame
from collections import defaultdict
from typing import Any
from copy import deepcopy

from . import graphics_helper as graphics


COLOURS = defaultdict(lambda: (67, 139, 181))
COLOURS.update({
    0: (255, 212, 69),
    1: (219, 219, 217),
    2: (199, 138, 54),
})

class Leaderboard:
    def __init__(self, roster:list[str], people_that_left:set[str]=None):
        self.scores = {person:0 for person in roster}
        self.games_played = {person:0 for person in roster}
        self.people_that_left = set() if people_that_left is None else people_that_left
    @property
    def num_people(self):
        return len(self.scores)
    def sorted_people(self) -> list[tuple[str,int,int]]:
        people = sorted(self.scores.keys(), reverse=True, key=lambda p:(self.scores[p], -self.games_played[p]))
        return [(person,self.scores[person],self.games_played[person]) for person in people]
    def update(self, additions:dict[str,int]) -> None:
        for person,increase in additions.items():
            self.scores[person] += increase
            self.games_played[person] += 1
    def as_surface(self, size:tuple[int,int]) -> tuple[pygame.Surface, bool]:
        # Get sizes and calculations
        # text_size = int(0.8 * min(size[0], size[1]) // self.num_people)
        text_size = int(0.9 * min(size[0], size[1]) // self.num_people)
        rect_height = int(0.9 * (size[1] // (self.num_people + 1)))
        y_little_spacer = rect_height * 0.22
        x_little_spacer = text_size * 0.25
        # Create surface
        surface = pygame.Surface(size, pygame.SRCALPHA)
        # Display leaderboard title
        font = pygame.font.Font('/Users/kapeluck/Documents/Personal/RonyPartyKart/application/fonts/MarioNett.ttf', int(text_size * 1.5))
        # text = font.render("LEADERBOARD", True, (10,10,10))
        text = graphics.outline_text_render("LEADERBOARD", font, (36, 175, 255), (10,10,10), 5)
        textpos = text.get_rect()
        textpos.centerx = surface.get_rect().centerx
        surface.blit(text, textpos)
        # Plot the actual leaderboard
        font = pygame.font.Font('/Users/kapeluck/Documents/Personal/RonyPartyKart/application/fonts/MarioNett.ttf', text_size)
        position_font = pygame.font.Font('/Users/kapeluck/Documents/Personal/RonyPartyKart/application/fonts/MarioNett.ttf', int(text_size * 1.3))
        sorted_scores = self.sorted_people()
        for pos,(person,score,games_played) in enumerate(sorted_scores):
            # draw the rectangle around the name
            y_pos = (pos + 1) * (size[1] // (self.num_people + 1))
            rect = pygame.Rect(x_little_spacer, y_pos, size[0]*0.95, rect_height)
            pygame.draw.rect(surface, COLOURS[pos], rect, border_radius=20)
            pygame.draw.rect(surface, (0,0,0), rect, width=2, border_radius=20)
            # draw the text on the rectangle
            # draw the position number
            interpolated_color = graphics.darken(COLOURS[pos], 0.20)
            # interpolated_color = (255,255,255)
            # text = position_font.render(f"#{pos + 1}", True, interpolated_color)
            text = graphics.outline_text_render(f"#{pos + 1}", position_font, interpolated_color, (10,10,10), 2)
            text = graphics.outline_text_render(f"{pos + 1}", position_font, interpolated_color, (10,10,10), 2)
            textpos = text.get_rect()
            textpos.topleft = (0, (pos + 1) * (size[1] // (self.num_people + 1)))
            textpos.topleft = (x_little_spacer * 2, (pos + 1) * (size[1] // (self.num_people + 1)))
            textpos.centery = rect.centery
            # textpos.topleft = (x_little_spacer * 1.5, y_little_spacer + (pos + 1) * (size[1] // (self.num_people + 1)))
            # text = pygame.transform.rotate(text, 15)
            surface.blit(text, textpos)
            # draw the score
            # text = font.render(f"{score}", True, interpolated_color)
            # textpos = text.get_rect()
            # textpos.topright = (text_size*2.50, y_little_spacer + (pos + 1) * (size[1] // (self.num_people + 1)))
            # surface.blit(text, textpos)
            # draw the name
            text = font.render(f"{person}", True, (10, 10, 10))
            # surface.blit(text, (text_size*3.25, y_little_spacer + (pos + 1) * (size[1] // (self.num_people + 1))))
            textrect = text.get_rect()
            textrect.topleft = (text_size*2.00, y_little_spacer + (pos + 1) * (size[1] // (self.num_people + 1)))
            surface.blit(text, textrect)
            
            # if the person left, cross their name out
            if person in self.people_that_left:
                pygame.draw.line(surface, (255,0,0), (textrect.left - text_size*0.1, textrect.centery), (textrect.right + text_size*0.1, textrect.centery), width=5)
            
            # draw the games played
            # text = font.render(f"{games_played} {'game' if games_played == 1 else 'games'}", True, interpolated_color)
            # text = font.render(f"{score} / {games_played} {'game' if games_played == 1 else 'games'}", True, interpolated_color)
            interpolated_color = graphics.darken(COLOURS[pos], 0.38)
            text = font.render(f"{score} / {games_played} ", True, interpolated_color)
            textpos = text.get_rect()
            textpos.topright = (size[0]*0.95 - text_size*0.25, y_little_spacer + (pos + 1) * (size[1] // (self.num_people + 1)))
            surface.blit(text, textpos)
        return surface, False
    def get_state(self) -> dict[str,Any]:
        return {
            'scores': deepcopy(self.scores),
            'games_played': deepcopy(self.games_played),
        }
    def set_state(self, state:dict[str,Any]):
        self.scores = state['scores']
        self.games_played = state['games_played']
