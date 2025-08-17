import pygame
from pygame.locals import QUIT
import jax
import jax.numpy as jnp
import pickle
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def main():
    # Initialise screen
    pygame.init()
    screen = pygame.display.set_mode((1280, 960), pygame.RESIZABLE)
    pygame.display.set_caption("Luke's Beerio Kart Program")

    try:
        background_image = pygame.image.load('/Users/kapeluck/Documents/Personal/RonyPartyKart/application/images/background.png').convert() # Use .convert_alpha() if image has transparency
    except pygame.error as e:
        print(f"Error loading image: {e}")
        pygame.quit(); exit()

    from components.leaderboard import Leaderboard
    from components.next_race import RaceDisplay
    from components.controller import Controller
    from components.spinner import Spinner

    # Load the generated races
    with open('generated_races.pckl', 'rb') as file:
        brackets = pickle.load(file)
    roster = set()
    for race in brackets:
        for person in race:
            roster.add(person)
    # Load the config to get remaining information
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader)
    people_that_left = config['people_that_left']
    if people_that_left is None:
        people_that_left = set()
    previous_races = config['previous_races']
    if previous_races is None:
        previous_races = []
    starting_race_index = len(previous_races)

    leaderboard = Leaderboard(roster, people_that_left)
    # brackets = [['Jensen', 'Rony', 'Anton', 'Michael'], ['Eric', 'Anton', 'Luke', 'Paul'], ['Alek', 'Eric', 'Anton', 'Michael'], ['Eric', 'Jensen', 'Matei', 'Alek'], ['Anton', 'Matei', 'Michael', 'Eric'], ['Jensen', 'Matei', 'Michael', 'Luke'], ['Luke', 'Alek', 'Rony'], ['Matei', 'Eric', 'Rony', 'Jensen'], ['Luke', 'Paul', 'Jensen', 'Michael'], ['Anton', 'Matei', 'Alek', 'Paul'], ['Alek', 'Michael', 'Paul', 'Rony'], ['Eric', 'Paul', 'Jensen', 'Rony'], ['Rony', 'Matei', 'Anton', 'Luke'], ['Jensen', 'Paul', 'Anton', 'Alek'], ['Paul', 'Matei', 'Michael', 'Luke'], ['Luke', 'Alek', 'Eric', 'Rony']]
    race_display = RaceDisplay(brackets)
    # SEED = 0
    # spinner = Spinner(jax.random.key(SEED), len(brackets))
    spinner = None
    # leaderboard.update({
    #     'rony' : 160,
    #     'luke': 24,
    #     'paul': 8,
    #     'benson': 0,
    # })

    controller = Controller(race_display, leaderboard, spinner)

    # There were already races that happened, roll the controller forward until we've gotten to that point
    for idx in range(starting_race_index):
        controller.next_race()

    background_image = pygame.transform.scale(background_image, screen.get_size())
    leaderboard_surface, _ = controller.leaderboard.as_surface(screen.get_size())
    race_surface, _ = controller.race_display.as_surface(screen.get_size())
    # background_image.blit(race_surface, (0,0))
    # background_image.blit(leaderboard_surface, (0, 0))
    screen.blit(background_image, (0, 0))
    screen.blit(race_surface, (0,0))
    screen.blit(leaderboard_surface, (0,0))
    # pygame.display.flip()
    pygame.display.update()

    # Event loop
    while True:
        event_ocurred = False
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == pygame.VIDEORESIZE:
                # Update the screen size to the new dimensions
                event_ocurred = True
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                screen.fill((0, 0, 0))
            elif event.type == pygame.KEYUP:
                # get the key
                if event.key == pygame.K_SPACE:
                    event_ocurred = True
                    controller.next_race()
                elif event.key in {pygame.K_BACKSPACE, pygame.K_DELETE}:
                    event_ocurred = True
                    controller.back()
                elif event.key in {pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4}:
                    race_surface_pos = race_surface.get_rect()
                    race_surface_pos.left = 0
                    race_surface_pos.centery = screen.get_rect().centery
                    mouse_over_index = controller.race_display.get_mouse_over_player_index(pygame.mouse.get_pos(), race_surface_pos.topleft, (screen.get_size()[0] * 0.5, screen.get_size()[1] * 0.75))
                    if mouse_over_index is not None:
                        event_ocurred = True
                        controller.reorder_keyboard_press(mouse_over_index, event.key)
                elif event.key == pygame.K_s:
                    with open('save_state.pckl', 'wb') as file:
                        pickle.dump(controller.get_state(), file)
                    previous_races_string = controller.get_previous_races_string()
                    print('-------------------------')
                    print(previous_races_string)
                    print('-------------------------')
                elif event.key == pygame.K_l:
                    with open('save_state.pckl', 'rb') as file:
                        state = pickle.load(file)
                        controller.set_state(state)
                        event_ocurred = True
                elif event.key == pygame.K_r:
                    # We want a reload/refresh of the races
                    # event_ocurred = True
                    pass
                    # controller.clear_back_stack()
                    # from RonyPartyKart import run
        if event_ocurred:
            anything_changing = False
            screen.fill((0,0,0))
            background_image = pygame.transform.scale(background_image, screen.get_size())
            leaderboard_surface, changing = controller.leaderboard.as_surface((screen.get_size()[0] * 0.5, screen.get_size()[1] * 0.9))
            anything_changing = anything_changing or changing
            race_surface, changing = controller.race_display.as_surface((screen.get_size()[0] * 0.5, screen.get_size()[1] * 0.75))
            anything_changing = anything_changing or changing
            screen.blit(background_image, (0,0))
            race_surface_pos = race_surface.get_rect()
            race_surface_pos.left = 0
            race_surface_pos.centery = screen.get_rect().centery
            leaderboard_surface_pos = leaderboard_surface.get_rect()
            leaderboard_surface_pos.left = screen.get_size()[0] * 0.5
            leaderboard_surface_pos.centery = screen.get_rect().centery
            screen.blit(race_surface, race_surface_pos)
            screen.blit(leaderboard_surface, leaderboard_surface_pos)
            # pygame.display.flip()
            pygame.display.update()
            event_ocurred = anything_changing


if __name__ == '__main__':
    main()