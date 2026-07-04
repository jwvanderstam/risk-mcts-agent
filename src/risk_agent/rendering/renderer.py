import json
import logging
import os
import sys

import pygame

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(
        self,
        number_of_players: int,
        board_file_path: str,
        map_width: int = 1150,
        height: int = 800,
        sidebar_width: int = 300,
    ) -> None:
        pygame.init()
        self.map_width = map_width
        self.height = height
        self.sidebar_width = sidebar_width
        self.total_width = map_width + sidebar_width

        self.screen = pygame.display.set_mode((self.total_width, self.height))
        pygame.display.set_caption('Risk Game Renderer')

        self.map_rect = pygame.Rect(0, 0, self.map_width, self.height)
        self.sidebar_rect = pygame.Rect(
            self.map_width, 0, self.sidebar_width, self.height
        )

        self.font_territory_details = pygame.font.SysFont(None, 20)
        self.font_territory_id = pygame.font.SysFont(None, 12)
        self.sidebar_font_title = pygame.font.SysFont(None, 36)
        self.sidebar_font_text = pygame.font.SysFont(None, 28)
        self.sidebar_font_player_list = pygame.font.SysFont(None, 24)
        self.sidebar_font_card_list = pygame.font.SysFont(None, 18)

        self.number_of_players = number_of_players
        self.player_colors = self._define_player_colors()

        self.territory_positions = {}
        self.continent_positions = {}

        # Load territory and continent positions from a JSON file
        self._load_positions_from_file(board_file_path)

    def _define_player_colors(self) -> dict:
        colors = [
            (255, 100, 100),  # Light Red
            (100, 100, 255),  # Light Blue
            (100, 255, 100),  # Light Green
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (200, 200, 200),  # Light Grey
            (255, 165, 0),  # Orange
        ]

        return {i: colors[i % len(colors)] for i in range(self.number_of_players)}

    def _load_positions_from_file(self, file_path: str) -> None:
        """
        Load territory positions from a JSON file.
        """
        with open(file_path) as f:
            data = json.load(f)

        self.territory_positions = data.get('territory_positions', {})
        self.continent_positions = data.get('continent_positions', {})

        # Change the keys to integers
        self.territory_positions = {
            int(k): v for k, v in self.territory_positions.items()
        }

        if not self.territory_positions:
            logger.warning(f"No territory positions found in '{file_path}'.")
        if not self.continent_positions:
            logger.warning(f"No continent positions found in '{file_path}'.")

    def _draw_wraparound_connection(
        self, line_color: tuple, pos1: tuple, pos2: tuple
    ) -> None:
        """Draw the Alaska(1)-Kamchatka(35) connection as two edge-wrapping segments."""
        pygame.draw.line(self.screen, line_color, (pos1[0], pos1[1]), (0, pos1[1]), 2)
        pygame.draw.line(
            self.screen,
            line_color,
            (pos2[0], pos2[1]),
            (self.map_width, pos2[1]),
            2,
        )

    def _draw_territory_connections(
        self,
        terr_id: int,
        pos1: tuple,
        adjacencies: list,
        line_color: tuple,
        drawn_connections: set,
    ) -> None:
        """Draw every not-yet-drawn connection from one territory to its neighbors."""
        for adj_id in adjacencies:
            adj_id_int = int(adj_id)
            connection_pair = tuple(sorted((terr_id, adj_id_int)))
            if connection_pair in drawn_connections:
                continue

            pos2 = self.territory_positions.get(adj_id_int)
            if not pos2:
                continue

            # Handle Alaska-Kamchatka wrap-around line TODO: make this more generic
            if (terr_id == 1 and adj_id_int == 35) or (
                terr_id == 35 and adj_id_int == 1
            ):
                self._draw_wraparound_connection(line_color, pos1, pos2)
            else:
                pygame.draw.line(
                    self.screen, line_color, pos1, pos2, 2
                )  # Thickness 2
            drawn_connections.add(connection_pair)

    def _draw_connections(self, game_state: 'GameState') -> None:
        line_color = (120, 120, 120)  # Medium-grey lines
        drawn_connections = set()

        if not game_state.board.adjacency_list:
            return

        for terr_id_str, adjacencies in game_state.board.adjacency_list.items():
            terr_id = int(terr_id_str)
            pos1 = self.territory_positions.get(terr_id)

            if not pos1:
                continue

            self._draw_territory_connections(
                terr_id, pos1, adjacencies, line_color, drawn_connections
            )

    def _draw_territories(self, game_state: 'GameState') -> None:
        territory_radius = 20
        text_color = (0, 0, 0)  # Black text for army count

        if not game_state.board.territories:
            return

        for terr_id_str, terr_data in game_state.board.territories.items():
            terr_id = int(terr_id_str)
            pos = self.territory_positions.get(terr_id)

            if not pos:
                logger.warning(
                    f"Skipping territory ID {terr_id} ('{terr_data['name']}') due to missing position."  # noqa: E501
                )
                continue

            owner = (
                game_state.owner[terr_id]
                if terr_id < len(game_state.owner) and game_state.owner[terr_id] != -1
                else None
            )
            armies = (
                game_state.armies[terr_id] if terr_id < len(game_state.armies) else 0
            )

            if owner is None or owner not in self.player_colors:
                fill_color = (180, 180, 180)  # Default grey for unowned
            else:
                fill_color = self.player_colors[owner]

            # Draw territory circle
            pygame.draw.circle(self.screen, fill_color, pos, territory_radius)
            pygame.draw.circle(
                self.screen, (50, 50, 50), pos, territory_radius, 2
            )  # Darker border

            # Draw army count
            army_text_surface = self.font_territory_details.render(
                str(armies), True, text_color
            )
            army_text_rect = army_text_surface.get_rect(center=pos)
            self.screen.blit(army_text_surface, army_text_rect)

            # Draw territory ID below the circle for reference
            id_text_surface = self.font_territory_id.render(
                str(terr_id), True, (80, 80, 80)
            )
            id_text_rect = id_text_surface.get_rect(
                center=(pos[0] + territory_radius - 3, pos[1] + territory_radius + 6)
            )
            self.screen.blit(id_text_surface, id_text_rect)

    def _draw_continents(self) -> None:
        line_color = (0, 0, 0)  # Black lines for continent borders

        for continent, lines in self.continent_positions.items():
            for line in lines:
                pygame.draw.line(self.screen, line_color, line[0], line[1], 2)

    def _draw_sidebar(self, game_state: 'GameState') -> None:
        pygame.draw.rect(
            self.screen, (50, 50, 60), self.sidebar_rect
        )  # Darker sidebar background

        padding = 20
        current_y = padding

        # Title
        title_surface = self.sidebar_font_title.render(
            'Game Info', True, (220, 220, 220)
        )
        title_rect = title_surface.get_rect(
            centerx=self.sidebar_rect.centerx, top=current_y
        )
        self.screen.blit(title_surface, title_rect)
        current_y += title_rect.height + 20

        # Turn Info
        turn_text = f'Turn: {game_state.current_turn + 1}'  # Assuming turn is 0-indexed
        turn_surface = self.sidebar_font_text.render(turn_text, True, (200, 200, 200))
        self.screen.blit(turn_surface, (self.sidebar_rect.left + padding, current_y))
        current_y += turn_surface.get_height() + 10

        # Phase Info
        phase_text = f'Phase: {game_state.current_turn_phase.capitalize()}'
        phase_surface = self.sidebar_font_text.render(phase_text, True, (200, 200, 200))
        self.screen.blit(phase_surface, (self.sidebar_rect.left + padding, current_y))
        current_y += phase_surface.get_height() + 20

        # Current Player Info
        cp_text = f'Current Player: Player {game_state.current_player}'
        cp_color = self.player_colors.get(game_state.current_player, (255, 255, 255))
        cp_surface = self.sidebar_font_text.render(cp_text, True, cp_color)
        self.screen.blit(cp_surface, (self.sidebar_rect.left + padding, current_y))
        current_y += cp_surface.get_height() + 20

        # Player List Title
        players_title_surface = self.sidebar_font_text.render(
            'Players:', True, (200, 200, 200)
        )
        self.screen.blit(
            players_title_surface, (self.sidebar_rect.left + padding, current_y)
        )
        current_y += players_title_surface.get_height() + 10

        # Player List
        player_box_height = 25
        for i in range(game_state.number_of_players):
            player_name = f'Player {i}'
            player_color = self.player_colors.get(i, (200, 200, 200))

            # Highlight current player
            if i == game_state.current_player:
                player_name += ' (Current)'

            player_surface = self.sidebar_font_player_list.render(
                player_name, True, (220, 220, 220)
            )  # Player name next to swatch

            # Color swatch
            swatch_rect = pygame.Rect(
                self.sidebar_rect.left + padding, current_y, 20, player_box_height - 5
            )
            pygame.draw.rect(self.screen, player_color, swatch_rect)

            self.screen.blit(
                player_surface, (self.sidebar_rect.left + padding + 30, current_y)
            )
            current_y += player_box_height + 5

        current_y += 10  # Extra space after player list

        # Cards List Title
        cards_title_surface = self.sidebar_font_text.render(
            'Cards:', True, (200, 200, 200)
        )
        self.screen.blit(
            cards_title_surface, (self.sidebar_rect.left + padding, current_y)
        )
        current_y += cards_title_surface.get_height() + 10

        # Cards List
        for player_id, hand in game_state.player_hands.items():
            player_name = f'Player {player_id} Cards:'
            player_surface = self.sidebar_font_card_list.render(
                player_name, True, (220, 220, 220)
            )
            self.screen.blit(
                player_surface, (self.sidebar_rect.left + padding, current_y)
            )
            current_y += player_surface.get_height() + 5

            for card in hand:
                card_text = f'{card[0]} {card[1]} ({card[2]})'
                card_surface = self.sidebar_font_card_list.render(
                    card_text, True, (200, 200, 200)
                )
                self.screen.blit(
                    card_surface,
                    (self.sidebar_rect.left + padding + 20, current_y),
                )
                current_y += card_surface.get_height() + 2

    def render(self, game_state: 'GameState') -> None:
        # Fill background
        self.screen.fill((220, 220, 220))  # Light grey background

        self._draw_connections(game_state=game_state)
        self._draw_territories(game_state=game_state)
        self._draw_continents()
        self._draw_sidebar(game_state=game_state)

        pygame.display.flip()

    def quit(self) -> None:
        # save a screenshot
        pygame.image.save(self.screen, 'screenshot.png')
        logger.info('Screenshot saved as screenshot.png')

        # Quit pygame
        pygame.quit()


if __name__ == '__main__':
    from engine.game_engine import GameEngine
    from game_elements.game_state import GameState

    current_game_state = GameState()
    board_file_path = '../src/data/standard_board.json'

    try:
        current_game_state.board.load_from_file(board_file_path)
        logger.info(f'Successfully loaded board from: {board_file_path}')
        if not current_game_state.board.territories:
            logger.warning(
                f'No territories found in the board file at {board_file_path}.'
            )
            exit()
    except FileNotFoundError:
        logger.error(f'Board file not found at "{board_file_path}"')
        logger.error(
            'Please ensure the path is correct relative to your execution directory.'
        )
        logger.error(f'Current working directory: {__import__("os").getcwd()}')
        exit()
    except Exception:
        logger.exception('Unexpected error loading board')
        exit()

    num_testing_players = 4
    current_game_state = GameEngine.initialise_random_game_state(
        current_game_state, num_testing_players
    )
    logger.info(
        f'Game state initialized for {current_game_state.number_of_players} players.'
    )

    try:
        risk_renderer = Renderer(
            number_of_players=num_testing_players,
            board_file_path=board_file_path,
            map_width=1280,
            height=800,
            sidebar_width=300,
        )
    except ValueError:
        logger.exception('Error initializing renderer')
        exit()

    logger.info('Starting render loop...')

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        risk_renderer.render(game_state=current_game_state)
        clock.tick(30)  # Limit to 30 FPS

    risk_renderer.quit()
    logger.info('Renderer quit.')
