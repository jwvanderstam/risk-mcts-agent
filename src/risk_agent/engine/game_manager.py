import logging
import os
import time

import pygame
import yaml

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.engine.game_engine import GameEngine
from risk_agent.game_elements.action import EndPhaseAction, EndTurnAction
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.basic_evaluation_player import BasicEvaluationPlayer
from risk_agent.players.heuristic_player import BasicHeuristicPlayer
from risk_agent.players.human_player import HumanPlayer
from risk_agent.players.mcts import ConfidentMCTSPlayer, MCTSPlayer
from risk_agent.players.mcts.config import MCTSConfig
from risk_agent.players.player import Player
from risk_agent.players.random_player import RandomPlayer
from risk_agent.rendering.renderer import Renderer
from risk_agent.utils.data_collector import DataCollector

logger = logging.getLogger(__name__)


class GameManager:
    def __init__(self, logging_config: dict) -> None:
        self.number_of_players: int = -1
        self.players: list[Player] = []
        self.game_state: GameState = GameState()
        self.board_file_path: str = ''
        self.max_turns: int = 1000
        self.max_attacking_armies: int = 500
        self.max_defending_armies: int = 500

        self.renderer: Renderer | None = None
        self.clock: pygame.time.Clock
        self.action_delay_ms: int = 1000  # Delay in milliseconds for action delay
        self.running_pygame_loop: bool = True
        self.save_final_game_state: bool = False

        self.logging_config: dict = logging_config

        self.data_collector: DataCollector | None = None

    def run_game(self) -> None:
        """
        Main game loop.
        """
        self.running_pygame_loop = True
        start_time = time.time()

        try:
            self._setup_game()
        except Exception:
            logger.exception('Error setting up game')
            return

        try:
            self._run_game_loop()
        except Exception:
            logger.exception('Error during game loop')
        finally:
            self._finalize_game(start_time)

    def _run_game_loop(self) -> None:
        """
        Repeatedly advance the game until it ends or the pygame window is closed.
        """
        while not self._game_is_over() and self.running_pygame_loop:
            self._handle_pygame_events()
            if not self.running_pygame_loop:
                self.renderer.quit()
                break

            if self.renderer:
                self.renderer.render(game_state=self.game_state)
                self.clock.tick(30)  # Maintain base FPS for event handling and smooth delay
                pygame.time.delay(self.action_delay_ms)

            if self.data_collector:
                # Log the game state at the start of each turn
                if (
                    self.game_state.current_turn_phase == 'reinforce'
                    and self.game_state.reinforcements_this_turn == 0
                ):
                    self.data_collector.collect_turn(
                        turn=self.game_state.current_turn,
                        game_state=self.game_state,
                    )

            self._execute_action_step()

    def _finalize_game(self, start_time: float) -> None:
        """
        Persist collected data, shut down the renderer, and save the final
        game state once the game loop has ended or been interrupted.
        """
        logger.info('Game loop ended or interrupted, saving data.')

        if self.data_collector:
            self.data_collector.collect_game(
                game_state=self.game_state,
                duration=int(time.time() - start_time),
            )
            self.data_collector.save_all()
            logger.info('Data collector saved all data.')

        try:
            if self.renderer:
                self.renderer.quit()
                logger.info('Renderer shut down successfully.')
        except Exception:
            logger.exception('Error shutting down renderer')

        try:
            if self.save_final_game_state:
                if self.game_state_save_path.endswith('.fen'):
                    with open(self.game_state_save_path, 'w') as f:
                        f.write(self.game_state.to_fen())
                else:
                    self.game_state.to_json_file(self.game_state_save_path)
                logger.info(f'Game state saved to {self.game_state_save_path}')
        except Exception:
            logger.exception('Error saving game state')

    def _handle_pygame_events(self) -> None:
        """Hanldle pygame events, primarly for quitting."""
        if not self.renderer:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running_pygame_loop = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running_pygame_loop = False

    def _create_mcts_player(
        self, player_id: int, merged_settings: dict
    ) -> MCTSPlayer | ConfidentMCTSPlayer:
        """
        Build an MCTSConfig from the merged player settings and instantiate
        the corresponding MCTS player (plain or Confident).
        """
        config = MCTSConfig(
            max_attacking_armies=self.max_attacking_armies,
            max_defending_armies=self.max_defending_armies,
            number_of_players=self.number_of_players,
            C=merged_settings.get('C', 1.41),
            search_policy=merged_settings.get('search_policy', 'max^n'),
            playout_policy=merged_settings.get('playout_policy', 'Random'),
            reinforce_all_heuristic=merged_settings.get(
                'reinforce_all_heuristic', False
            ),
            fortify_all_heuristic=merged_settings.get('fortify_all_heuristic', False),
            gamma=merged_settings.get('gamma', 1.0),
            stopping_condition=merged_settings.get('stopping_condition', 'TimeBased'),
            selection_policy=merged_settings.get('selection_policy', 'MaxChild'),
            evaluative_policy=merged_settings.get('evaluative_policy', 'Dummy'),
            think_time=merged_settings.get('think_time', 1.0),
            max_iterations=merged_settings.get('max_iterations', 1000),
            policy_convergence_window=merged_settings.get(
                'policy_convergence_window', 10
            ),
        )

        print(config)

        if config.search_policy == 'Confident' or config.playout_policy == 'Confident':
            print('Creating ConfidentMCTSPlayer')
            return ConfidentMCTSPlayer(
                player_id=player_id,
                config=config,
                battle_computer=self.battle_computer,
                data_collector=self.data_collector,
            )
        return MCTSPlayer(
            player_id=player_id,
            config=config,
            battle_computer=self.battle_computer,
            data_collector=self.data_collector,
        )

    def _setup_players(self) -> None:
        """
        Setup players based on the specified player types.
        This method is called during game initialization.
        """
        default_player_settings = self.settings.get('player_settings', {})

        for i, player_config in enumerate(self.settings['players']):
            player_type = player_config.get('type')
            player_specific_settings = player_config.get('settings', {})

            if player_type == 'human' and not self.rendering_enabled:
                raise ValueError('Human player requires rendering to be enabled.')

            default_settings_for_type = default_player_settings.get(
                player_type, {}
            ).copy()
            merged_settings = {**default_settings_for_type, **player_specific_settings}

            player_instance = None

            if player_type == 'random':
                player_instance = RandomPlayer(
                    player_id=i,
                )
            elif player_type == 'basic_heuristic':
                player_instance = BasicHeuristicPlayer(
                    player_id=i,
                )
            elif player_type == 'basic_evaluation':
                player_instance = BasicEvaluationPlayer(
                    player_id=i,
                    battle_computer=self.battle_computer,
                )
            elif player_type == 'mcts':
                player_instance = self._create_mcts_player(i, merged_settings)
            elif player_type == 'human':
                player_instance = HumanPlayer(
                    player_id=i,
                )
            else:
                raise ValueError(f'Unknown player type: {player_type}')

            self.players.append(player_instance)

        logger.info(f'Initialized players: {self.players}')

    def _setup_game(self) -> None:
        """
        Setup the game by initializing players and game state.
        """
        if self.load_game_state:
            if self.game_state_load_path.endswith('.fen'):
                with open(self.game_state_load_path) as f:
                    self.game_state.from_fen(f.read().strip(), self.board_file_path)
            else:
                self.game_state.from_json_file(
                    self.game_state_load_path, self.board_file_path
                )
        else:
            self.game_state.board.load_from_file(self.board_file_path)
            self.game_state = GameEngine.initialise_random_game_state(
                self.game_state, self.number_of_players
            )

        if not hasattr(self, 'battle_computer'):
            self.battle_computer = BattleComputer(
                max_attacking_armies=self.max_attacking_armies,
                max_defending_armies=self.max_defending_armies,
            )

        self._setup_players()

        # Calculate the initial reinforcements for the first player
        self.game_state.base_reinforcements_this_turn = (
            GameEngine.calculate_base_reinforcements(self.game_state, 0)
        )

        # Notify mcts player(s) of the initial game state
        self._notify_players_of_game_state_update()

        if self.rendering_enabled:
            pygame.init()
            self.renderer = Renderer(
                number_of_players=self.number_of_players,
                board_file_path=self.board_file_path,
            )
        self.clock = pygame.time.Clock()

    def _notify_players_of_game_state_update(self) -> None:
        """
        Notify concerned players of the updated game state.
        This is useful for players that need to react to changes in the game state.
        """
        for i, player in enumerate(self.players):
            if (
                isinstance(player, MCTSPlayer)
                and i not in self.game_state.defeated_players
            ):
                player.notify_game_state_update(
                    GameEngine.determinize_game_state(self.game_state.copy(), i)
                )

    def _execute_action_step(self) -> None:
        """
        Play a turn for the current player.
        """
        current_player_id = self.game_state.current_player
        current_player_obj = self.players[current_player_id]

        logger.info(
            f'Current player: {current_player_id}, '
            f'Current round: {self.game_state.current_round}, '
            f'Current turn: {self.game_state.current_turn}, '
            f'Current phase: {self.game_state.current_turn_phase}'
        )

        if current_player_id in self.game_state.defeated_players:
            logger.info(
                f'Player {current_player_id} has been defeated and cannot take actions.'
            )
            self.game_state = GameEngine.apply_action(self.game_state, EndTurnAction())
            return

        decision_game_state = GameEngine.determinize_game_state(
            self.game_state.copy(), current_player_id
        )
        valid_actions = GameEngine.get_valid_actions(decision_game_state)

        logger.info(
            f'Valid actions for player {current_player_id}: {len(valid_actions)}'
        )

        if not valid_actions:
            logger.warning(
                f'Player {current_player_id} has no valid actions. Forcing phase/turn end.'
            )
            if self.game_state.current_turn_phase == 'fortify':
                chosen_action = EndTurnAction()
            else:
                chosen_action = EndPhaseAction()
        else:
            chosen_action = current_player_obj.decide_action(
                decision_game_state, valid_actions
            )
            # Verify that the chosen action is valid
            if chosen_action not in valid_actions:
                raise ValueError(
                    f'Chosen action {chosen_action} is not in valid actions: {valid_actions}'
                )
        logger.info(f'Player {current_player_id} chose action: {chosen_action}')

        if self.data_collector:
            self.data_collector.collect_action(
                turn=self.game_state.current_turn,
                player_id=current_player_id,
                action=chosen_action,
            )

        self.game_state = GameEngine.apply_action(
            self.game_state, chosen_action, self.battle_computer
        )

        self._notify_players_of_game_state_update()

    def _game_is_over(self) -> bool:
        """Determines if the game has ended."""
        if self.game_state.current_turn >= self.max_turns:
            logger.info(f'Game over: Maximum turns ({self.max_turns}) reached.')
            return True
        if self._check_for_winner():
            logger.info(
                f'Game over: Player {self.game_state.current_player} has won '
                f'in turn {self.game_state.current_turn}.'
            )
            return True
        return False

    def _check_for_winner(self) -> bool:
        """
        Check if there is a winner in the game.
        A player wins if they own all territories.
        """
        return self.game_state.is_terminal()

    def load_settings(self, settings_path: str = 'data/settings.yaml') -> None:
        """
        Read game settings from a configuration file.
        """
        try:
            with open(settings_path) as file:
                self.settings = yaml.safe_load(file)
                self.process_settings()
        except FileNotFoundError:
            logger.error(f'Settings file not found: {settings_path}')
        except yaml.YAMLError:
            logger.exception('Error reading settings file')
        except Exception:
            logger.exception('Unexpected error loading settings')

    def process_settings(self) -> None:
        """
        Process the loaded settings to initialize game parameters.
        This method should be called after load_settings.
        """
        if not self.settings:
            raise ValueError('Settings have not been loaded. Call load_settings first.')

        self.number_of_players = self.settings['game_settings']['number_of_players']

        # Validate the number of players
        if self.number_of_players < 0:
            raise ValueError('Number of players must be set.')
        if self.number_of_players < 2:
            raise ValueError('Number of players must be at least 2.')
        if self.number_of_players != len(self.settings['players']):
            raise ValueError(
                'Number of players must match the number of player types specified.'
            )

        self.max_turns = self.settings.get('game_settings').get('max_turns', 1000)
        self.max_attacking_armies = self.settings.get('game_settings').get(
            'max_attacking_armies', 100
        )
        self.max_defending_armies = self.settings.get('game_settings').get(
            'max_defending_armies', 100
        )
        self.load_game_state = self.settings.get('game_settings').get(
            'load_game_state', False
        )

        self.board_file_path = self.settings.get('paths').get(
            'board_file', './data/standard_board.json'
        )

        if self.board_file_path == '':
            raise ValueError('Board file path must be set.')
        if not os.path.exists(self.board_file_path):
            raise FileNotFoundError(f'Board file not found: {self.board_file_path}')

        self.game_state_load_path = self.settings.get('paths').get(
            'game_state_load_file', './data/game_state.json'
        )
        self.game_state_save_path = self.settings.get('paths').get(
            'game_state_save_file', './data/game_state.json'
        )
        self.save_final_game_state = self.settings.get('game_settings').get(
            'save_final_game_state', False
        )
        self.rendering_enabled = (
            self.settings.get('environment_settings')
            .get('rendering')
            .get('enabled', True)
        )
        self.action_delay_ms = (
            self.settings.get('environment_settings')
            .get('rendering')
            .get('action_delay_ms', 1000)
        )
