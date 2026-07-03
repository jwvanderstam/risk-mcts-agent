import logging
import os
from collections import Counter

import pandas as pd

from risk_agent.game_elements.action import Action
from risk_agent.game_elements.game_state import GameState

logger = logging.getLogger(__name__)


class DataCollector:
    """
    A class to log game metrics to a CSV file.
    """

    def __init__(
        self,
        experiment_path: str,
        number_of_players: int,
        player_types: list[str],
        max_turns: int,
        max_attacking_armies: int,
        max_defending_armies: int,
    ) -> None:
        self.experiment_path: str = experiment_path
        self.number_of_players: int = number_of_players
        self.player_types: list[str] = player_types
        self.max_turns: int = max_turns
        self.max_attacking_armies: int = max_attacking_armies
        self.max_defending_armies: int = max_defending_armies
        self.game_id: int = -1  # Unique identifier for the game

        self.action_data: list[dict] = []
        self.turn_data: list[dict] = []
        self.game_data: list[dict] = []
        self.player_data: list[list[dict]] = [[] for _ in range(self.number_of_players)]

        # Ensure the output directory exists
        self._ensure_paths_exist()

    def collect_player(self, player_id: int, data: dict) -> None:
        """Records player-level metrics."""
        self.player_data[player_id].append(data)

    def collect_action(self, turn: int, player_id: int, action: 'Action') -> None:
        """Records an action-level event."""
        self.action_data.append(
            {
                'game_id': self.game_id,
                'turn': turn,
                'player_id': player_id,
                'player_type': self.player_types[player_id],
                'action_type': action.__class__.__name__,
            }
        )

    def collect_turn(self, turn: int, game_state: 'GameState') -> None:
        """Records a snapshot of the game state at the end of a turn."""
        turn_record = {
            'game_id': self.game_id,
            'turn': turn,
            'active_player_id': game_state.current_player,
        }
        # Add territory and army counts for each player
        territory_counts = Counter(
            game_state.owner[t] for t in game_state.territory_ids
        )
        army_counts = Counter()
        for territory_id in game_state.territory_ids:
            army_counts[game_state.owner[territory_id]] += game_state.armies[
                territory_id
            ]

        for i in range(self.number_of_players):
            turn_record[f'player_{i}_armies'] = army_counts.get(i, 0)
            turn_record[f'player_{i}_territories'] = territory_counts.get(i, 0)
        self.turn_data.append(turn_record)

    def collect_game(
        self,
        game_state: 'GameState',
        duration: float = 0,
    ) -> None:
        """Records the final outcome of a single game."""
        game_record = {
            'game_id': str(self.game_id),
            'duration': duration,
        }

        # Determine the winner and reason for ending
        winner = -1

        active_players = [
            p
            for p in range(self.number_of_players)
            if p not in game_state.defeated_players
        ]
        if len(active_players) == 1:
            winner = active_players[0]
            end_reason = 'Conquest'
        elif game_state.current_turn >= self.max_turns:
            end_reason = 'Max Turns Reached'
        elif any(
            armies >= self.max_attacking_armies or armies > self.max_defending_armies
            for armies in game_state.armies[1:]
        ):
            end_reason = 'Max armies exceeded'
        else:
            end_reason = 'Unknown/Error'

        winner_type = self.player_types[winner] if winner >= 0 else 'None'

        game_record['winner'] = str(winner)
        game_record['winner_type'] = winner_type
        game_record['end_reason'] = end_reason
        game_record['total_rounds'] = str(game_state.current_round)
        game_record['total_turns'] = str(game_state.current_turn)

        # Add player types and starting positions
        for i in range(self.number_of_players):
            game_record[f'player_{i}_type'] = str(self.player_types[i])
            game_record[f'player_{i}_placement'] = str(
                self.number_of_players - game_state.defeated_players.index(i)
                if i in game_state.defeated_players
                else 1
            )

        self.game_data.append(game_record)

    def save_all(self) -> None:
        """
        Saves the collected metrics to the specified CSV file.
        """
        if not self.action_data and not self.turn_data and not self.game_data:
            logger.warning('No data collected to save.')
            return

        # Ensure the output directory exists
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        try:
            for i in range(self.number_of_players):
                player_data = self.player_data[i]
                if player_data:
                    pd.DataFrame(player_data).to_csv(
                        f'{self.experiment_path}/player_level/run_{self.game_id}_player_{i}_data.csv',
                        index=False,
                    )

            pd.DataFrame(self.action_data).to_csv(
                f'{self.experiment_path}/action_level/run_{self.game_id}.csv',
                index=False,
            )
            pd.DataFrame(self.turn_data).to_csv(
                f'{self.experiment_path}/turn_level/run_{self.game_id}.csv', index=False
            )
            pd.DataFrame(self.game_data).to_csv(
                f'{self.experiment_path}/game_level.csv', index=False
            )

            logger.info(f'Metrics successfully saved to {self.experiment_path}')
        except OSError as e:
            logger.error(f'Error saving metrics to file: {e}')

    def reset(self) -> None:
        """
        Reset everything but the game data.
        """
        self.action_data = []
        self.turn_data = []
        self.player_data = [[] for _ in range(self.number_of_players)]
        # game_data is not reset to keep the final game outcomes
        logger.info('DataCollector reset for a new game run.')

    def _ensure_paths_exist(self) -> None:
        """
        Ensure the output directory exists.
        """
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
            logger.info(f'Created output directory: {self.experiment_path}')

        if not os.path.exists(f'{self.experiment_path}/action_level'):
            os.makedirs(f'{self.experiment_path}/action_level')
            logger.info(
                f'Created action level directory: {self.experiment_path}/action_level'
            )

        if not os.path.exists(f'{self.experiment_path}/turn_level'):
            os.makedirs(f'{self.experiment_path}/turn_level')
            logger.info(
                f'Created turn level directory: {self.experiment_path}/turn_level'
            )

        if not os.path.exists(f'{self.experiment_path}/player_level'):
            os.makedirs(f'{self.experiment_path}/player_level')
            logger.info(
                f'Created player level directory: {self.experiment_path}/player_level'
            )

    def load_existing_data(self) -> None:
        """
        Load existing game data from CSV files if they exist.
        """
        try:
            game_data_path = f'{self.experiment_path}/game_level.csv'
            if os.path.exists(game_data_path):
                self.game_data = pd.read_csv(game_data_path).to_dict('records')
                logger.info(f'Loaded existing game data from {game_data_path}')
            else:
                logger.info(f'No existing game data found at {game_data_path}')
        except OSError as e:
            logger.error(f'Error loading existing game data: {e}')
