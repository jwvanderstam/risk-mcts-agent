import array
import json

from risk_agent.game_elements.board import Board


class GameState:
    def __init__(self) -> None:
        """
        Initialise the game state with default values.
        """
        self.board = Board()
        self.number_of_players: int = -1
        # owner[t] / armies[t] are indexed directly by territory_id.
        # Index 0 is unused (territory ids start at 1); owner[0] stays -1.
        self.owner: array.array = array.array('b', [])
        self.armies: array.array = array.array('i', [])
        self.player_hands: dict[int, list[tuple[int, int, str]]] = {}
        self.deck: list[tuple[int, int, str]] = []
        self.current_player: int = -1
        self.current_turn_phase: str = (
            ''  # 'trade_cards', 'reinforce', 'attack', or 'fortify'
        )
        self.current_round: int = 0
        self.current_turn: int = 0
        self.defeated_players: list[int] = []

        self.conquered_territory_this_turn: bool = False
        self.base_reinforcements_this_turn: int = 0
        self.reinforcements_this_turn: int = 0
        self.card_trade_in_this_turn: int = -1
        self.fortified_territory_this_turn: bool = False

    def reset_arrays(self, num_territories: int) -> None:
        """
        (Re)initialise the owner/armies arrays to hold `num_territories` slots
        (indices 0..num_territories-1; index 0 is left unused as a sentinel).
        """
        self.owner = array.array('b', [-1]) * num_territories
        self.armies = array.array('i', [0]) * num_territories

    @property
    def territory_ids(self) -> range:
        """
        The valid territory ids for this state (excludes the unused index 0).
        """
        return range(1, len(self.owner))

    def territories_owned_by(self, player_id: int) -> list[int]:
        """
        Return the list of territory ids currently owned by a player.
        """
        return [t for t in self.territory_ids if self.owner[t] == player_id]

    def is_terminal(self) -> bool:
        """
        Check if the game state is terminal (i.e., only one player remains).
        """
        return len(set(self.owner[1:])) == 1

    def determine_winner(self) -> int:
        """
        Determine the winner of the game if it is terminal.
        Returns the player ID of the winner, or -1 if no winner is determined.
        """
        if not self.is_terminal():
            return -2
        for player_id in range(self.number_of_players):
            if player_id not in self.defeated_players:
                return player_id
        return -1

    def __str__(self) -> str:
        """
        Return a string representation of the game state.
        """
        return (
            f'GameState(current_player={self.current_player}, '
            f'current_turn_phase={self.current_turn_phase}, '
            f'current_turn={self.current_turn}, '
            f'owner={list(self.owner)}, '
            f'armies={list(self.armies)}, '
            f'player_hands={self.player_hands}, '
            f'defeated_players={self.defeated_players})'
        )

    def __eq__(self, value: object) -> bool:
        # Compare two game states for equality.
        if not isinstance(value, GameState):
            return False
        return (
            self.board == value.board
            and self.number_of_players == value.number_of_players
            and self.owner == value.owner
            and self.armies == value.armies
            and self.player_hands == value.player_hands
            and self.deck == value.deck
            and self.current_player == value.current_player
            and self.current_turn_phase == value.current_turn_phase
            and self.current_round == value.current_round
            and self.current_turn == value.current_turn
            and self.defeated_players == value.defeated_players
            and self.conquered_territory_this_turn
            == value.conquered_territory_this_turn
            and self.reinforcements_this_turn == value.reinforcements_this_turn
            and self.base_reinforcements_this_turn
            == value.base_reinforcements_this_turn
            and self.card_trade_in_this_turn == value.card_trade_in_this_turn
            and self.fortified_territory_this_turn
            == value.fortified_territory_this_turn
        )

    def copy(self) -> 'GameState':
        """
        Create a copy of the game state.
        """
        new_state = GameState()
        # The board is never mutated after being loaded, so it is safe (and
        # much cheaper) to share the reference instead of deep-copying it.
        new_state.board = self.board
        new_state.number_of_players = self.number_of_players
        new_state.owner = array.array(self.owner.typecode, self.owner)
        new_state.armies = array.array(self.armies.typecode, self.armies)

        # Create a deep copy of player hands and deck
        new_state.player_hands = {p: list(h) for p, h in self.player_hands.items()}
        new_state.deck = list(self.deck)

        new_state.current_player = self.current_player
        new_state.current_turn_phase = self.current_turn_phase
        new_state.current_round = self.current_round
        new_state.current_turn = self.current_turn
        new_state.defeated_players = self.defeated_players.copy()

        new_state.conquered_territory_this_turn = self.conquered_territory_this_turn
        new_state.reinforcements_this_turn = self.reinforcements_this_turn
        new_state.base_reinforcements_this_turn = self.base_reinforcements_this_turn
        new_state.card_trade_in_this_turn = self.card_trade_in_this_turn
        new_state.fortified_territory_this_turn = self.fortified_territory_this_turn

        return new_state

    def to_json_file(self, file_path: str) -> None:
        """
        Save the game state to a JSON file.
        """
        # Save everything except the board
        data = self.__dict__.copy()
        data['board'] = None
        data['owner'] = list(self.owner)
        data['armies'] = list(self.armies)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def from_json_file(self, file_path: str, board_file_path: str) -> None:
        """
        Load the game state from a JSON file.
        """
        # Load everything except the board
        with open(file_path) as f:
            data = json.load(f)
        self.__dict__.update(data)

        self.owner = array.array('b', self.owner)
        self.armies = array.array('i', self.armies)

        # Convert dictionary keys from string back to int after loading
        self.player_hands = {int(k): v for k, v in self.player_hands.items()}

        # Load the board separately
        self.board = Board()
        self.board.load_from_file(board_file_path)
