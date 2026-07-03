import array
import json

from risk_agent.game_elements.board import Board
from risk_agent.game_elements.zobrist import (
    armies_key,
    compute_hands_deck_hash,
    defeated_key,
    flag_key,
    owner_key,
)


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

        # Incrementally-maintained Zobrist hash, split in two so that
        # mutating player_hands/deck (which requires a full recompute; see
        # recompute_hands_deck_hash()) never forces a recompute of the much
        # larger board-related contribution.
        self.board_hash: int = 0
        self.hands_deck_hash: int = 0

    @property
    def zobrist_hash(self) -> int:
        return self.board_hash ^ self.hands_deck_hash

    def __hash__(self) -> int:
        return self.zobrist_hash

    def recompute_hash(self) -> int:
        """
        Recompute board_hash and hands_deck_hash from scratch. Used to
        initialise a freshly-built state's hash and as a correctness oracle
        for the incremental updates performed by set_owner()/set_armies()/
        etc. elsewhere.
        """
        board_hash = 0
        for t in self.territory_ids:
            board_hash ^= owner_key(t, self.owner[t])
            board_hash ^= armies_key(t, self.armies[t])
        board_hash ^= flag_key('current_turn_phase', self.current_turn_phase)
        board_hash ^= flag_key('current_player', self.current_player)
        board_hash ^= flag_key('current_round', self.current_round)
        board_hash ^= flag_key('current_turn', self.current_turn)
        for player_id in self.defeated_players:
            board_hash ^= defeated_key(player_id)
        board_hash ^= flag_key(
            'conquered_territory_this_turn', self.conquered_territory_this_turn
        )
        board_hash ^= flag_key(
            'base_reinforcements_this_turn', self.base_reinforcements_this_turn
        )
        board_hash ^= flag_key(
            'reinforcements_this_turn', self.reinforcements_this_turn
        )
        board_hash ^= flag_key(
            'card_trade_in_this_turn', self.card_trade_in_this_turn
        )
        board_hash ^= flag_key(
            'fortified_territory_this_turn', self.fortified_territory_this_turn
        )
        self.board_hash = board_hash
        self.hands_deck_hash = compute_hands_deck_hash(self.player_hands, self.deck)
        return self.zobrist_hash

    def set_owner(self, territory_id: int, new_owner: int) -> None:
        """
        Set owner[territory_id], keeping board_hash in sync.
        """
        self.board_hash ^= owner_key(territory_id, self.owner[territory_id])
        self.owner[territory_id] = new_owner
        self.board_hash ^= owner_key(territory_id, new_owner)

    def set_armies(self, territory_id: int, new_armies: int) -> None:
        """
        Set armies[territory_id], keeping board_hash in sync.
        """
        self.board_hash ^= armies_key(territory_id, self.armies[territory_id])
        self.armies[territory_id] = new_armies
        self.board_hash ^= armies_key(territory_id, new_armies)

    def add_armies(self, territory_id: int, delta: int) -> None:
        self.set_armies(territory_id, self.armies[territory_id] + delta)

    def add_defeated_player(self, player_id: int) -> None:
        self.board_hash ^= defeated_key(player_id)
        self.defeated_players.append(player_id)

    def set_scalar(self, attr_name: str, new_value: object) -> None:
        """
        Set a scalar board-related field (current_turn_phase, current_player,
        current_round, current_turn, or one of the four turn flags), keeping
        board_hash in sync.
        """
        old_value = getattr(self, attr_name)
        self.board_hash ^= flag_key(attr_name, old_value)
        self.board_hash ^= flag_key(attr_name, new_value)
        setattr(self, attr_name, new_value)

    def recompute_hands_deck_hash(self) -> None:
        """
        Recompute hands_deck_hash from scratch. Must be called after any
        direct mutation of player_hands/deck (append/remove/extend/pop/...):
        unlike board_hash, this isn't updated incrementally, since a single
        card added to or removed from a hand can shift the sorted-occurrence
        index of every other card in that hand (see zobrist.hand_card_key).
        Hand sizes are small (<= ~10), so a full recompute here is cheap.
        """
        self.hands_deck_hash = compute_hands_deck_hash(self.player_hands, self.deck)

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

        new_state.board_hash = self.board_hash
        new_state.hands_deck_hash = self.hands_deck_hash

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

        self.recompute_hash()
