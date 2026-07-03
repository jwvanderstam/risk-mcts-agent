"""
A FEN-inspired, single-line, human-readable serialisation of a GameState.

Unlike chess FEN, fields are delimited rather than run-length-encoded,
since army counts and hand sizes are unbounded here. The board itself is
not serialised (mirroring GameState.to_json_file()/from_json_file()) -
decode_fen() reloads it from a board file path.

Format (space-separated top-level fields):
  1.  board: territories in ascending territory_id order, each encoded as
      "owner,armies", joined by "/".
  2.  number_of_players
  3.  current_player
  4.  current_turn_phase (word, or "-" if empty)
  5.  current_round
  6.  current_turn
  7.  defeated_players: comma-separated ints, or "-" if empty
  8.  conquered_territory_this_turn: "1" or "0"
  9.  base_reinforcements_this_turn
  10. reinforcements_this_turn
  11. card_trade_in_this_turn
  12. fortified_territory_this_turn: "1" or "0"
  13. deck: cards joined by ";", or "-" if empty
  14. player_hands: one hand per player_id (0..number_of_players-1),
      joined by "|"; each hand is its cards joined by ";", or "-" if empty

Cards are encoded as "territory,bonus,type" (comma-separated, not "-",
since bonus/territory can be -1 for wild/unknown placeholder cards).
"""

from risk_agent.game_elements.board import Board
from risk_agent.game_elements.game_state import GameState

_FIELD_SEP = ' '
_TERRITORY_SEP = '/'
_TERRITORY_FIELD_SEP = ','
_CARD_SEP = ';'
_CARD_FIELD_SEP = ','
_HAND_SEP = '|'
_LIST_SEP = ','
_EMPTY = '-'


def _encode_card(card: tuple[int, int, str]) -> str:
    territory, bonus, card_type = card
    return f'{territory}{_CARD_FIELD_SEP}{bonus}{_CARD_FIELD_SEP}{card_type}'


def _decode_card(token: str) -> tuple[int, int, str]:
    territory_str, bonus_str, card_type = token.split(_CARD_FIELD_SEP)
    return (int(territory_str), int(bonus_str), card_type)


def _encode_cards(cards: list[tuple[int, int, str]]) -> str:
    if not cards:
        return _EMPTY
    return _CARD_SEP.join(_encode_card(c) for c in cards)


def _decode_cards(token: str) -> list[tuple[int, int, str]]:
    if token == _EMPTY:
        return []
    return [_decode_card(t) for t in token.split(_CARD_SEP)]


def encode_fen(state: GameState) -> str:
    """
    Encode a GameState as a single-line FEN-like string.
    """
    board_field = _TERRITORY_SEP.join(
        f'{state.owner[t]}{_TERRITORY_FIELD_SEP}{state.armies[t]}'
        for t in state.territory_ids
    )

    defeated_field = (
        _LIST_SEP.join(str(p) for p in state.defeated_players)
        if state.defeated_players
        else _EMPTY
    )

    hands_field = _HAND_SEP.join(
        _encode_cards(state.player_hands.get(player_id, []))
        for player_id in range(state.number_of_players)
    )

    fields = [
        board_field,
        str(state.number_of_players),
        str(state.current_player),
        state.current_turn_phase or _EMPTY,
        str(state.current_round),
        str(state.current_turn),
        defeated_field,
        '1' if state.conquered_territory_this_turn else '0',
        str(state.base_reinforcements_this_turn),
        str(state.reinforcements_this_turn),
        str(state.card_trade_in_this_turn),
        '1' if state.fortified_territory_this_turn else '0',
        _encode_cards(state.deck),
        hands_field,
    ]
    return _FIELD_SEP.join(fields)


def decode_fen(fen: str, board_file_path: str) -> GameState:
    """
    Decode a string produced by encode_fen() back into a GameState.
    """
    (
        board_field,
        number_of_players_str,
        current_player_str,
        current_turn_phase_field,
        current_round_str,
        current_turn_str,
        defeated_field,
        conquered_str,
        base_reinforcements_str,
        reinforcements_str,
        card_trade_in_str,
        fortified_str,
        deck_field,
        hands_field,
    ) = fen.split(_FIELD_SEP)

    state = GameState()
    state.board = Board()
    state.board.load_from_file(board_file_path)

    state.number_of_players = int(number_of_players_str)

    territory_tokens = board_field.split(_TERRITORY_SEP)
    state.reset_arrays(len(territory_tokens) + 1)
    for territory_id, token in enumerate(territory_tokens, start=1):
        owner_str, armies_str = token.split(_TERRITORY_FIELD_SEP)
        state.owner[territory_id] = int(owner_str)
        state.armies[territory_id] = int(armies_str)

    state.current_player = int(current_player_str)
    state.current_turn_phase = (
        '' if current_turn_phase_field == _EMPTY else current_turn_phase_field
    )
    state.current_round = int(current_round_str)
    state.current_turn = int(current_turn_str)
    state.defeated_players = (
        []
        if defeated_field == _EMPTY
        else [int(p) for p in defeated_field.split(_LIST_SEP)]
    )
    state.conquered_territory_this_turn = conquered_str == '1'
    state.base_reinforcements_this_turn = int(base_reinforcements_str)
    state.reinforcements_this_turn = int(reinforcements_str)
    state.card_trade_in_this_turn = int(card_trade_in_str)
    state.fortified_territory_this_turn = fortified_str == '1'

    state.deck = _decode_cards(deck_field)

    state.player_hands = {
        player_id: _decode_cards(token)
        for player_id, token in enumerate(hands_field.split(_HAND_SEP))
    }

    state.recompute_hash()

    return state
