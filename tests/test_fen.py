from pathlib import Path

from risk_agent.game_elements.fen import decode_fen, encode_fen
from risk_agent.game_elements.game_state import GameState

BOARD_FILE = str(Path(__file__).parent.parent / 'data' / 'standard_board.json')


def _make_fresh_state(board) -> GameState:
    state = GameState()
    state.board = board
    state.number_of_players = 3
    state.reset_arrays(max(board.territories.keys()) + 1)
    for t in board.territories:
        state.owner[t] = t % 3
        state.armies[t] = 1 + (t % 5)
    state.player_hands = {0: [], 1: [], 2: []}
    state.deck = [(i, i, 'infantry') for i in range(3)]
    state.current_player = 0
    state.current_turn_phase = 'reinforce'
    state.current_round = 0
    state.current_turn = 0
    state.defeated_players = []
    state.recompute_hash()
    return state


def test_round_trip_fresh_state(board):
    state = _make_fresh_state(board)
    decoded = decode_fen(encode_fen(state), BOARD_FILE)
    assert decoded == state


def test_round_trip_with_hands_deck_and_progress(board):
    state = _make_fresh_state(board)
    state.current_turn_phase = 'attack'
    state.current_round = 3
    state.current_turn = 9
    state.defeated_players = [1]
    state.conquered_territory_this_turn = True
    state.base_reinforcements_this_turn = 5
    state.reinforcements_this_turn = 2
    state.card_trade_in_this_turn = 4
    state.fortified_territory_this_turn = True
    state.player_hands = {
        0: [(1, 2, 'infantry'), (-1, -1, 'unknown'), (-1, -1, 'unknown')],
        1: [],
        2: [(3, 4, 'wild')],
    }
    state.deck = [(-1, -1, 'wild'), (5, 6, 'cavalry')]
    state.recompute_hash()

    decoded = decode_fen(encode_fen(state), BOARD_FILE)
    assert decoded == state


def test_round_trip_via_gamestate_wrappers(board):
    state = _make_fresh_state(board)
    fen = state.to_fen()

    loaded = GameState()
    loaded.from_fen(fen, BOARD_FILE)
    assert loaded == state


def test_round_trip_each_turn_phase(board):
    for phase in ['trade_cards', 'reinforce', 'attack', 'fortify']:
        state = _make_fresh_state(board)
        state.current_turn_phase = phase
        state.recompute_hash()
        decoded = decode_fen(encode_fen(state), BOARD_FILE)
        assert decoded.current_turn_phase == phase
        assert decoded == state


def test_round_trip_empty_hands_and_deck(board):
    state = _make_fresh_state(board)
    state.player_hands = {0: [], 1: [], 2: []}
    state.deck = []
    state.recompute_hash()
    decoded = decode_fen(encode_fen(state), BOARD_FILE)
    assert decoded.deck == []
    assert decoded.player_hands == {0: [], 1: [], 2: []}
    assert decoded == state
