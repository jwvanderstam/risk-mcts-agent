from risk_agent.game_elements.game_state import GameState


def _make_state(board) -> GameState:
    state = GameState()
    state.board = board
    state.number_of_players = 2
    state.reset_arrays(max(board.territories.keys()) + 1)
    for t in board.territories:
        state.owner[t] = 0 if t == 1 else 1
        state.armies[t] = 5 if t == 1 else 1
    state.player_hands = {0: [], 1: []}
    state.deck = []
    state.current_player = 0
    state.current_turn_phase = 'attack'
    state.current_round = 0
    state.current_turn = 0
    state.defeated_players = []
    return state


def test_copy_mutation_does_not_affect_original(board):
    state = _make_state(board)
    copy = state.copy()

    copy.armies[1] = 999
    copy.owner[2] = 0
    copy.player_hands[0].append((0, 1, 'infantry'))
    copy.defeated_players.append(1)

    assert state.armies[1] == 5
    assert state.owner[2] == 1
    assert state.player_hands[0] == []
    assert state.defeated_players == []


def test_copy_shares_board_reference(board):
    state = _make_state(board)
    copy = state.copy()
    assert state.board is copy.board


def test_copy_produces_equal_state(board):
    state = _make_state(board)
    copy = state.copy()
    assert state == copy


def test_eq_detects_difference(board):
    state = _make_state(board)
    copy = state.copy()
    copy.armies[1] = 6
    assert state != copy


def test_territories_owned_by(board):
    state = _make_state(board)
    assert state.territories_owned_by(0) == [1]
    assert 1 not in state.territories_owned_by(1)
    assert set(state.territories_owned_by(1)) == set(board.territories) - {1}


def test_is_terminal_false_for_multi_owner_board(board):
    state = _make_state(board)
    assert not state.is_terminal()


def test_is_terminal_true_when_one_owner_remains(board):
    state = _make_state(board)
    for t in state.territory_ids:
        state.owner[t] = 0
    assert state.is_terminal()
