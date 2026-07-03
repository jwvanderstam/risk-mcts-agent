from risk_agent.game_elements.game_state import GameState


def _make_state(board) -> GameState:
    state = GameState()
    state.board = board
    state.number_of_players = 2
    state.territory_owners = {t: 0 if t == 1 else 1 for t in board.territories}
    state.territory_armies = {t: 5 if t == 1 else 1 for t in board.territories}
    state.player_territories = {
        0: [1],
        1: [t for t in board.territories if t != 1],
    }
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

    copy.territory_armies[1] = 999
    copy.territory_owners[2] = 0
    copy.player_territories[0].append(2)
    copy.player_hands[0].append((0, 1, 'infantry'))
    copy.defeated_players.append(1)

    assert state.territory_armies[1] == 5
    assert state.territory_owners[2] == 1
    assert state.player_territories[0] == [1]
    assert state.player_hands[0] == []
    assert state.defeated_players == []


def test_copy_produces_equal_state(board):
    state = _make_state(board)
    copy = state.copy()
    assert state == copy


def test_eq_detects_difference(board):
    state = _make_state(board)
    copy = state.copy()
    copy.territory_armies[1] = 6
    assert state != copy


def test_is_terminal_false_for_multi_owner_board(board):
    state = _make_state(board)
    assert not state.is_terminal()


def test_is_terminal_true_when_one_owner_remains(board):
    state = _make_state(board)
    for t in state.territory_owners:
        state.territory_owners[t] = 0
    assert state.is_terminal()
