from risk_agent.engine.game_engine import GameEngine
from risk_agent.game_elements.action import AttackAction, EndPhaseAction
from risk_agent.game_elements.game_state import GameState


def _make_attack_phase_state(board) -> GameState:
    """Player 0 holds only territory 1 (adjacent to 2, 4, 35, all owned by
    player 1); player 1 holds everything else."""
    state = GameState()
    state.board = board
    state.number_of_players = 2
    state.reset_arrays(max(board.territories.keys()) + 1)
    for t in board.territories:
        state.owner[t] = 0 if t == 1 else 1
        state.armies[t] = 5 if t == 1 else 3
    state.player_hands = {0: [], 1: []}
    state.deck = []
    state.current_player = 0
    state.current_turn_phase = 'attack'
    state.current_round = 0
    state.current_turn = 0
    state.defeated_players = []
    return state


def test_get_valid_actions_attack_phase_targets_only_enemy_neighbours(board):
    state = _make_attack_phase_state(board)
    actions = GameEngine.get_valid_actions(state)

    attack_actions = [a for a in actions if isinstance(a, AttackAction)]
    end_phase_actions = [a for a in actions if isinstance(a, EndPhaseAction)]

    assert len(end_phase_actions) == 1
    targets = {a.to_territory for a in attack_actions}
    assert targets == set(board.adjacency_list[1])
    for action in attack_actions:
        assert action.from_territory == 1
        assert action.attacking_armies == 4  # armies[1] - 1
        assert action.defending_armies == 3


def test_get_valid_actions_no_attacks_when_only_one_army(board):
    state = _make_attack_phase_state(board)
    state.armies[1] = 1

    actions = GameEngine.get_valid_actions(state)
    assert all(not isinstance(a, AttackAction) for a in actions)
    assert len(actions) == 1
    assert isinstance(actions[0], EndPhaseAction)


def test_calculate_base_reinforcements_continent_bonus(board):
    state = _make_attack_phase_state(board)
    north_america = next(
        c for c in board.continents.values() if c['name'] == 'North America'
    )
    for t in north_america['territories']:
        state.owner[t] = 0

    reinforcements = GameEngine.calculate_base_reinforcements(state, player_id=0)
    assert reinforcements >= north_america['bonus']
