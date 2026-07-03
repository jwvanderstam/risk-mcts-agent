import random

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.engine.game_engine import GameEngine
from risk_agent.game_elements.action import (
    AttackAction,
    EndPhaseAction,
    EndTurnAction,
)
from risk_agent.game_elements.game_state import GameState
from risk_agent.game_elements.zobrist import compute_hands_deck_hash
from risk_agent.players.mcts.config import MCTSConfig
from risk_agent.players.mcts.tree import MCTSNode, MCTSTree
from tests.test_golden_replay import _build_initial_state, _rebuild_action

# Only a handful of full games is enough coverage for a per-step differential
# check; the golden fixture has 20 games with 600-5000 steps each.
GAMES_TO_CHECK = 5


def _apply_step(game_state: GameState, step: dict) -> GameState:
    action = _rebuild_action(step)
    if isinstance(action, AttackAction):
        outcome = tuple(step['outcome'])
        return action.apply_outcome(game_state=game_state, outcome=outcome)
    if isinstance(action, EndTurnAction):
        card = tuple(step['card']) if step['card'] is not None else None
        return action.apply(game_state=game_state, card=card)
    return action.apply(game_state=game_state)


def test_hash_consistent_through_real_game_engine_initialisation(board):
    """
    Regression test: GameEngine.initialise_random_game_state() must produce
    a state whose hash already reflects base_reinforcements_this_turn, since
    every real caller (GameManager._setup_game(), and every ad-hoc script)
    sets that field via plain attribute assignment right after this call
    returns - bypassing set_scalar() - rather than through this module's
    other, incrementally-hash-maintained mutation paths. Building the
    initial state some other way (as the golden-fixture replay tests do,
    snapshotting a fully-populated state and calling recompute_hash() last)
    would not have caught a staleness bug here.
    """
    random.seed(123)
    battle_computer = BattleComputer(max_attacking_armies=100, max_defending_armies=100)

    state = GameState()
    state.board = board
    state = GameEngine.initialise_random_game_state(state, 4)
    state.base_reinforcements_this_turn = GameEngine.calculate_base_reinforcements(
        state, state.current_player
    )
    assert state.zobrist_hash == state.copy().recompute_hash()

    for _ in range(200):
        if state.is_terminal():
            break
        current_player_id = state.current_player
        if current_player_id in state.defeated_players:
            state = GameEngine.apply_action(state, EndTurnAction())
        else:
            valid_actions = GameEngine.get_valid_actions(state)
            action = (
                random.choice(valid_actions)
                if valid_actions
                else (
                    EndTurnAction()
                    if state.current_turn_phase == 'fortify'
                    else EndPhaseAction()
                )
            )
            state = GameEngine.apply_action(state, action, battle_computer)
        assert state.zobrist_hash == state.copy().recompute_hash()


def test_incremental_hash_matches_full_recompute_after_every_step(
    golden_games, board
):
    num_territories = max(board.territories.keys()) + 1

    for game in golden_games[:GAMES_TO_CHECK]:
        game_state = _build_initial_state(game, board, num_territories)
        game_state.recompute_hash()
        assert game_state.zobrist_hash == game_state.copy().recompute_hash()

        for step in game['steps']:
            game_state = _apply_step(game_state, step)
            # Recompute on a copy so we don't clobber the incrementally
            # maintained hash being verified.
            assert game_state.zobrist_hash == game_state.copy().recompute_hash()


def test_equal_states_have_equal_hash(board):
    state = GameState()
    state.board = board
    state.number_of_players = 2
    state.reset_arrays(max(board.territories.keys()) + 1)
    for t in board.territories:
        state.owner[t] = 0 if t == 1 else 1
        state.armies[t] = 3
    state.player_hands = {0: [(0, 1, 'infantry')], 1: []}
    state.deck = [(1, 2, 'cavalry')]
    state.current_player = 0
    state.current_turn_phase = 'attack'
    state.recompute_hash()

    copy = state.copy()
    assert state == copy
    assert hash(state) == hash(copy)


def test_game_state_is_hashable(board):
    state = GameState()
    state.board = board
    state.reset_arrays(max(board.territories.keys()) + 1)
    state.recompute_hash()
    {state}  # noqa: B018 - putting it in a set is the actual assertion


def test_duplicate_unknown_cards_do_not_cancel_in_hash():
    """
    Regression test for the Zobrist XOR-cancellation pitfall: two identical
    synthetic (-1, -1, 'unknown') placeholder cards used during
    determinization must not hash the same as zero or one such cards.
    """
    zero_unknown = compute_hands_deck_hash({0: []}, [])
    one_unknown = compute_hands_deck_hash({0: [(-1, -1, 'unknown')]}, [])
    two_unknown = compute_hands_deck_hash(
        {0: [(-1, -1, 'unknown'), (-1, -1, 'unknown')]}, []
    )

    assert len({zero_unknown, one_unknown, two_unknown}) == 3


def _make_two_child_tree(board) -> tuple[MCTSTree, GameState, GameState]:
    root_state = GameState()
    root_state.board = board
    root_state.number_of_players = 2
    root_state.reset_arrays(max(board.territories.keys()) + 1)
    for t in board.territories:
        root_state.owner[t] = 0 if t == 1 else 1
        root_state.armies[t] = 3
    root_state.player_hands = {0: [], 1: []}
    root_state.deck = []
    root_state.current_player = 0
    root_state.current_turn_phase = 'reinforce'
    root_state.recompute_hash()

    config = MCTSConfig(
        max_attacking_armies=5,
        max_defending_armies=5,
        number_of_players=2,
    )
    tree = MCTSTree(root_state=root_state, player_id=0, config=config)

    state_a = root_state.copy()
    state_a.set_armies(1, 10)
    state_b = root_state.copy()
    state_b.set_armies(1, 20)

    child_a = MCTSNode(state=state_a)
    child_b = MCTSNode(state=state_b)
    tree.root.children = [child_a, child_b]

    return tree, state_a, state_b


def test_update_root_promotes_matching_child_by_hash(board):
    tree, state_a, state_b = _make_two_child_tree(board)
    child_a = tree.root.children[0]

    tree.update_root(state_a.copy())

    assert tree.root is child_a
    assert tree.root.state == state_a
