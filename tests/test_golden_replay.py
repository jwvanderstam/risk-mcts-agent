"""Regression harness for the array-based GameState refactor.

Each golden game recorded in fixtures/golden_games.json contains:
  - `initial`: a snapshot of the state right after
    GameEngine.initialise_random_game_state() + calculate_base_reinforcements(),
    captured directly rather than re-derived by re-running
    initialise_random_game_state() against the refactored code. That
    function's internal RNG-consumption order (e.g. which order a player's
    territories are iterated over while distributing starting armies) is an
    implementation detail the refactor is free to change; the initial
    territory/army layout for a given seed is not.
  - `steps`: the exact sequence of (action, outcome/card) pairs chosen by a
    random-vs-random game played against the pre-refactor dict-based
    GameState. Replaying that recorded trace via Action.apply()/
    apply_outcome() - bypassing get_valid_actions() and the BattleComputer's
    RNG entirely - lets us confirm the state-mutation logic stays
    behaviourally identical across the refactor without depending on
    territory enumeration order, which the refactor is also free to change.
"""

from risk_agent.game_elements.action import (
    AttackAction,
    EndPhaseAction,
    EndTurnAction,
    FortifyAction,
    ReinforceAction,
    TradeCardsAction,
)
from risk_agent.game_elements.board import Board
from risk_agent.game_elements.game_state import GameState


def _rebuild_action(step: dict):
    params = step['params']
    match step['type']:
        case 'ReinforceAction':
            return ReinforceAction(**params)
        case 'AttackAction':
            return AttackAction(**params)
        case 'FortifyAction':
            return FortifyAction(**params)
        case 'TradeCardsAction':
            return TradeCardsAction(
                player_id=params['player_id'],
                cards=[tuple(c) for c in params['cards']],
                value=params['value'],
            )
        case 'EndTurnAction':
            return EndTurnAction()
        case 'EndPhaseAction':
            return EndPhaseAction()
        case _:
            raise TypeError(f'Unhandled action type: {step["type"]}')


def _build_initial_state(game: dict, board: Board, num_territories: int) -> GameState:
    initial = game['initial']

    state = GameState()
    state.board = board
    state.number_of_players = game['num_players']
    state.reset_arrays(num_territories)
    for territory_str, owner in initial['territory_owners'].items():
        state.owner[int(territory_str)] = owner
    for territory_str, armies in initial['territory_armies'].items():
        state.armies[int(territory_str)] = armies

    state.player_hands = {
        int(p): [tuple(c) for c in hand]
        for p, hand in initial['player_hands'].items()
    }
    state.deck = [tuple(c) for c in initial['deck']]
    state.current_player = initial['current_player']
    state.current_turn_phase = initial['current_turn_phase']
    state.current_turn = initial['current_turn']
    state.current_round = initial['current_round']
    state.defeated_players = list(initial['defeated_players'])
    state.conquered_territory_this_turn = initial['conquered_territory_this_turn']
    state.base_reinforcements_this_turn = initial['base_reinforcements_this_turn']
    state.reinforcements_this_turn = initial['reinforcements_this_turn']
    state.card_trade_in_this_turn = initial['card_trade_in_this_turn']
    state.fortified_territory_this_turn = initial['fortified_territory_this_turn']
    return state


def _replay(game: dict, board: Board) -> GameState:
    num_territories = max(board.territories.keys()) + 1
    game_state = _build_initial_state(game, board, num_territories)

    for step in game['steps']:
        action = _rebuild_action(step)
        if isinstance(action, AttackAction):
            outcome = tuple(step['outcome'])
            game_state = action.apply_outcome(game_state=game_state, outcome=outcome)
        elif isinstance(action, EndTurnAction):
            card = tuple(step['card']) if step['card'] is not None else None
            game_state = action.apply(game_state=game_state, card=card)
        else:
            game_state = action.apply(game_state=game_state)

    return game_state


def test_golden_games_replay_to_identical_final_state(golden_games, board):
    for game in golden_games:
        final_state = _replay(game, board)
        expected = game['final']

        assert final_state.is_terminal() == expected['terminal']
        assert final_state.determine_winner() == expected['winner']
        assert final_state.current_turn == expected['current_turn']
        assert final_state.current_round == expected['current_round']
        assert sorted(final_state.defeated_players) == expected['defeated_players']

        actual_owners = {
            str(t): final_state.owner[t] for t in final_state.territory_ids
        }
        actual_armies = {
            str(t): final_state.armies[t] for t in final_state.territory_ids
        }
        assert actual_owners == expected['territory_owners']
        assert actual_armies == expected['territory_armies']
