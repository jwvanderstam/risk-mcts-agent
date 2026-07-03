"""Regression harness for the array-based GameState refactor.

Each golden game recorded in fixtures/golden_games.json contains the exact
sequence of (action, outcome/card) pairs chosen by a random-vs-random game
played with the pre-refactor dict-based GameState. Replaying that recorded
trace via Action.apply()/apply_outcome() - bypassing get_valid_actions() and
the BattleComputer's RNG entirely - lets us confirm the state-mutation logic
stays behaviourally identical across the refactor without depending on
territory enumeration order, which the refactor is free to change.
"""

import random

from risk_agent.engine.game_engine import GameEngine
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


def _replay(game: dict, board: Board) -> GameState:
    # Reproduce the exact RNG stream state that generation left
    # initialise_random_game_state() with: seed, then the num_players draw.
    random.seed(game['seed'])
    num_players = random.choice([2, 3, 4, 5, 6])
    assert num_players == game['num_players']

    game_state = GameState()
    game_state.board = board
    game_state = GameEngine.initialise_random_game_state(
        game_state, game['num_players']
    )
    game_state.base_reinforcements_this_turn = (
        GameEngine.calculate_base_reinforcements(
            game_state, game_state.current_player
        )
    )

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

        actual_owners = {str(k): v for k, v in final_state.territory_owners.items()}
        actual_armies = {str(k): v for k, v in final_state.territory_armies.items()}
        assert actual_owners == expected['territory_owners']
        assert actual_armies == expected['territory_armies']
