import json
from pathlib import Path

import pytest

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.game_elements.board import Board

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
BOARD_FILE = str(Path(__file__).parent.parent / 'data' / 'standard_board.json')


@pytest.fixture(scope='session')
def board() -> Board:
    b = Board()
    b.load_from_file(BOARD_FILE)
    return b


@pytest.fixture(scope='session')
def small_battle_computer() -> BattleComputer:
    """A BattleComputer with tiny army caps so the stationary-distribution
    matrix is built in milliseconds instead of minutes. Only usable for
    battles with attacking/defending armies <= 5."""
    return BattleComputer(max_attacking_armies=5, max_defending_armies=5)


@pytest.fixture(scope='session')
def golden_games() -> list[dict]:
    with open(FIXTURES_DIR / 'golden_games.json') as f:
        return json.load(f)
