import random

# Fixed seed: keys must be stable across processes/runs so that two
# GameState instances built independently (e.g. one freshly initialised,
# one loaded from disk) hash consistently. This RNG is entirely separate
# from the `random` module used for gameplay (shuffling, battle outcomes,
# random.choice(valid_actions), ...) so generating keys never perturbs
# game replay determinism.
_SEED = 0x515C0FFEE
_rng = random.Random(_SEED)

_owner_keys: dict[tuple[int, int], int] = {}
_armies_keys: dict[tuple[int, int], int] = {}
_defeated_keys: dict[int, int] = {}
_flag_keys: dict[tuple[str, object], int] = {}
_hand_card_keys: dict[tuple[int, tuple, int], int] = {}
_deck_card_keys: dict[tuple[tuple, int], int] = {}


def _new_key() -> int:
    return _rng.getrandbits(64)


def owner_key(territory_id: int, owner: int) -> int:
    k = (territory_id, owner)
    if k not in _owner_keys:
        _owner_keys[k] = _new_key()
    return _owner_keys[k]


def armies_key(territory_id: int, armies: int) -> int:
    k = (territory_id, armies)
    if k not in _armies_keys:
        _armies_keys[k] = _new_key()
    return _armies_keys[k]


def defeated_key(player_id: int) -> int:
    if player_id not in _defeated_keys:
        _defeated_keys[player_id] = _new_key()
    return _defeated_keys[player_id]


def flag_key(name: str, value: object) -> int:
    """
    Generic key for a scalar (non-territory, non-hand) GameState field,
    keyed by field name and its current value - covers current_turn_phase,
    current_player, current_round, current_turn, and the four turn flags.
    """
    k = (name, value)
    if k not in _flag_keys:
        _flag_keys[k] = _new_key()
    return _flag_keys[k]


def hand_card_key(player_id: int, card: tuple, occurrence_index: int) -> int:
    """
    Key for the `occurrence_index`-th card (in sorted order) of a player's
    hand. Including the occurrence index means two identical cards in the
    same hand (e.g. two synthetic (-1, -1, 'unknown') placeholder cards
    used during determinization) get distinct keys instead of XOR-cancelling
    each other out, which would otherwise make "2 unknown cards" hash the
    same as "0 unknown cards".
    """
    k = (player_id, card, occurrence_index)
    if k not in _hand_card_keys:
        _hand_card_keys[k] = _new_key()
    return _hand_card_keys[k]


def deck_card_key(card: tuple, occurrence_index: int) -> int:
    k = (card, occurrence_index)
    if k not in _deck_card_keys:
        _deck_card_keys[k] = _new_key()
    return _deck_card_keys[k]


def compute_hands_deck_hash(
    player_hands: dict[int, list[tuple]], deck: list[tuple]
) -> int:
    """
    Hash of player_hands + deck, independent of list/dict iteration order
    but sensitive to each player's card multiset (including duplicates).
    """
    h = 0
    for player_id in sorted(player_hands.keys()):
        for occurrence_index, card in enumerate(sorted(player_hands[player_id])):
            h ^= hand_card_key(player_id, card, occurrence_index)
    for occurrence_index, card in enumerate(sorted(deck)):
        h ^= deck_card_key(card, occurrence_index)
    return h
