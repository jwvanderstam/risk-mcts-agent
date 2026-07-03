import logging
import random
from collections import Counter, deque
from itertools import combinations

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.game_elements.action import (
    Action,
    AttackAction,
    EndPhaseAction,
    EndTurnAction,
    FortifyAction,
    ReinforceAction,
    TradeCardsAction,
)
from risk_agent.game_elements.game_state import GameState

logger = logging.getLogger(__name__)


class GameEngine:
    def __init__(self) -> None:
        return None

    @staticmethod
    def initialise_random_game_state(
        game_state: GameState, number_of_players: int
    ) -> GameState:
        """
        Initialise a game state with random values.
        """
        game_state.current_player = 0
        game_state.current_turn_phase = 'reinforce'
        game_state.current_turn = 0
        game_state.conquered_territory_this_turn = False
        game_state.number_of_players = number_of_players

        # Randomize player positions
        territories = list(game_state.board.territories.keys())
        random.shuffle(territories)
        game_state.reset_arrays(max(game_state.board.territories.keys()) + 1)
        for i, territory in enumerate(territories):
            game_state.owner[territory] = i % number_of_players

        # Each player starts with 30 armies, one per territory, then distributed randomly
        for player_id in range(number_of_players):
            player_territories = game_state.territories_owned_by(player_id)
            for territory in player_territories:
                game_state.armies[territory] = 1

            armies_to_distribute = 30 - len(player_territories)
            for _ in range(armies_to_distribute):
                random_territory = random.choice(player_territories)
                game_state.armies[random_territory] += 1

        # Populate the deck with cards
        card_types = {
            0: 'infantry',
            1: 'cavalry',
            2: 'artillery',
        }

        game_state.deck = [
            (i, territory, card_types[territory % 3])
            for i, territory in enumerate(territories)
        ]
        game_state.deck.append((len(game_state.deck), -1, 'wild'))
        game_state.deck.append((len(game_state.deck), -1, 'wild'))

        # Initialize player hands
        game_state.player_hands = {i: [] for i in range(number_of_players)}

        # This builds up the state via direct field assignment rather than
        # the incremental set_owner()/set_armies()/set_scalar() setters (there
        # is no prior state to increment from), so establish the Zobrist hash
        # with a single full recompute instead.
        game_state.recompute_hash()

        return game_state

    @staticmethod
    def apply_action(
        game_state: GameState,
        action: Action,
        battle_computer: BattleComputer | None = None,
    ) -> GameState:
        """
        Apply the action to the game state and return the new game state.
        """
        # If the actions is an AttackAction, pass the battle computer the action
        if isinstance(action, AttackAction):
            if battle_computer is None:
                raise ValueError('Battle computer must be provided for attack action.')
            new_game_state = action.apply(
                game_state=game_state, battle_computer=battle_computer
            )
        else:
            new_game_state = action.apply(game_state=game_state)

        return new_game_state

    @staticmethod
    def _find_reachable_territories(
        territory: int,
        player_id: int,
        game_state: GameState,
    ) -> list[int]:
        """
        Find all reachable territories for a player from a given territory.
        Where reachable means that the territories are connected through the player's own territories.
        Uses a breadth-first search to find all reachable territories.
        """
        reachable_territories = set()
        visited = set()
        queue = deque([territory])

        while queue:
            current_territory = queue.popleft()
            if current_territory in visited:
                continue
            visited.add(current_territory)

            if game_state.owner[current_territory] == player_id:
                reachable_territories.add(current_territory)
                for adjacent_territory in game_state.board.adjacency_list[
                    current_territory
                ]:
                    if (
                        adjacent_territory not in visited
                        and game_state.owner[adjacent_territory] == player_id
                    ):
                        queue.append(adjacent_territory)

        return list(reachable_territories)

    @staticmethod
    def get_valid_actions(game_state: GameState) -> list[Action]:
        """
        Get the valid actions for the current game state.
        """
        # to avoid repeated dict lookups, we pull out local variables
        owners = game_state.owner
        armies = game_state.armies
        player = game_state.current_player

        valid_actions = []

        match game_state.current_turn_phase:
            case 'trade_cards':
                # If the player has less than 3 cards, they cannot trade
                # if the player has 3 or more cards, they can trade
                # if the player has 5 or more cards, they must trade
                if len(game_state.player_hands[player]) < 3:
                    return [EndPhaseAction()]

                possible_card_trades = GameEngine.determine_card_trades(
                    game_state=game_state,
                )
                if possible_card_trades:
                    valid_actions.extend(possible_card_trades)

                if not len(game_state.player_hands[player]) > 4:
                    valid_actions.append(EndPhaseAction())
            case 'reinforce':
                total_reinforcements = GameEngine.determine_reinforcements(
                    game_state=game_state,
                )

                if game_state.reinforcements_this_turn == total_reinforcements:
                    return [EndPhaseAction()]

                for territory in game_state.territories_owned_by(player):
                    for number_of_armies in range(
                        1,
                        total_reinforcements - game_state.reinforcements_this_turn + 1,
                    ):
                        valid_actions.append(
                            ReinforceAction(
                                territory=territory, armies=number_of_armies
                            )
                        )
            case 'attack':
                for territory in game_state.territories_owned_by(player):
                    if armies[territory] > 1:
                        for target_territory in game_state.board.adjacency_list[
                            territory
                        ]:
                            if owners[target_territory] != player:
                                valid_actions.append(
                                    AttackAction(
                                        from_territory=territory,
                                        to_territory=target_territory,
                                        attacking_armies=armies[territory] - 1,
                                        defending_armies=armies[target_territory],
                                    )
                                )
                valid_actions.append(EndPhaseAction())
            case 'fortify':
                if not game_state.fortified_territory_this_turn:
                    for from_territory in game_state.territories_owned_by(player):
                        if armies[from_territory] > 1:
                            reachable_territories = (
                                GameEngine._find_reachable_territories(
                                    territory=from_territory,
                                    player_id=player,
                                    game_state=game_state,
                                )
                            )
                            for to_territory in reachable_territories:
                                if (
                                    to_territory != from_territory
                                    and owners[to_territory] == player
                                ):
                                    valid_actions.append(
                                        FortifyAction(
                                            from_territory=from_territory,
                                            to_territory=to_territory,
                                            armies=armies[from_territory] - 1,
                                        )
                                    )
                valid_actions.append(EndTurnAction())
        return valid_actions

    @staticmethod
    def determine_card_trades(
        game_state: GameState,
    ) -> list:
        """
        Return a list of valid card trades for the current player.
        """
        current_player_cards = game_state.player_hands[game_state.current_player]
        if len(current_player_cards) < 3:
            return []

        valid_trades = []

        # Only these types can form three-of-a-kind; wilds are used as substitutes
        card_types = ['infantry', 'cavalry', 'artillery']
        card_values = {
            'infantry': 4,
            'cavalry': 6,
            'artillery': 8,
            'mixed': 10,
        }

        for card_combination in combinations(current_player_cards, 3):
            counts = Counter(card[2] for card in card_combination)
            wilds = counts.get('wild', 0)
            value = 0

            # Check for three-of-a-kind trades
            for type in card_types:
                if counts[type] == 3 or (counts[type] == 2 and wilds == 1):
                    value = card_values[type]
                    break

            # Check for mixed trades
            if value == 0:
                present_types = set(
                    card[2] for card in card_combination if card[2] != 'wild'
                )
                if len(present_types) + wilds == 3:
                    value = card_values['mixed']

            if value > 0:
                valid_trades.append(
                    TradeCardsAction(
                        player_id=game_state.current_player,
                        cards=list(card_combination),
                        value=value,
                    )
                )

        return valid_trades

    @staticmethod
    def calculate_base_reinforcements(
        game_state: GameState,
        player_id: int,
    ) -> int:
        """
        Calculate the base number of reinforcements for a player.
        The base reinforcements are calculated as follows:
        - 3 armies for every 3 territories owned, with a minimum of 3 armies.
        - Additional armies for owning continents.
        """
        base_reinforcements = max(
            3, len(game_state.territories_owned_by(player_id)) // 3
        )

        continent_bonus = 0
        for continent in game_state.board.continents.values():
            if all(
                game_state.owner[territory] == player_id
                for territory in continent['territories']
            ):
                continent_bonus += continent['bonus']

        return base_reinforcements + continent_bonus

    @staticmethod
    def determine_reinforcements(
        game_state: GameState,
    ) -> int:
        """
        Determine the number of reinforcements for the current player based on the
        pre-calculated base amount and any card trade-ins.
        """
        if game_state.card_trade_in_this_turn > 0:
            return (
                game_state.base_reinforcements_this_turn
                + game_state.card_trade_in_this_turn
            )
        else:
            return game_state.base_reinforcements_this_turn

    @staticmethod
    def get_border_territories(game_state: GameState, player_id: int) -> list[int]:
        """
        Get the border territories for a player.
        """
        border_territories = []
        for territory in game_state.territories_owned_by(player_id):
            for adjacent_territory in game_state.board.adjacency_list[territory]:
                if game_state.owner[adjacent_territory] != player_id:
                    border_territories.append(territory)
                    break
        return border_territories

    @staticmethod
    def determinize_game_state(game_state: GameState, player_id: int) -> GameState:
        """
        Create a determinized game state for a specific player.
        This is used to create a game state that is deterministic for the player,
        meaning that all other players' hands are replaced with unknown cards.
        """
        new_state = game_state.copy()

        for pid, hand in new_state.player_hands.items():
            if pid != player_id:
                number_of_cards = len(hand)
                for i in range(number_of_cards):
                    new_state.deck.append(hand[i])
                    new_state.player_hands[pid][i] = (-1, -1, 'unknown')

        # Shuffle the deck
        random.shuffle(new_state.deck)
        new_state.recompute_hands_deck_hash()

        return new_state

    @staticmethod
    def make_trade_possible_in_game_state(
        game_state: GameState,
        player_id: int,
        trade_type: str,
    ) -> GameState:
        """
        Make a trade possible in the game state by replacing 'unknown' cards with
        typed cards necessary for the trade.
        """
        new_state = game_state.copy()

        if (
            trade_type == 'infantry'
            or trade_type == 'cavalry'
            or trade_type == 'artillery'
        ):
            for i in range(3):
                # Replace the first 'unknown' card with the specified type
                new_state.player_hands[player_id][i] = (
                    -1,
                    -1,
                    trade_type,
                )
        elif trade_type == 'mixed':
            # For a mixed trade, replace the first three 'unknown' cards
            # with one each of infantry, cavalry, and artillery
            card_types = ['infantry', 'cavalry', 'artillery']
            for i in range(3):
                new_state.player_hands[player_id][i] = (
                    -1,
                    -1,
                    card_types[i],
                )

        new_state.recompute_hands_deck_hash()

        return new_state
