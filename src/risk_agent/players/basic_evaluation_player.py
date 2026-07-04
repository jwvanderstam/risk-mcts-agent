import logging
import math
import random

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.engine.game_engine import GameEngine
from risk_agent.game_elements.action import Action, AttackAction, TradeCardsAction
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.player import Player

logger = logging.getLogger(__name__)


class BasicEvaluationPlayer(Player):
    """
    A player based on the basic evaluation player by Wolf (2005).
    """

    def __init__(self, player_id: int, battle_computer: 'BattleComputer') -> None:
        super().__init__(player_id)
        self.player_id = player_id
        self.battle_computer = battle_computer

    def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Decide on an action to take based on the current game state and legal actions.
        """
        best_evaluation = float('-inf')
        best_action: Action | None = None
        equal_evaluations = 0

        for action in legal_actions:
            evaluation = self.evaluate_action(game_state, action)

            logger.info(f'Evaluating action {action} with evaluation {evaluation}')

            # If the evaluation is better than the best found so far, update
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_action = action
                equal_evaluations = 1
            elif evaluation == best_evaluation:
                equal_evaluations += 1
                # override the action with a an equable evaluation
                if random.random() < 1 / equal_evaluations:
                    best_action = action

        return best_action if best_action is not None else random.choice(legal_actions)

    def evaluate_action(self, game_state: GameState, action: Action) -> float:
        """
        Evaluate the action based on the basic evaluation strategy.
        This is a placeholder for the actual evaluation logic.
        """
        # If we are in the trade_cards phase, use the trade evaluation function or return 0 for the EndPhaseAction
        if game_state.current_turn_phase == 'trade_cards':
            if isinstance(action, TradeCardsAction):
                return self.evaluate_trade_action(game_state, action)
            else:
                return 0.0

        # Otherwise, based on the type of action, use the appropriate evaluation method
        if isinstance(action, AttackAction):
            return self.evaluate_attack_action(game_state, action)
        else:
            return self.evaluate_generic_action(game_state, action)

    def evaluate_generic_action(self, game_state: GameState, action: Action) -> float:
        """
        Evaluate a generic action (e.g., ReinforceAction, FortifyAction, TradeCardsAction).
        """
        new_game_state = action.apply(game_state)
        # Evaluate the new game state after applying the action, but based on the current turn phase
        return self.evaluate_game_state(
            new_game_state, self.player_id, game_state.current_turn_phase
        )

    def evaluate_trade_action(
        self, game_state: GameState, action: TradeCardsAction
    ) -> float:
        """
        Evaluate a trade action based on the number of cards traded.
        """
        new_game_state = action.apply(game_state)

        evaluation = 0.0

        evaluation += self.occupied_territories_trade_feature(
            new_game_state, self.player_id
        )
        evaluation += self.trade_value_trade_feature(action)

        return evaluation

    def evaluate_attack_action(
        self, game_state: GameState, action: AttackAction
    ) -> float:
        """
        Evaluate an attack action based on the battle evaluater from the paper.
        """
        outcomes = self.battle_computer.get_all_outcomes(
            action.attacking_armies, action.defending_armies
        )

        attacker_win_prob = self.battle_computer.get_attacker_win_rate(
            action.attacking_armies, action.defending_armies
        )
        defender_win_prob = 1 - attacker_win_prob

        expected_attackers_left = 0.0
        expected_defenders_left = 0.0

        for probability, (final_attackers, final_defenders) in outcomes:
            if final_defenders == 0 and final_attackers > 0:
                expected_attackers_left += final_attackers * (
                    probability / attacker_win_prob
                )
            else:
                expected_defenders_left += final_defenders * (
                    probability / defender_win_prob
                )

        attacker_win_evaluation = self.evaluate_game_state(
            action.apply_outcome(game_state, (int(expected_attackers_left), 0)),
            self.player_id,
            'attack',
        )
        defender_win_evaluation = self.evaluate_game_state(
            action.apply_outcome(game_state, (0, int(expected_defenders_left))),
            self.player_id,
            'attack',
        )

        return (
            attacker_win_prob * attacker_win_evaluation
            + defender_win_prob * defender_win_evaluation
        )

    def evaluate_game_state(
        self, game_state: GameState, player_id: int, turn_phase: str
    ) -> float:
        """
        Evaluate the game state for a specific player.
        This is a placeholder for the actual evaluation logic.
        """
        evaluation = 0.0

        # always
        evaluation += 0.10 * self.continent_safety_feature(game_state, player_id)
        evaluation += 0.23 * self.continent_threat_feature(game_state, player_id)
        evaluation += 0.02 * self.more_than_one_army_feature(game_state, player_id)

        # not in reinforcement and fortification phase
        if turn_phase != 'reinforce' and turn_phase != 'fortify':
            evaluation += 0.52 * self.armies_feature(game_state, player_id)
            evaluation += 0.53 * self.best_enemy_feature(game_state, player_id)
            evaluation += 0.26 * self.enemy_estimated_reinforcements_feature(
                game_state, player_id
            )
            evaluation += 0.30 * self.enemy_occupied_continents_feature(
                game_state, player_id
            )
            evaluation += 0.28 * self.hinterland_feature(game_state, player_id)
            evaluation += 0.19 * self.occupied_territories_feature(
                game_state, player_id
            )
            evaluation += 0.36 * self.own_estimated_reinforcements_feature(
                game_state, player_id
            )
            evaluation += 0.57 * self.own_occupied_continents_feature(
                game_state, player_id
            )
            evaluation += 0.03 * self.own_occupied_risk_card_territories_feature(
                game_state, player_id
            )
            evaluation += 0.05 * self.risk_cards_feature(game_state, player_id)
            evaluation += 0.5 * self.victory_proximity_feature(game_state, player_id)

        # not in attack phase
        if turn_phase != 'attack':
            evaluation += 1.6 * self.distance_to_frontier_feature(game_state, player_id)
            evaluation += 0.04 * self.maximum_threat_feature(game_state, player_id)

        return evaluation

    def threat(self, game_state: GameState, territory_id: int) -> float:
        """
        Calculate the maximum threat of any enemy player to a specific territory.
        """
        max_threat = 0.0
        player_id = game_state.owner[territory_id]
        defending_armies = game_state.armies[territory_id]

        for neighbor_id in game_state.board.adjacency_list[territory_id]:
            if game_state.owner[neighbor_id] == player_id:
                continue

            enemy_armies = game_state.armies[neighbor_id]

            threat = self.battle_computer.get_attacker_win_rate(
                attacking_armies=enemy_armies,
                defending_armies=defending_armies,
            )

            if threat > max_threat:
                max_threat = threat

        return max_threat

    def distance_to_nearest_enemy(
        self, game_state: GameState, territory_id: int
    ) -> int:
        """
        Calculate the distance to the nearest enemy territory.
        """
        player_id = game_state.owner[territory_id]
        visited = set()
        queue = [(territory_id, 0)]

        while queue:
            current_territory, distance = queue.pop(0)

            if current_territory in visited:
                continue

            visited.add(current_territory)

            for neighbor_id in game_state.board.adjacency_list[current_territory]:
                if game_state.owner[neighbor_id] != player_id:
                    return distance + 1

                queue.append((neighbor_id, distance + 1))

        return 10  # Arbitrary large distance if no enemy found

    def continent_rating(self, game_state: GameState) -> dict[int, float]:
        """
        Calculate the rating of each continent (constant value for each continent).
        """
        continent_ratings = {}

        for continent_id, continent in game_state.board.continents.items():
            continent_rating = 15.0
            continent_rating += continent['bonus']

            # calculate the number of borders for the continent
            borders = 0
            for territory_id in continent['territories']:
                for neighbor_id in game_state.board.adjacency_list[territory_id]:
                    if neighbor_id not in continent['territories']:
                        borders += 1

            continent_rating -= 4 * borders

            continent_rating /= len(continent['territories'])

            continent_ratings[continent_id] = continent_rating

        return continent_ratings

    def armies_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: The number of armies the player has relative to the total number of armies.
        """
        player_armies = 0
        total_armies = 0
        for territory_id in game_state.territory_ids:
            if game_state.owner[territory_id] == player_id:
                player_armies += game_state.armies[territory_id]
            total_armies += game_state.armies[territory_id]

        return player_armies / total_armies if total_armies > 0 else 0.0

    def best_enemy_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Negative measure of the power of the best enemy player.
        """
        best_enemy_rating = 0

        for enemy_id in range(game_state.number_of_players):
            if enemy_id == player_id:
                continue

            enemy_rating = self.armies_feature(game_state, enemy_id)

            enemy_territories = len(
                [
                    territory_id
                    for territory_id in game_state.territory_ids
                    if game_state.owner[territory_id] == enemy_id
                ]
            )
            enemy_rating += enemy_territories / len(game_state.board.territories)
            enemy_rating /= 2

            if enemy_rating > best_enemy_rating:
                best_enemy_rating = enemy_rating

        return -best_enemy_rating

    def _border_threat_stats(
        self, game_state: GameState, continent: dict, exclude_owner_id: int
    ) -> tuple[float, float]:
        """
        Sum of squared threats and the maximum single threat across all
        border territories of a continent, i.e. territories with a neighbor
        not owned by exclude_owner_id.
        """
        threat_sum = 0.0
        max_threat = 0.0
        for territory_id in continent['territories']:
            for neighbor_id in game_state.board.adjacency_list[territory_id]:
                if game_state.owner[neighbor_id] != exclude_owner_id:
                    threat = self.threat(game_state, territory_id)
                    threat_sum += threat**2
                    if threat > max_threat:
                        max_threat = threat
        return threat_sum, max_threat

    def _single_enemy_owner_of_continent(
        self, game_state: GameState, continent: dict, player_id: int
    ) -> int | None:
        """
        Return the enemy player id if the continent is entirely owned by a
        single player other than player_id, else None.
        """
        enemy_player_id = -1
        for territory_id in continent['territories']:
            owner = game_state.owner[territory_id]
            if owner == player_id:
                return None
            elif enemy_player_id == -1:
                enemy_player_id = owner
            elif owner != enemy_player_id:
                return None
        return enemy_player_id

    def continent_safety_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Negative measurement of the threat of an enemy player to the player's continents.
        """
        continent_safety_feature: float = 0.0
        continent_ratings = self.continent_rating(game_state)

        for continent_id, continent in game_state.board.continents.items():
            # check if the player owns the continent
            if all(
                game_state.owner[territory_id] == player_id
                for territory_id in continent['territories']
            ):
                threat_sum, max_threat = self._border_threat_stats(
                    game_state, continent, player_id
                )
                continent_safety_feature += (
                    threat_sum + max_threat
                ) * continent_ratings[continent_id]

        return -continent_safety_feature

    def continent_threat_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Measurement of the threat from to enemy continents.
        """
        continent_threat_feature: float = 0.0
        continent_ratings = self.continent_rating(game_state)

        for continent_id, continent in game_state.board.continents.items():
            enemy_player_id = self._single_enemy_owner_of_continent(
                game_state, continent, player_id
            )
            if enemy_player_id is not None:
                threat_sum, _ = self._border_threat_stats(
                    game_state, continent, enemy_player_id
                )
                continent_threat_feature += threat_sum * continent_ratings[continent_id]

        return continent_threat_feature

    def distance_to_frontier_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Measurement of the army distrobution troughout the player's territories.
        """
        player_armies: int = 0
        sum_term: int = 0

        for territory_id in game_state.territory_ids:
            if game_state.owner[territory_id] == player_id:
                player_armies += game_state.armies[territory_id]
                distance = self.distance_to_nearest_enemy(game_state, territory_id)
                sum_term += distance * game_state.armies[territory_id]

        return player_armies / sum_term if sum_term > 0 else 0.0

    def enemy_estimated_reinforcements_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Negative estimated reinforcements of the enemy players.
        """
        enemy_reinforcements: int = 0

        for enemy_id in range(game_state.number_of_players):
            if enemy_id == player_id:
                continue

            enemy_reinforcements += GameEngine.calculate_base_reinforcements(
                game_state, enemy_id
            )

        return -enemy_reinforcements

    def enemy_occupied_continents_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Number of continents occupied by enemy players.
        """
        enemy_continents: int = 0

        for continent_id, continent in game_state.board.continents.items():
            # check if any enemy player owns the continent
            owns_continent: bool = True
            enemy_player_id = -1
            for territory_id in continent['territories']:
                if game_state.owner[territory_id] == player_id:
                    owns_continent = False
                elif enemy_player_id == -1:
                    enemy_player_id = game_state.owner[territory_id]
                elif game_state.owner[territory_id] != enemy_player_id:
                    owns_continent = False
                    break

            # if an enemy player owns the continent, increase the count
            if owns_continent:
                enemy_continents += 1

        return -enemy_continents

    def hinterland_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Percentage of hinterland territories owned by the player.
        Hinterland territories are those that are not adjacent to any enemy territory.
        """
        player_territories = 0
        hinterland_territories = 0

        for territory_id in game_state.territory_ids:
            if game_state.owner[territory_id] == player_id:
                is_hinterland = True
                for neighbor_id in game_state.board.adjacency_list[territory_id]:
                    if game_state.owner[neighbor_id] != player_id:
                        is_hinterland = False
                        break
                if is_hinterland:
                    hinterland_territories += 1
                player_territories += 1

        return (
            hinterland_territories / player_territories
            if player_territories > 0
            else 0.0
        )

    def maximum_threat_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Measurement of the probability that the player will be able to
        successfully occupy an enemy territory during the next attacking phase.
        """
        max_threat = 0.0

        for territory_id in game_state.territory_ids:
            attacking_armies = game_state.armies[territory_id] - 1
            if game_state.owner[territory_id] == player_id:
                for neighbor_id in game_state.board.adjacency_list[territory_id]:
                    if game_state.owner[neighbor_id] != player_id:
                        threat = self.battle_computer.get_attacker_win_rate(
                            attacking_armies=attacking_armies,
                            defending_armies=game_state.armies[neighbor_id],
                        )

                        if threat > max_threat:
                            max_threat = threat

        return max_threat

    def more_than_one_army_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Percentage of territories owned by the player that have more than one army.
        """
        territories_with_more_than_one_army = 0
        total_territories = 0

        for territory_id in game_state.territory_ids:
            if game_state.owner[territory_id] == player_id:
                total_territories += 1
                if game_state.armies[territory_id] > 1:
                    territories_with_more_than_one_army += 1

        return (
            territories_with_more_than_one_army / total_territories
            if total_territories > 0
            else 0.0
        )

    def occupied_territories_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Percentage of territories owned by the player.
        """
        occupied_territories = 0
        total_territories = len(game_state.board.territories)

        for territory_id in game_state.territory_ids:
            if game_state.owner[territory_id] == player_id:
                occupied_territories += 1

        return (
            occupied_territories / total_territories if total_territories > 0 else 0.0
        )

    def own_estimated_reinforcements_feature(
        self, game_state: GameState, player_id: int
    ) -> int:
        """
        Feature: Estimated reinforcements of the player's territories.
        """
        own_reinforcements = GameEngine.calculate_base_reinforcements(
            game_state, player_id
        )
        return own_reinforcements

    def own_occupied_continents_feature(
        self, game_state: GameState, player_id: int
    ) -> int:
        """
        Feature: Number of continents occupied by the player.
        """
        own_continents = 0

        for continent_id, continent in game_state.board.continents.items():
            # check if the player owns the continent
            if all(
                game_state.owner[territory_id] == player_id
                for territory_id in continent['territories']
            ):
                own_continents += 1

        return own_continents

    def own_occupied_risk_card_territories_feature(
        self, game_state: GameState, player_id: int
    ) -> int:
        """
        Feature: Number of risk cards in the possesion of the player that correspond to
        territories owned by the player.
        """
        own_risk_card_territories = 0

        for card in game_state.player_hands[player_id]:
            card_territory_id = card[1]

            for territory_id in game_state.territory_ids:
                if (
                    game_state.owner[territory_id] == player_id
                    and territory_id == card_territory_id
                ):
                    own_risk_card_territories += 1
                    break

        return own_risk_card_territories

    def risk_cards_feature(self, game_state: GameState, player_id: int) -> int:
        """
        Feature: Number of risk cards in the possession of the player.
        """
        return (
            len(game_state.player_hands[player_id])
            if player_id in game_state.player_hands
            else 0
        )

    def occupied_territories_trade_feature(
        self, game_state: GameState, player_id: int
    ) -> float:
        """
        Feature: Percentage of territories owned by the player that are occupied.
        """
        return (
            self.own_occupied_risk_card_territories_feature(game_state, player_id) * 2
        )

    def trade_value_trade_feature(self, action: TradeCardsAction) -> float:
        """
        Feature: Value of the trade action based on the number of cards traded.
        """
        return -1 * (10 - action.value)

    def victory_proximity_feature(self, game_state: GameState, player_id: int) -> float:
        """
        Feature: Strongly rewards being close to winning the game.
        Kicks in when player controls 80%+ of territories.
        Uses exponential growth to heavily incentivize final conquests.
        """
        total_territories = len(game_state.board.territories)
        occupied_territories = sum(
            1
            for territory_id in game_state.territory_ids
            if game_state.owner[territory_id] == player_id
        )

        control_percentage = occupied_territories / total_territories

        # Only activate when controlling 80% or more of the board
        if control_percentage < 0.8:
            return 0.0

        # Exponential growth as player gets closer to victory
        # At 80%: ~0.0
        # At 90%: ~0.25
        # At 95%: ~1.0
        # At 100%: ~10.0 (massive reward for winning)

        # Normalize to [0, 1] range where 0.8 -> 0 and 1.0 -> 1
        normalized = (control_percentage - 0.8) / 0.2

        # Use exponential function: e^(5x) - 1 to create strong growth
        # This gives approximately:
        # 80% control: 0.0
        # 85% control: 0.28
        # 90% control: 1.12
        # 95% control: 4.48
        # 100% control: 147.4 (extremely high reward for victory)

        return math.exp(5 * normalized) - 1.0
