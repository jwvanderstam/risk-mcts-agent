import logging
import random
from collections.abc import Callable
from math import log, sqrt

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.engine.game_engine import GameEngine
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
from risk_agent.players.mcts.config import MCTSConfig

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    A node in the MCTS tree.
    Each node represents a game state and contains information about its children,
    parent, visit count, and total reward.
    """

    def __init__(self, state: GameState, action: Action | None = None) -> None:
        """
        Initialize the MCTS node with a game state.
        """
        self.state: GameState = state
        self.action: Action | None = (
            action  # Action that led to this state, if applicable
        )
        self.children: list[MCTSNode] = []
        self.parent: MCTSNode | None = None  # Parent node in the tree
        self.depth: int

        self.visit_count: int = 0
        self.scores: list[float] = [0.0] * state.number_of_players


class ChanceNode(MCTSNode):
    """
    A specialized MCTS node that represents a chance node in the tree.
    This node is used when the outcome of an action is probabilistic,
    such as in battles with multiple possible outcomes.
    """

    def __init__(self, state: GameState, action: Action | None = None) -> None:
        super().__init__(state, action)
        self.is_chance_node: bool = True  # Indicates this is a chance node
        self.child_probabilities: list[float] = []  # Probabilities for each child


class OutcomeNode(MCTSNode):
    """
    A specialized MCTS node that represents an outcome of a chance node.
    This node is used to store the result of a probabilistic action,
    such as the result of a battle.
    """

    def __init__(
        self,
        state: GameState,
        action: Action | None = None,
        outcome: tuple[int, int] | str | None = None,
    ) -> None:
        super().__init__(state, action)
        self.outcome = outcome  # The outcome of the action that led to this state


class MCTSTree:
    """
    The Monte Carlo Tree Search tree structure.
    This class includes all the necessary methods for MCTS operations,
    including expansion, simulation, backpropagation, and action selection.
    """

    def __init__(
        self,
        root_state: GameState,
        player_id: int,
        config: MCTSConfig,
    ) -> None:
        """
        Initialize the MCTS tree with a root state and player ID.
        """
        self.root: MCTSNode = MCTSNode(state=root_state)
        self.player_id: int = player_id  # ID of the player using this MCTS tree

        self.C: float = config.C

        search_policies = {
            'max^n': self._max_n_search_policy,
            'Paranoid': self._paranoid_search_policy,
        }
        self.search_policy: Callable[[float, MCTSNode], MCTSNode] = search_policies.get(
            config.search_policy, self._max_n_search_policy
        )

        playout_policies = {
            'Random': self._random_playout,
            'Heuristic': self._heuristic_playout,
        }
        self.playout_policy: Callable[[list[Action], GameState], Action] = (
            playout_policies.get(config.playout_policy, self._random_playout)
        )

        self.reinforce_all_heuristic: bool = config.reinforce_all_heuristic
        self.fortify_all_heuristic: bool = config.fortify_all_heuristic
        self.gamma: float = config.gamma

        selection_policies: dict[str, Callable[[], MCTSNode]] = {
            'MaxChild': self._max_child_selection,
            'RobustChild': self._robust_child_selection,
        }
        self.selection_policy: Callable[[], MCTSNode] = selection_policies.get(
            config.selection_policy,
            self._max_child_selection,
        )

        evaluative_policies: dict[str, Callable[[], MCTSNode | None]] = {
            'Dummy': self._dummy_child_selection,
            'MaxRobustChild': self._max_robust_child_selection,
            'MaxChildPolicyConvergence': self._max_child_policy_convergence,
            'RobustChildPolicyConvergence': self._robust_child_policy_convergence,
        }
        self.evaluative_policy: Callable[[], MCTSNode | None] = evaluative_policies.get(
            config.evaluative_policy,
            self._dummy_child_selection,
        )

        if (
            config.evaluative_policy == 'MaxChildPolicyConvergence'
            or config.evaluative_policy == 'RobustChildPolicyConvergence'
        ):
            self.policy_convergence_window = config.policy_convergence_window
            self.policy_convergence_i = 0
            self.policy_convergence_previous_child: MCTSNode | None = None

        # Track maximum depth reached during search
        self.max_depth_reached: int = 0

        self.trade_probabilities = {
            3: {
                'infantry': 564 / 13_244,  # = 4.26%
                'cavalry': 564 / 13_244,  # = 4.26%
                'artillery': 564 / 13_244,  # = 4.26%
                'mixed': 3_962 / 13_244,  # = 29.91%
                'no_trade': 1 - (564 + 564 + 564 + 3_962) / 13_244,  # = 57.31%
            },
            4: {
                'infantry': 17_017 / 135_751,
                'cavalry': 17_017 / 135_751,
                'artillery': 17_017 / 135_751,
                'mixed': 75_153 / 135_751,
                'no_trade': 1 - (17_017 + 17_017 + 17_017 + 75_153) / 135_751,
            },
            5: {
                'infantry': 258_804 / 1_086_008,
                'cavalry': 258_804 / 1_086_008,
                'artillery': 258_804 / 1_086_008,
                'mixed': 1 - (258_804 + 258_804 + 258_804) / 1_086_008,
            },
            6: {
                'infantry': 1 / 3 * 1_142_232 / 7_059_052,
                'cavalry': 1 / 3 * 1_142_232 / 7_059_052,
                'artillery': 1 / 3 * 1_142_232 / 7_059_052,
                'mixed': (7_059_052 - 1_142_232) / 7_059_052,
            },
            7: {
                'infantry': 1 / 3 * 3_570_138 / 38_320_568,
                'cavalry': 1 / 3 * 3_570_138 / 38_320_568,
                'artillery': 1 / 3 * 3_570_138 / 38_320_568,
                'mixed': (38_320_568 - 3_570_138) / 38_320_568,
            },
            8: {
                'infantry': 1 / 3 * 9_394_242 / 185_299_635,
                'cavalry': 1 / 3 * 9_394_242 / 185_299_635,
                'artillery': 1 / 3 * 9_394_242 / 185_299_635,
                'mixed': (185_299_635 - 9_394_242) / 185_299_635,
            },
            9: {
                'infantry': 1 / 3 * 20_738_718 / 812_242_512,
                'cavalry': 1 / 3 * 20_738_718 / 812_242_512,
                'artillery': 1 / 3 * 20_738_718 / 812_242_512,
                'mixed': (812_242_512 - 20_738_718) / 812_242_512,
            },
        }

    def update_root(self, new_game_state: GameState) -> None:
        """
        Update the root node of the tree with a new game state.
        """
        # Reset maximum depth reached for the new root
        self.max_depth_reached = 0

        # Look for a direct action child (non-chance) matching new state
        for child in self.root.children:
            if not isinstance(child, ChanceNode) and child.state == new_game_state:
                self.root = child
                self.root.parent = None  # Reset parent to None for new root
                self.root.depth = 0  # Reset depth for new root

                logger.info(
                    f'[MCTSTree] Successfully updated root to existing action node'
                    f', visits: {child.visit_count}'
                )
                return

        # Look for matching outcome under any chance nodes
        for chance in self.root.children:
            if isinstance(chance, ChanceNode):
                for outcome in chance.children:
                    if outcome.state == new_game_state:
                        # Promote the outcome node as new root
                        self.root = outcome
                        logger.info(
                            f'[MCTSTree] Updated root via outcome node'
                            f', visits: {outcome.visit_count}'
                        )
                        return

        # If the new state is not a child, create a new root node
        logger.info(
            '[MCTSTree] No matching child found for new game state. '
            'Creating a new root node.'
        )
        self.root = MCTSNode(state=new_game_state)

    def perform_iteration(self, battle_computer: BattleComputer) -> None:
        """
        Perform a single MCTS iteration, including selection, expansion,
        simulation, and backpropagation.
        """
        # Selection
        node, depth = self.selection(self.root, self.C)
        logger.info(
            f'[MCTSTree] Selected node for expansion:, '
            f'Visit count: {node.visit_count}, '
            f'Scores: {node.scores}, '
            f'Depth: {depth}, '
        )
        self.max_depth_reached = max(self.max_depth_reached, depth)

        # Determine winner via terminal check or playout simulation
        # Determine leaf and perform simulation
        if node.state.is_terminal():
            node_to_simulate = node
        else:
            # Expansion
            node_to_simulate = self.expansion(
                node,
                battle_computer,
            )

        logger.info(f'[MCTSTree] Expanded node with {len(node.children)} children.')

        if isinstance(node, ChanceNode):
            logger.info(
                '[MCTSTree] Node is a chance node, number of probabilities: '
                f'{len(node.child_probabilities)}'
            )

        logger.info(f'[MCTSTree] Selected node type: {type(node_to_simulate).__name__}')

        # Simulation
        rewards = self.playout(node_to_simulate.state, battle_computer, self.gamma)

        logger.info(f'[MCTSTree] Simulation rewards for node: {rewards}')

        # Backpropagation
        self.backpropagation(node_to_simulate, rewards)

        logger.info(
            f'[MCTSTree] Backpropagation complete. Root scores: {self.root.scores}'
        )

    def selection(self, node: MCTSNode, C: float) -> tuple[MCTSNode, int]:  # noqa: N803
        """
        Select a node to expand with the UCT (Upper Confidence Bound for Trees) formula.
        """
        depth: int = 0
        while node.children:
            if isinstance(node, ChanceNode):
                logger.info(
                    '[MCTSTree] Selecting child from chance node based on probabilities.'
                )
                node = random.choices(
                    node.children,
                    weights=node.child_probabilities,
                    k=1,
                )[0]
            elif any(child.visit_count == 0 for child in node.children):
                logger.info('[MCTSTree] Selecting unvisited child.')
                for child in node.children:
                    if child.visit_count == 0:
                        node = child
                        break
            else:
                logger.info('[MCTSTree] Selecting best child using search policy.')
                node = self.search_policy(C, node)

            depth += 1
        return node, depth

    def expansion(
        self,
        node: MCTSNode,
        battle_computer: BattleComputer,
    ) -> MCTSNode:
        """
        Expand the node by generating all possible actions from the current state
        and creating child nodes for each action.
        """
        # If the node is terminal return it
        if node.state.is_terminal():
            return node

        actions = GameEngine.get_valid_actions(node.state)

        if (
            self.reinforce_all_heuristic
            and node.state.current_turn_phase == 'reinforce'
            and len(actions) > 1
        ):
            max_armies = max(
                action.armies
                for action in actions
                if isinstance(action, ReinforceAction)
            )
            actions = [
                action
                for action in actions
                if isinstance(action, ReinforceAction) and action.armies == max_armies
            ]

        if (
            self.fortify_all_heuristic
            and node.state.current_turn_phase == 'fortify'
            and len(actions) > 1
        ):
            max_armies = max(
                action.armies for action in actions if isinstance(action, FortifyAction)
            )
            actions = [
                action
                for action in actions
                if isinstance(action, FortifyAction) and action.armies == max_armies
            ]

        child_to_return = None

        for action in actions:
            if isinstance(action, AttackAction):
                # If the action is an AttackAction, the following node will be a
                # chance node, so we create it and do not yet apply the action
                # But we create outcome nodes for all possible outcomes
                chance_node = ChanceNode(state=node.state.copy(), action=action)
                chance_node.parent = node
                node.children.append(chance_node)

                outcomes = battle_computer.get_all_outcomes(
                    attacking_armies=action.attacking_armies,
                    defending_armies=action.defending_armies,
                )

                for prob, outcome in outcomes:
                    outcome_state = action.apply_outcome(node.state.copy(), outcome)
                    outcome_node = OutcomeNode(
                        state=outcome_state,
                        action=action,
                        outcome=outcome,
                    )
                    outcome_node.parent = chance_node
                    chance_node.children.append(outcome_node)
                    chance_node.child_probabilities.append(prob)

                child_to_return = chance_node.children[0]
            elif (
                isinstance(action, EndTurnAction)
                and node.state.current_player == self.player_id
            ):
                # In case of the EndTurnAction for the player, if a territory was conquered,
                # we need to handle all three cases of the types of card that could
                # be awarded to the player, so we add a chance node with all possible
                # outcomes (card types)
                logger.info(
                    '[MCTSTree] Expanding EndTurnAction for player actual player.'
                )
                chance_node = ChanceNode(state=node.state.copy(), action=action)
                chance_node.parent = node
                node.children.append(chance_node)

                card_types = {
                    'infantry': 14 / 44,
                    'cavalry': 14 / 44,
                    'artillery': 14 / 44,
                    'wild': 2 / 44,
                }
                for card_type in card_types.keys():
                    # Add an unknown card with this type to the player's hand
                    card = (-1, -1, card_type)
                    outcome_state = action.apply(node.state.copy(), card)
                    outcome_node = OutcomeNode(
                        state=outcome_state,
                        action=action,
                        outcome=card_type,
                    )
                    outcome_node.parent = chance_node
                    chance_node.children.append(outcome_node)
                    chance_node.child_probabilities.append(card_types[card_type])

                    # For each of the outcome nodes, we also need to create the next
                    # chance node as the next player can have trade actions based
                    # on certain probabilities
                    child_to_return = self._generate_trade_option_nodes(outcome_node)

                if not child_to_return:
                    child_to_return = chance_node.children[0]
            elif (
                isinstance(action, EndTurnAction)
                and node.state.current_player != self.player_id
                and node.state.conquered_territory_this_turn
            ):
                logger.info('[MCTSTree] Expanding EndTurnAction for another player.')
                # If the action is an EndTurnAction for another player, we can apply it
                # directly with a unknown card added to the player's hand
                card = (-1, -1, 'unknown')
                new_state = action.apply(node.state.copy(), card)
                new_node = MCTSNode(state=new_state, action=action)
                new_node.parent = node
                node.children.append(new_node)

                # And we also need to create the next chance node as the next player
                # can have trade actions based on certain probabilities
                child_to_return = self._generate_trade_option_nodes(new_node)

                if not child_to_return:
                    child_to_return = new_node
            else:
                # For all other actions, just apply the action directly
                new_state = action.apply(node.state.copy())
                new_node = MCTSNode(state=new_state, action=action)
                new_node.parent = node
                node.children.append(new_node)

                # If this is the first child created, set it to return
                if child_to_return is None:
                    child_to_return = new_node

        # If no children were created, return the current node
        if not node.children:
            logger.warning(
                '[MCTSTree] No children created for node. Returning current node.'
            )
            return node

        if not child_to_return:
            # If no specific child was selected, return the first child
            child_to_return = node.children[0]

        # Then return an arbitrary child node to start the simulation
        return child_to_return

    def playout(
        self,
        state: GameState,
        battle_computer: BattleComputer,
        gamma: float,
    ) -> list[float]:
        """
        Simulate a random playout from the current state until a terminal state is reached.
        Return the reward for the player.
        """
        current_state = state.copy()
        playout_start_turn = current_state.current_turn

        # Randomly play out the game until a terminal state is reached
        while not current_state.is_terminal():
            # check if the current player is not defeated
            if current_state.current_player in current_state.defeated_players:
                current_state = GameEngine.apply_action(current_state, EndTurnAction())
            else:
                if current_state.current_turn_phase == 'trade_cards':
                    # If the current turn phase is 'trade_cards', handle trading via the
                    # trade probabilities
                    number_of_cards = len(
                        current_state.player_hands[current_state.current_player]
                    )
                    if number_of_cards in self.trade_probabilities:
                        trade_type = random.choices(
                            list(self.trade_probabilities[number_of_cards].keys()),
                            weights=list(
                                self.trade_probabilities[number_of_cards].values()
                            ),
                            k=1,
                        )[0]
                        current_state = GameEngine.make_trade_possible_in_game_state(
                            current_state,
                            trade_type=trade_type,
                            player_id=current_state.current_player,
                        )

                legal_actions = GameEngine.get_valid_actions(current_state)
                if not legal_actions:
                    logger.warning(
                        '[MCTSTree] No legal actions available.'
                        f'Ending playout for state: {current_state}'
                    )
                    break

                # Use the playout policy to select an action
                action = self.playout_policy(legal_actions, current_state)

                if isinstance(action, AttackAction):
                    if (
                        action.attacking_armies > battle_computer.max_attacking_armies
                        or action.defending_armies
                        > battle_computer.max_defending_armies
                    ):
                        logger.warning(
                            '[MCTSTree] Attack action exceeds max armies. Returning 0.0 reward.'
                        )
                        return [0.0] * current_state.number_of_players

                    # Apply the attack action with battle computer logic
                    current_state = action.apply(current_state, battle_computer)

                    # If the attack action led to defeating a player and more than
                    # four cards for the current player, trade in cards for reinforcements
                    if (
                        len(current_state.player_hands[current_state.current_player])
                        > 4
                    ):
                        current_state.current_turn_phase = 'trade_cards'
                else:
                    # For non-attack actions, just apply the action directly
                    current_state = action.apply(current_state)

        logger.info(
            f'[MCTSTree] Winner determined as {current_state.determine_winner()} '
            f'in playout of length {current_state.current_turn - playout_start_turn} turns.'
        )

        # Calculate the reward for the player
        winner = current_state.determine_winner()
        return [
            (gamma ** (current_state.current_turn - playout_start_turn))
            if i == winner
            else 0.0
            for i in range(state.number_of_players)
        ]

    def backpropagation(self, node: MCTSNode, rewards: list[float]) -> None:
        """
        Backpropagate the reward from the leaf node up to the root.
        Update the visit count and win rate for each node in the path.
        """
        while node:
            node.visit_count += 1
            node.scores = [node.scores[i] + rewards[i] for i in range(len(node.scores))]

            if node.parent is None:
                break

            node = node.parent

    def _generate_trade_option_nodes(
        self,
        node: MCTSNode,
    ) -> MCTSNode | None:
        """
        Generate trade option nodes for the given MCTS node.
        """
        child_to_return = None
        number_of_cards = len(node.state.player_hands[node.state.current_player])

        if number_of_cards in self.trade_probabilities:
            next_chance_node = ChanceNode(state=node.state.copy(), action=node.action)
            next_chance_node.parent = node
            node.children.append(next_chance_node)

            for trade_type, prob in self.trade_probabilities[number_of_cards].items():
                next_state = GameEngine.make_trade_possible_in_game_state(
                    node.state.copy(),
                    trade_type=trade_type,
                    player_id=node.state.current_player,
                )
                next_outcome_node = OutcomeNode(
                    state=next_state,
                    action=node.action,
                    outcome=trade_type,
                )
                next_outcome_node.parent = next_chance_node
                next_chance_node.children.append(next_outcome_node)
                next_chance_node.child_probabilities.append(prob)

                if child_to_return is None:
                    child_to_return = next_outcome_node

        return child_to_return

    def _random_playout(
        self, legal_actions: list[Action], game_state: GameState
    ) -> Action:
        """
        Select a random action from the legal actions.
        """
        return random.choice(legal_actions)

    def _heuristic_playout(
        self, legal_actions: list[Action], game_state: GameState
    ) -> Action:
        """
        Select an action based on basic heuristics.
        """
        # Implement a heuristic-based playout policy here
        # based on the current phase, we can make a simple heuristic decision
        match game_state.current_turn_phase:
            case 'trade_cards':
                # If a card trade with a value of >= 8 is available, take it
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, TradeCardsAction) and action.value >= 8
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'reinforce':
                # Otherwise, only reinforce to border territories
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, ReinforceAction)
                    and action.territory
                    in GameEngine.get_border_territories(game_state, self.player_id)
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'attack':
                # Try to attack only is it has a ratio of at least 1.5 attacking
                # armies to defending armies
                selected_actions = [
                    action
                    for action in legal_actions
                    if (
                        isinstance(action, AttackAction)
                        and action.attacking_armies >= 1.5 * action.defending_armies
                    )
                    or isinstance(action, EndPhaseAction)
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'fortify':
                # Try to fortify from a non-border territory to a border territory
                border_territories = GameEngine.get_border_territories(
                    game_state, self.player_id
                )
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, FortifyAction)
                    and action.from_territory not in border_territories
                    and action.to_territory in border_territories
                ]

                if selected_actions:
                    return random.choice(selected_actions)

        # If no specific action is selected, choose any legal action
        return random.choice(legal_actions)

    def _dummy_child_selection(self) -> None:
        return None

    def _max_child_selection(self) -> MCTSNode:
        return max(
            self.root.children,
            key=lambda child: (
                child.scores[self.player_id] / child.visit_count
                if child.visit_count > 0
                else float('-inf')
            ),
        )

    def _robust_child_selection(self) -> MCTSNode:
        return max(self.root.children, key=lambda child: child.visit_count)

    def _max_robust_child_selection(self) -> MCTSNode | None:
        if not self.root.children or any(
            child.visit_count == 0 for child in self.root.children
        ):
            return None

        max_child = self._max_child_selection()
        robust_child = self._robust_child_selection()

        if max_child == robust_child:
            return max_child
        return None

    def _max_child_policy_convergence(self) -> MCTSNode | None:
        if not self.root.children or any(
            child.visit_count == 0 for child in self.root.children
        ):
            return None

        max_child = self._max_child_selection()

        if self.policy_convergence_previous_child is None:
            self.policy_convergence_previous_child = max_child
            self.policy_convergence_i = 1
            return None

        if max_child == self.policy_convergence_previous_child:
            self.policy_convergence_i += 1
            if self.policy_convergence_i >= self.policy_convergence_window:
                return max_child
        else:
            self.policy_convergence_i = 0

        return None

    def _robust_child_policy_convergence(self) -> MCTSNode | None:
        if not self.root.children or any(
            child.visit_count == 0 for child in self.root.children
        ):
            return None

        robust_child = self._robust_child_selection()

        if self.policy_convergence_previous_child is None:
            self.policy_convergence_previous_child = robust_child
            self.policy_convergence_i = 1
            return None

        if robust_child == self.policy_convergence_previous_child:
            self.policy_convergence_i += 1
            if self.policy_convergence_i >= self.policy_convergence_window:
                return robust_child
        else:
            self.policy_convergence_i = 0

        return None

    def _max_n_search_policy(self, C: float, node: MCTSNode) -> MCTSNode:  # noqa: N803
        return max(
            node.children,
            key=lambda child: (
                child.scores[node.state.current_player] / child.visit_count
            )
            + C * sqrt(log(node.visit_count) / child.visit_count),
        )

    def _paranoid_search_policy(self, C: float, node: MCTSNode) -> MCTSNode:  # noqa: N803
        if self.root.state.current_player == node.state.current_player:
            return max(
                node.children,
                key=lambda child: (
                    child.scores[node.state.current_player] / child.visit_count
                )
                + C * sqrt(log(node.visit_count) / child.visit_count),
            )
        else:
            return max(
                node.children,
                key=lambda child: (
                    (
                        1
                        - (
                            child.scores[self.root.state.current_player]
                            / child.visit_count
                        )
                    )
                    + C * sqrt(log(node.visit_count) / child.visit_count)
                ),
            )


class ConfidentMCTSTree(MCTSTree):
    """
    An MCTS tree that can handle alliances either via search policy or playout policy.
    """

    def __init__(
        self,
        root_state: GameState,
        player_id: int,
        config: MCTSConfig,
        alliance_player_id: int,
    ) -> None:
        super().__init__(root_state, player_id, config)
        self.alliance_player_id = alliance_player_id

        # Override either search or playout policy based on config
        if config.search_policy == 'Confident':
            self.search_policy = self._confident_search_policy
        elif config.playout_policy == 'Confident':
            self.playout_policy = self._confident_playout
        else:
            raise ValueError(
                'ConfidentMCTSTree requires either search_policy or '
                'playout_policy to be "Confident".'
            )

    def _confident_playout(
        self, legal_actions: list[Action], game_state: GameState
    ) -> Action:
        """
        Select an action based on a confident playout policy that considers alliances.
        """
        selected_actions: list[Action] = []

        # If the current player is the alliance player, prefer actions that benefit both
        if (
            game_state.current_player == self.player_id
            and len(game_state.defeated_players) != game_state.number_of_players - 2
        ):
            # Do not attack the alliance player unless necessary
            if game_state.current_turn_phase == 'attack':
                selected_actions = [
                    action
                    for action in legal_actions
                    if not (
                        isinstance(action, AttackAction)
                        and game_state.owner[action.to_territory]
                        == self.alliance_player_id
                    )
                ]
        elif (
            game_state.current_player == self.alliance_player_id
            and len(game_state.defeated_players) != game_state.number_of_players - 2
        ):
            # Do not attack the main player unless necessary
            if game_state.current_turn_phase == 'attack':
                selected_actions = [
                    action
                    for action in legal_actions
                    if not (
                        isinstance(action, AttackAction)
                        and game_state.owner[action.to_territory] == self.player_id
                    )
                ]

        if selected_actions:
            return random.choice(selected_actions)

        # Fallback to random action if no specific action is selected
        return random.choice(legal_actions)

    def _confident_search_policy(self, C: float, node: MCTSNode) -> MCTSNode:  # noqa: N803
        # If the current player or the alliance player is to move, maximize their combined score
        if node.state.current_player in [
            self.player_id,
            self.alliance_player_id,
        ] and not (self.player_id == self.alliance_player_id):
            logger.info(
                '[ConfidentMCTSTree] Using confident search policy for player or alliance.'
            )
            return max(
                node.children,
                key=lambda child: (
                    (child.scores[self.player_id] / child.visit_count)
                    + (child.scores[self.alliance_player_id] / child.visit_count) / 2
                )
                + C * sqrt(log(node.visit_count) / child.visit_count),
            )
        else:
            # Otherwise, revert to max^n search policy
            logger.info('[ConfidentMCTSTree] Reverting to max^n search policy.')
            return self._max_n_search_policy(C, node)
