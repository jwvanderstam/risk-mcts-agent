import logging
import random

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.game_elements.action import (
    Action,
    EndPhaseAction,
)
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.mcts.config import MCTSConfig
from risk_agent.players.mcts.stopping import (
    IterationBasedStoppingCondition,
    StoppingCondition,
    TimeBasedStoppingCondition,
)
from risk_agent.players.mcts.tree import ConfidentMCTSTree, MCTSNode, MCTSTree
from risk_agent.players.player import Player
from risk_agent.utils.data_collector import DataCollector

logger = logging.getLogger(__name__)


class MCTSPlayer(Player):
    """
    A player that makes decisions using Monte Carlo Tree Search (MCTS).
    """

    def __init__(
        self,
        player_id: int,
        config: MCTSConfig,
        battle_computer: BattleComputer,
        data_collector: DataCollector | None = None,
    ) -> None:
        super().__init__(player_id)
        self.player_id = player_id
        self.config = config
        self.data_collector = data_collector
        self.battle_computer = battle_computer

        self.mcts_tree = MCTSTree(
            root_state=GameState(),
            player_id=player_id,
            config=config,
        )

        self.current_game_state: GameState | None = None

        stopping_conditions: dict[str, type[StoppingCondition]] = {
            'TimeBased': TimeBasedStoppingCondition,
            'IterationBased': IterationBasedStoppingCondition,
        }
        self.stopping_condition_type: type[StoppingCondition] = stopping_conditions.get(
            config.stopping_condition, TimeBasedStoppingCondition
        )

        if config.stopping_condition == 'TimeBased':
            self.stopping_condition_parameters = {
                'think_time': config.think_time,
                'selection_policy': config.selection_policy,
            }
        elif config.stopping_condition == 'IterationBased':
            self.stopping_condition_parameters = {
                'max_iterations': config.max_iterations,
                'selection_policy': config.selection_policy,
                'evaluative_policy': config.evaluative_policy,
            }
        else:
            raise ValueError('Invalid stopping condition.')

        # Initialize stopping condition
        self.stopping_condition: StoppingCondition = self.stopping_condition_type(
            **self.stopping_condition_parameters
        )

    def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Decide on an action to take based on the current game state and legal actions.
        """
        try:
            # Ensure MCTS tree is based on this exact game_state.
            if self.current_game_state is None or self.current_game_state != game_state:
                self.current_game_state = game_state
                self.mcts_tree.update_root(self.current_game_state)

            # Perform MCTS iterations untill the stopping_condition is met or a
            # suitable child is found by the evaluative_policy
            best_child: MCTSNode | None = None

            self.stopping_condition.start()
            while not self.stopping_condition.is_met() and best_child is None:
                self.mcts_tree.perform_iteration(self.battle_computer)
                best_child = self.mcts_tree.evaluative_policy()
                self.stopping_condition.update()
            if best_child is None:
                best_child = self.mcts_tree.selection_policy()

            if self.data_collector:
                self.data_collector.collect_player(
                    self.player_id,
                    self.process_data(
                        turn=game_state.current_turn,
                        mcts_tree=self.mcts_tree,
                        root_node=self.mcts_tree.root,
                        best_child=best_child,
                    ),
                )

            return (
                best_child.action
                if best_child.action is not None
                else random.choice(legal_actions)
            )

        except Exception:
            logger.exception(
                f'[MCTSPlayer {self.player_id}]: Error during action decision'
            )
            return random.choice(legal_actions) if legal_actions else EndPhaseAction()

    def notify_game_state_update(self, game_state: GameState) -> None:
        """
        Sends the latest game state to the thinking process.
        Called by GameManager after any action.
        """
        self.mcts_tree.update_root(game_state)

    def process_data(
        self, turn: int, mcts_tree: MCTSTree, root_node: MCTSNode, best_child: MCTSNode
    ) -> dict:
        """
        Process the data collected during the MCTS search.
        """
        return {
            'turn': turn,
            'root_visit_count': root_node.visit_count,
            'root_children_count': len(root_node.children),
            'root_scores': root_node.scores,
            'selected_action': best_child.action.__class__.__name__,
            'best_child_visit_count': best_child.visit_count,
            'best_child_scores': best_child.scores,
            'best_child_children_count': len(best_child.children),
            'max_depth_reached': mcts_tree.max_depth_reached,
        }


class ConfidentMCTSPlayer(MCTSPlayer):
    """
    A variant of MCTSPlayer that uses multiple trees for evaluating alliances.
    Works with either ConfidentPlayout or CondidentSelection policies.
    """

    def __init__(
        self,
        player_id: int,
        config: MCTSConfig,
        battle_computer: BattleComputer,
        data_collector: DataCollector | None = None,
    ) -> None:
        super().__init__(player_id, config, battle_computer, data_collector)
        self.mcts_trees: dict[int, MCTSTree] = {}
        self.mcts_tree = None  # type: ignore

        # Initialize MCTS trees for all players
        for alliance_player_id in range(config.number_of_players):
            self.mcts_trees[alliance_player_id] = ConfidentMCTSTree(
                root_state=GameState(),
                player_id=player_id,
                config=config,
                alliance_player_id=alliance_player_id,
            )

        logger.info(
            f'[ConfidentMCTSPlayer {self.player_id}]: Initialized with'
            f' {len(self.mcts_trees)} MCTS trees for alliances.'
        )

    def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Decide on an action to take based on the current game state and legal actions.
        Modified to iterate over multiple MCTS trees for different alliances.
        """
        try:
            # Ensure all MCTS trees are based on this exact game_state.
            if self.current_game_state is None or self.current_game_state != game_state:
                self.current_game_state = game_state
                for tree in self.mcts_trees.values():
                    tree.update_root(game_state)

            # Perform MCTS iterations untill the stopping_condition is met or a
            # suitable child is found by the evaluative_policy
            self.stopping_condition.start()
            while not self.stopping_condition.is_met():
                for tree in self.mcts_trees.values():
                    tree.perform_iteration(self.battle_computer)
                self.stopping_condition.update()

            # Select a child from the best tree based on the highest win rate for the player in the root
            logger.info(
                f'[ConfidentMCTSPlayer {self.player_id}]: Selecting action from best tree.'
                f' Win rates: {[t.root.scores[self.player_id] / t.root.visit_count for t in self.mcts_trees.values()]}'
            )
            best_tree = max(
                self.mcts_trees.values(),
                key=lambda t: t.root.scores[self.player_id] / t.root.visit_count,
            )
            best_child = best_tree.selection_policy()

            if self.data_collector:
                self.data_collector.collect_player(
                    self.player_id,
                    self.process_data(
                        turn=game_state.current_turn,
                        mcts_tree=best_tree,
                        root_node=best_tree.root,
                        best_child=best_child,
                    ),
                )

            return (
                best_child.action
                if best_child.action is not None
                else random.choice(legal_actions)
            )
        except Exception:
            logger.exception(
                f'[ConfidentMCTSPlayer {self.player_id}]: Error during action decision'
            )
            return random.choice(legal_actions) if legal_actions else EndPhaseAction()

    def notify_game_state_update(self, game_state: GameState) -> None:
        """
        Sends the latest game state to the thinking process.
        Called by GameManager after any action.
        """
        for tree in self.mcts_trees.values():
            tree.update_root(game_state)
