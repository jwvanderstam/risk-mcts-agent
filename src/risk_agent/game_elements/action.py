import random
from abc import ABC, abstractmethod

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.game_elements.game_state import GameState


class Action(ABC):
    """
    Abstract base class for all actions in the game.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialise the action.
        """
        pass

    @abstractmethod
    def apply(self, game_state: GameState) -> GameState:
        """
        Apply the action to the game state and return the new game state.
        """
        pass

    def __str__(self) -> str:
        return f'Action: {type(self).__name__}, with attributes {", ".join(f"{k}={v}" for k, v in vars(self).items())}'  # noqa: E501

    def __eq__(self, value: object) -> bool:
        # Check if the value is an instance of Action and has the same attributes
        if not isinstance(value, Action):
            return False
        return vars(self) == vars(value)


class EndPhaseAction(Action):
    """
    Action to end the current phase.
    """

    def __init__(self) -> None:
        """
        Initialise the end phase action.
        """
        super().__init__()

    def apply(self, game_state: GameState) -> GameState:
        """
        Apply the end phase action to the game state.
        """
        new_game_state = game_state.copy()
        match game_state.current_turn_phase:
            case 'trade_cards':
                new_game_state.set_scalar('current_turn_phase', 'reinforce')
            case 'reinforce':
                new_game_state.set_scalar('current_turn_phase', 'attack')
            case 'attack':
                new_game_state.set_scalar('current_turn_phase', 'fortify')

        return new_game_state


class EndTurnAction(Action):
    """
    Action to end the current turn.
    """

    def __init__(self) -> None:
        """
        Initialise the end turn action.
        """
        super().__init__()

    def apply(
        self, game_state: GameState, card: tuple[int, int, str] | None = None
    ) -> GameState:
        """
        Apply the end turn action to the game state.
        """
        new_game_state = game_state.copy()

        if new_game_state.conquered_territory_this_turn:
            if card is not None:
                new_game_state.player_hands[new_game_state.current_player].append(card)
                if card in new_game_state.deck:
                    new_game_state.deck.remove(card)
            else:
                card = new_game_state.deck.pop(
                    random.choice(range(len(new_game_state.deck)))
                )
                new_game_state.player_hands[new_game_state.current_player].append(card)
            new_game_state.recompute_hands_deck_hash()

        new_game_state.set_scalar('current_turn_phase', 'trade_cards')
        new_game_state.set_scalar('current_turn', new_game_state.current_turn + 1)
        if new_game_state.current_turn % new_game_state.number_of_players == 0:
            new_game_state.set_scalar('current_round', new_game_state.current_round + 1)
        new_game_state.set_scalar(
            'current_player',
            (new_game_state.current_player + 1) % new_game_state.number_of_players,
        )

        new_game_state.set_scalar('conquered_territory_this_turn', False)
        new_game_state.set_scalar('reinforcements_this_turn', 0)
        new_game_state.set_scalar('card_trade_in_this_turn', -1)
        new_game_state.set_scalar('fortified_territory_this_turn', False)

        # Import GameEngine here to avoid circular imports
        from risk_agent.engine.game_engine import GameEngine

        new_game_state.set_scalar(
            'base_reinforcements_this_turn',
            GameEngine.calculate_base_reinforcements(
                new_game_state, new_game_state.current_player
            ),
        )

        return new_game_state


class TradeCardsAction(Action):
    """
    Action to trade cards.
    """

    def __init__(
        self, player_id: int, cards: list[tuple[int, int, str]], value: int
    ) -> None:
        """
        Initialise the trade cards action with the player ID and cards to trade.
        """
        super().__init__()
        self.player_id = player_id
        self.cards = cards
        self.value = value

    def apply(self, game_state: GameState) -> GameState:
        """
        Apply the trade cards action to the game state.
        """
        new_game_state = game_state.copy()

        if game_state.card_trade_in_this_turn != -1:
            new_game_state.set_scalar(
                'card_trade_in_this_turn',
                new_game_state.card_trade_in_this_turn + self.value,
            )
        else:
            new_game_state.set_scalar('card_trade_in_this_turn', self.value)

        added_extra_armies = False
        for card in self.cards:
            if not added_extra_armies:
                for territory in new_game_state.territory_ids:
                    if (
                        new_game_state.owner[territory] == self.player_id
                        and card[1] == territory
                    ):
                        new_game_state.add_armies(territory, 2)
                        added_extra_armies = True

        for card in self.cards:
            new_game_state.deck.append(card)

        for card in self.cards:
            new_game_state.player_hands[self.player_id].remove(card)

        random.shuffle(new_game_state.deck)
        new_game_state.recompute_hands_deck_hash()

        return new_game_state


class ReinforceAction(Action):
    """
    Action to reinforce a territory.
    """

    def __init__(self, territory: int, armies: int) -> None:
        """
        Initialise the reinforce action with the territory and number of armies.
        """
        super().__init__()
        self.territory = territory
        self.armies = armies

    def apply(self, game_state: GameState) -> GameState:
        """
        Apply the reinforce action to the game state.
        """
        new_game_state = game_state.copy()
        new_game_state.set_scalar(
            'reinforcements_this_turn',
            new_game_state.reinforcements_this_turn + self.armies,
        )
        new_game_state.add_armies(self.territory, self.armies)
        return new_game_state


class AttackAction(Action):
    """
    Action of attacking a territory.
    """

    def __init__(
        self,
        from_territory: int,
        to_territory: int,
        attacking_armies: int,
        defending_armies: int,
    ) -> None:
        """
        Initialise the attack action with the attacker, defender territories
        and number of armies.
        """
        super().__init__()
        self.from_territory = from_territory
        self.to_territory = to_territory
        self.attacking_armies = attacking_armies
        self.defending_armies = defending_armies

    def apply_outcome(
        self, game_state: GameState, outcome: tuple[int, int]
    ) -> GameState:
        """
        Apply the outcome of the attack to the game state.
        """
        new_game_state = game_state.copy()

        attacking_player = new_game_state.current_player
        defending_player = new_game_state.owner[self.to_territory]

        if outcome[0] == 0:
            # defender wins, attacker loses all armies (except one)
            new_game_state.add_armies(self.from_territory, -self.attacking_armies)
            # the defender has the remaining armies from the outcome
            new_game_state.set_armies(self.to_territory, outcome[1])
        else:
            # attacker wins
            new_game_state.add_armies(self.from_territory, -self.attacking_armies)
            new_game_state.set_armies(self.to_territory, outcome[0])

            # transfer the territory to the attacker
            new_game_state.set_owner(self.to_territory, new_game_state.current_player)

            new_game_state.set_scalar('conquered_territory_this_turn', True)

            # If the defending player has no territories left, they are defeated
            # Their cards are transferred to the attacker, and check whether the
            # player now has more than 4 cards and therefore and has to trade
            if not any(
                new_game_state.owner[territory] == defending_player
                for territory in new_game_state.territory_ids
            ):
                new_game_state.add_defeated_player(defending_player)

                new_game_state.player_hands[attacking_player].extend(
                    new_game_state.player_hands[defending_player]
                )
                new_game_state.player_hands[defending_player] = []
                new_game_state.recompute_hands_deck_hash()

                # If the attacking player has more than 4 cards, they must trade
                if len(new_game_state.player_hands[attacking_player]) > 4:
                    new_game_state.set_scalar('current_turn_phase', 'trade_cards')

        return new_game_state

    def apply(
        self, game_state: 'GameState', battle_computer: 'BattleComputer | None' = None
    ) -> 'GameState':
        """
        Apply the attack action to the game state.
        """
        if battle_computer is None:
            raise ValueError('battle_computer must be provided for AttackAction.apply')

        # get the combat outcomes from the game engine
        outcome = battle_computer.get_outcome(
            attacking_armies=self.attacking_armies,
            defending_armies=self.defending_armies,
        )

        # apply_outcome() makes its own copy of game_state, so no need to
        # copy it here first.
        return self.apply_outcome(game_state=game_state, outcome=outcome)


class FortifyAction(Action):
    """
    Action of fortifying a territory.
    """

    def __init__(self, from_territory: int, to_territory: int, armies: int) -> None:
        """
        Initialise the fortify action with the from, to territories
        and number of armies.
        """
        super().__init__()
        self.from_territory = from_territory
        self.to_territory = to_territory
        self.armies = armies

    def apply(self, game_state: GameState) -> GameState:
        """
        Apply the fortify action to the game state.
        """
        new_game_state = game_state.copy()
        new_game_state.add_armies(self.from_territory, -self.armies)
        new_game_state.add_armies(self.to_territory, self.armies)
        new_game_state.set_scalar('fortified_territory_this_turn', True)
        return new_game_state
