"""
MESH Token Economics Implementation
Manages staking, rewards, and economic incentives
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class StakeInfo:
    """Staking information for a node"""
    node_id: str
    amount: float
    locked_until: float
    rewards_earned: float
    last_claim: float
    delegation_from: List[str]
    slash_history: List[float]


@dataclass
class Transaction:
    """Token transaction record"""
    tx_id: str
    from_address: str
    to_address: str
    amount: float
    tx_type: str  # transfer, stake, unstake, reward, slash
    timestamp: float
    metadata: Dict[str, Any]


class TokenEconomics:
    """
    MESH token economics engine
    Handles all economic aspects of the network
    """

    def __init__(self):
        """Initialize token economics"""
        # Token parameters
        self.total_supply = 1_000_000_000  # 1 billion MESH
        self.circulating_supply = 100_000_000  # 100 million initial
        self.inflation_rate = 0.05  # 5% annual
        self.base_reward_rate = 0.10  # 10% APY base staking reward

        # Economic parameters
        self.min_stake = 1000  # Minimum stake to run a node
        self.unstake_period = 7 * 24 * 3600  # 7 days
        self.slash_percentage = 0.10  # 10% slash for misbehavior

        # State
        self.balances: Dict[str, float] = {}
        self.stakes: Dict[str, StakeInfo] = {}
        self.pending_unstakes: Dict[str, List[Tuple[float, float]]] = {}
        self.reward_pool = 10_000_000  # Initial reward pool

    def get_balance(self, address: str) -> float:
        """Get token balance for address"""
        return self.balances.get(address, 0.0)

    def transfer(self, from_addr: str, to_addr: str, amount: float) -> bool:
        """Transfer tokens between addresses"""
        if amount <= 0:
            return False

        from_balance = self.get_balance(from_addr)
        if from_balance < amount:
            logger.warning(f"Insufficient balance: {from_addr} has {from_balance}, needs {amount}")
            return False

        # Execute transfer
        self.balances[from_addr] = from_balance - amount
        self.balances[to_addr] = self.get_balance(to_addr) + amount

        logger.info(f"Transferred {amount} MESH from {from_addr} to {to_addr}")
        return True

    def stake(self, node_id: str, amount: float) -> bool:
        """Stake tokens for a node"""
        if amount < self.min_stake:
            logger.warning(f"Stake amount {amount} below minimum {self.min_stake}")
            return False

        balance = self.get_balance(node_id)
        if balance < amount:
            return False

        # Create or update stake
        if node_id in self.stakes:
            stake_info = self.stakes[node_id]
            stake_info.amount += amount
        else:
            stake_info = StakeInfo(
                node_id=node_id,
                amount=amount,
                locked_until=0,
                rewards_earned=0,
                last_claim=time.time(),
                delegation_from=[],
                slash_history=[]
            )
            self.stakes[node_id] = stake_info

        # Deduct from balance
        self.balances[node_id] = balance - amount

        logger.info(f"Node {node_id} staked {amount} MESH")
        return True

    def begin_unstake(self, node_id: str, amount: float) -> bool:
        """Begin unstaking process"""
        if node_id not in self.stakes:
            return False

        stake_info = self.stakes[node_id]
        if stake_info.amount < amount:
            return False

        # Lock tokens for unstaking period
        unlock_time = time.time() + self.unstake_period

        if node_id not in self.pending_unstakes:
            self.pending_unstakes[node_id] = []

        self.pending_unstakes[node_id].append((amount, unlock_time))
        stake_info.amount -= amount
        stake_info.locked_until = max(stake_info.locked_until, unlock_time)

        logger.info(f"Node {node_id} began unstaking {amount} MESH")
        return True

    def complete_unstake(self, node_id: str) -> float:
        """Complete unstaking for unlocked tokens"""
        if node_id not in self.pending_unstakes:
            return 0

        current_time = time.time()
        completed_amount = 0
        remaining_unstakes = []

        for amount, unlock_time in self.pending_unstakes[node_id]:
            if unlock_time <= current_time:
                # Return to balance
                self.balances[node_id] = self.get_balance(node_id) + amount
                completed_amount += amount
            else:
                remaining_unstakes.append((amount, unlock_time))

        self.pending_unstakes[node_id] = remaining_unstakes

        if completed_amount > 0:
            logger.info(f"Completed unstaking {completed_amount} MESH for {node_id}")

        return completed_amount

    def calculate_rewards(self, node_id: str, 
                         compute_contribution: float,
                         uptime: float) -> float:
        """Calculate staking rewards for a node"""
        if node_id not in self.stakes:
            return 0

        stake_info = self.stakes[node_id]
        time_staked = time.time() - stake_info.last_claim

        # Base staking reward
        base_reward = (stake_info.amount * self.base_reward_rate * 
                      time_staked / (365 * 24 * 3600))

        # Performance multipliers
        compute_multiplier = 1 + (compute_contribution * 0.5)  # Up to 50% bonus
        uptime_multiplier = uptime  # 0-1 based on uptime

        # Calculate total reward
        total_reward = base_reward * compute_multiplier * uptime_multiplier

        # Cap at available reward pool
        total_reward = min(total_reward, self.reward_pool * 0.01)  # Max 1% of pool

        return total_reward

    def claim_rewards(self, node_id: str, 
                     compute_contribution: float,
                     uptime: float) -> float:
        """Claim accumulated rewards"""
        reward = self.calculate_rewards(node_id, compute_contribution, uptime)

        if reward > 0:
            # Add to balance
            self.balances[node_id] = self.get_balance(node_id) + reward

            # Update stake info
            stake_info = self.stakes[node_id]
            stake_info.rewards_earned += reward
            stake_info.last_claim = time.time()

            # Deduct from reward pool
            self.reward_pool -= reward

            logger.info(f"Node {node_id} claimed {reward} MESH in rewards")

        return reward

    def slash_stake(self, node_id: str, reason: str) -> float:
        """Slash stake for misbehavior"""
        if node_id not in self.stakes:
            return 0

        stake_info = self.stakes[node_id]
        slash_amount = stake_info.amount * self.slash_percentage

        # Reduce stake
        stake_info.amount -= slash_amount
        stake_info.slash_history.append(time.time())

        # Add to reward pool
        self.reward_pool += slash_amount

        logger.warning(f"Slashed {slash_amount} MESH from {node_id} for {reason}")

        # Remove node if stake below minimum
        if stake_info.amount < self.min_stake:
            del self.stakes[node_id]
            logger.warning(f"Node {node_id} removed due to insufficient stake")

        return slash_amount

    def get_staking_info(self) -> Dict[str, Any]:
        """Get overall staking statistics"""
        total_staked = sum(s.amount for s in self.stakes.values())

        return {
            "total_staked": total_staked,
            "staking_ratio": total_staked / self.circulating_supply,
            "active_stakers": len(self.stakes),
            "reward_pool": self.reward_pool,
            "avg_stake": total_staked / max(len(self.stakes), 1),
            "apr": self.base_reward_rate
        }

    def delegate_stake(self, delegator: str, node_id: str, amount: float) -> bool:
        """Delegate stake to a node operator"""
        if not self.transfer(delegator, node_id, amount):
            return False

        if node_id in self.stakes:
            self.stakes[node_id].delegation_from.append(delegator)

        logger.info(f"{delegator} delegated {amount} MESH to {node_id}")
        return True


class StakeManager:
    """
    Manages staking operations and validator selection
    """

    def __init__(self, economics: TokenEconomics):
        """Initialize stake manager"""
        self.economics = economics
        self.validator_set: List[str] = []
        self.epoch_length = 3600  # 1 hour epochs
        self.last_epoch = 0

    def update_validators(self) -> List[str]:
        """Update active validator set based on stakes"""
        current_time = time.time()

        if current_time - self.last_epoch < self.epoch_length:
            return self.validator_set

        # Sort nodes by stake
        staked_nodes = [
            (node_id, info.amount) 
            for node_id, info in self.economics.stakes.items()
            if info.amount >= self.economics.min_stake
        ]

        staked_nodes.sort(key=lambda x: x[1], reverse=True)

        # Select top validators (e.g., top 100)
        max_validators = 100
        self.validator_set = [node_id for node_id, _ in staked_nodes[:max_validators]]

        self.last_epoch = current_time

        logger.info(f"Updated validator set: {len(self.validator_set)} validators")
        return self.validator_set

    def is_validator(self, node_id: str) -> bool:
        """Check if node is in active validator set"""
        return node_id in self.validator_set

    def get_validator_power(self, node_id: str) -> float:
        """Get voting power of a validator"""
        if node_id not in self.economics.stakes:
            return 0

        stake_amount = self.economics.stakes[node_id].amount
        total_stake = sum(
            self.economics.stakes[v].amount 
            for v in self.validator_set
        )

        return stake_amount / max(total_stake, 1)

    def calculate_block_reward(self, proposer: str) -> float:
        """Calculate block reward for proposer"""
        base_block_reward = 10.0  # 10 MESH per block

        # Bonus based on stake
        if proposer in self.economics.stakes:
            stake_bonus = math.log10(
                self.economics.stakes[proposer].amount / self.economics.min_stake
            )
            return base_block_reward * (1 + stake_bonus * 0.1)

        return base_block_reward
