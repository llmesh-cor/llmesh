"""
Governance module for LLLLMESH Network
Implements on-chain governance and voting
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ProposalStatus(Enum):
    """Proposal status enum"""
    PENDING = "pending"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


@dataclass
class Proposal:
    """Governance proposal"""
    proposal_id: str
    proposer: str
    title: str
    description: str
    proposal_type: str
    parameters: Dict[str, Any]
    created_at: float
    voting_start: float
    voting_end: float
    status: ProposalStatus
    votes_for: float
    votes_against: float
    votes_abstain: float
    execution_time: Optional[float] = None


@dataclass
class Vote:
    """Individual vote record"""
    voter: str
    proposal_id: str
    vote_type: str  # for, against, abstain
    voting_power: float
    timestamp: float
    reason: Optional[str] = None


class GovernanceModule:
    """
    On-chain governance implementation
    Manages proposals, voting, and execution
    """

    def __init__(self, token_economics):
        """Initialize governance module"""
        self.token_economics = token_economics
        self.proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = {}

        # Governance parameters
        self.proposal_threshold = 10000  # 10k MESH to create proposal
        self.quorum_threshold = 0.10  # 10% of staked tokens
        self.approval_threshold = 0.51  # 51% approval
        self.voting_period = 3 * 24 * 3600  # 3 days
        self.execution_delay = 24 * 3600  # 1 day timelock

        # Proposal types and their validators
        self.proposal_types = {
            "parameter_change": self._validate_parameter_change,
            "token_mint": self._validate_token_mint,
            "emergency_pause": self._validate_emergency_pause,
            "upgrade": self._validate_upgrade
        }

    def create_proposal(self,
                       proposer: str,
                       title: str,
                       description: str,
                       proposal_type: str,
                       parameters: Dict[str, Any]) -> Optional[str]:
        """Create a new governance proposal"""

        # Check proposer has sufficient stake
        if proposer not in self.token_economics.stakes:
            logger.warning(f"Proposer {proposer} has no stake")
            return None

        stake_amount = self.token_economics.stakes[proposer].amount
        if stake_amount < self.proposal_threshold:
            logger.warning(f"Insufficient stake: {stake_amount} < {self.proposal_threshold}")
            return None

        # Validate proposal type
        if proposal_type not in self.proposal_types:
            logger.warning(f"Invalid proposal type: {proposal_type}")
            return None

        # Validate parameters
        validator = self.proposal_types[proposal_type]
        if not validator(parameters):
            logger.warning("Invalid proposal parameters")
            return None

        # Create proposal
        proposal_id = f"MESH-{int(time.time())}-{proposer[:8]}"
        current_time = time.time()

        proposal = Proposal(
            proposal_id=proposal_id,
            proposer=proposer,
            title=title,
            description=description,
            proposal_type=proposal_type,
            parameters=parameters,
            created_at=current_time,
            voting_start=current_time + 3600,  # 1 hour delay
            voting_end=current_time + 3600 + self.voting_period,
            status=ProposalStatus.PENDING,
            votes_for=0,
            votes_against=0,
            votes_abstain=0
        )

        self.proposals[proposal_id] = proposal
        self.votes[proposal_id] = []

        logger.info(f"Created proposal {proposal_id}: {title}")
        return proposal_id

    def cast_vote(self,
                 voter: str,
                 proposal_id: str,
                 vote_type: str,
                 reason: Optional[str] = None) -> bool:
        """Cast vote on a proposal"""

        # Validate proposal
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]
        current_time = time.time()

        # Check voting period
        if current_time < proposal.voting_start:
            logger.warning("Voting has not started")
            return False

        if current_time > proposal.voting_end:
            logger.warning("Voting has ended")
            return False

        # Check voter has stake
        if voter not in self.token_economics.stakes:
            return False

        # Check if already voted
        existing_votes = [v for v in self.votes[proposal_id] if v.voter == voter]
        if existing_votes:
            logger.warning(f"{voter} already voted on {proposal_id}")
            return False

        # Calculate voting power
        voting_power = self.token_economics.stakes[voter].amount

        # Record vote
        vote = Vote(
            voter=voter,
            proposal_id=proposal_id,
            vote_type=vote_type,
            voting_power=voting_power,
            timestamp=current_time,
            reason=reason
        )

        self.votes[proposal_id].append(vote)

        # Update proposal vote counts
        if vote_type == "for":
            proposal.votes_for += voting_power
        elif vote_type == "against":
            proposal.votes_against += voting_power
        else:  # abstain
            proposal.votes_abstain += voting_power

        logger.info(f"{voter} voted {vote_type} on {proposal_id}")
        return True

    def finalize_proposal(self, proposal_id: str) -> bool:
        """Finalize voting and determine outcome"""

        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]
        current_time = time.time()

        # Check voting has ended
        if current_time <= proposal.voting_end:
            return False

        # Calculate total votes
        total_votes = proposal.votes_for + proposal.votes_against + proposal.votes_abstain
        total_stake = sum(s.amount for s in self.token_economics.stakes.values())

        # Check quorum
        participation_rate = total_votes / max(total_stake, 1)
        if participation_rate < self.quorum_threshold:
            proposal.status = ProposalStatus.REJECTED
            logger.info(f"Proposal {proposal_id} rejected: insufficient quorum")
            return True

        # Check approval
        if total_votes > 0:
            approval_rate = proposal.votes_for / total_votes
            if approval_rate >= self.approval_threshold:
                proposal.status = ProposalStatus.PASSED
                proposal.execution_time = current_time + self.execution_delay
                logger.info(f"Proposal {proposal_id} passed")
            else:
                proposal.status = ProposalStatus.REJECTED
                logger.info(f"Proposal {proposal_id} rejected")
        else:
            proposal.status = ProposalStatus.REJECTED

        return True

    def execute_proposal(self, proposal_id: str) -> bool:
        """Execute a passed proposal"""

        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]
        current_time = time.time()

        # Check proposal passed
        if proposal.status != ProposalStatus.PASSED:
            return False

        # Check timelock
        if current_time < proposal.execution_time:
            logger.warning("Proposal still in timelock")
            return False

        # Execute based on type
        success = self._execute_proposal_action(proposal)

        if success:
            proposal.status = ProposalStatus.EXECUTED
            logger.info(f"Executed proposal {proposal_id}")

        return success

    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals"""
        current_time = time.time()

        active = []
        for proposal in self.proposals.values():
            if (proposal.status == ProposalStatus.PENDING and 
                current_time >= proposal.voting_start and
                current_time <= proposal.voting_end):
                proposal.status = ProposalStatus.ACTIVE

            if proposal.status == ProposalStatus.ACTIVE:
                active.append(proposal)

        return active

    def get_proposal_details(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a proposal"""

        if proposal_id not in self.proposals:
            return None

        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]

        return {
            "proposal": proposal,
            "votes": votes,
            "vote_breakdown": {
                "for": proposal.votes_for,
                "against": proposal.votes_against,
                "abstain": proposal.votes_abstain
            },
            "participation": len(votes),
            "top_voters": sorted(
                [(v.voter, v.voting_power) for v in votes],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def _validate_parameter_change(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameter change proposal"""
        required = ["parameter_name", "new_value"]
        return all(k in parameters for k in required)

    def _validate_token_mint(self, parameters: Dict[str, Any]) -> bool:
        """Validate token mint proposal"""
        required = ["amount", "recipient", "reason"]
        if not all(k in parameters for k in required):
            return False

        # Check mint amount is reasonable (< 1% of supply)
        max_mint = self.token_economics.total_supply * 0.01
        return parameters["amount"] <= max_mint

    def _validate_emergency_pause(self, parameters: Dict[str, Any]) -> bool:
        """Validate emergency pause proposal"""
        required = ["component", "duration", "reason"]
        return all(k in parameters for k in required)

    def _validate_upgrade(self, parameters: Dict[str, Any]) -> bool:
        """Validate upgrade proposal"""
        required = ["version", "upgrade_hash", "description"]
        return all(k in parameters for k in required)

    def _execute_proposal_action(self, proposal: Proposal) -> bool:
        """Execute the action specified in a proposal"""

        if proposal.proposal_type == "parameter_change":
            # In production, this would update network parameters
            logger.info(f"Updating parameter: {proposal.parameters}")
            return True

        elif proposal.proposal_type == "token_mint":
            # Mint new tokens
            amount = proposal.parameters["amount"]
            recipient = proposal.parameters["recipient"]

            self.token_economics.balances[recipient] = (
                self.token_economics.get_balance(recipient) + amount
            )
            self.token_economics.circulating_supply += amount

            logger.info(f"Minted {amount} MESH to {recipient}")
            return True

        elif proposal.proposal_type == "emergency_pause":
            # In production, this would pause components
            logger.warning(f"Emergency pause: {proposal.parameters}")
            return True

        elif proposal.proposal_type == "upgrade":
            # In production, this would trigger upgrade
            logger.info(f"Upgrade approved: {proposal.parameters}")
            return True

        return False
