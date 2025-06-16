"""
Proof of Compute consensus mechanism for MESH network
"""

import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComputeProof:
    """Proof of compute work"""
    node_id: str
    model_hash: str
    input_hash: str
    output_hash: str
    computation_time: float
    timestamp: float
    nonce: int

    def verify(self) -> bool:
        """Verify the proof is valid"""
        # Check proof of work
        proof_string = f"{self.node_id}{self.model_hash}{self.input_hash}{self.output_hash}{self.nonce}"
        proof_hash = hashlib.sha256(proof_string.encode()).hexdigest()

        # Require leading zeros based on difficulty
        return proof_hash.startswith("0000")


class ProofOfCompute:
    """
    Proof of Compute consensus mechanism
    Rewards nodes based on actual AI compute performed
    """

    def __init__(self):
        """Initialize consensus mechanism"""
        self.proofs: Dict[str, List[ComputeProof]] = {}
        self.reputation_scores: Dict[str, float] = {}
        self.stake_requirements: Dict[str, float] = {}
        self.verification_threshold = 0.67  # 2/3 majority

    async def verify_node(self, node: Any) -> bool:
        """Verify a node can join the network"""
        # Check minimum stake
        if node.stake < self.get_minimum_stake():
            return False

        # Check reputation if existing node
        if node.node_id in self.reputation_scores:
            if self.reputation_scores[node.node_id] < 0.5:
                return False

        return True

    async def submit_proof(self, proof: ComputeProof) -> bool:
        """Submit proof of compute work"""
        # Verify proof format
        if not proof.verify():
            logger.warning(f"Invalid proof from node {proof.node_id}")
            self._penalize_node(proof.node_id)
            return False

        # Store proof for verification
        if proof.node_id not in self.proofs:
            self.proofs[proof.node_id] = []

        self.proofs[proof.node_id].append(proof)

        # Trigger verification if enough proofs
        if len(self.proofs[proof.node_id]) >= 10:
            await self._verify_node_work(proof.node_id)

        return True

    async def validate_inference(self, 
                               model_name: str,
                               input_data: Any,
                               output_data: Any,
                               node_id: str) -> bool:
        """Validate an inference result"""
        # In production, this would:
        # 1. Select random validators
        # 2. Have them run the same inference
        # 3. Compare results
        # 4. Reach consensus

        validators = await self._select_validators(node_id)
        validations = await self._collect_validations(
            validators, model_name, input_data, output_data
        )

        return self._reach_consensus(validations)

    def calculate_rewards(self, period: float = 3600) -> Dict[str, float]:
        """Calculate rewards for nodes based on compute contribution"""
        rewards = {}
        total_compute = 0

        # Calculate total valid compute
        for node_id, proofs in self.proofs.items():
            valid_proofs = [p for p in proofs 
                           if time.time() - p.timestamp < period]
            compute_score = sum(1 / p.computation_time for p in valid_proofs)
            total_compute += compute_score
            rewards[node_id] = compute_score

        # Normalize rewards
        if total_compute > 0:
            for node_id in rewards:
                rewards[node_id] = (rewards[node_id] / total_compute) * 1000  # 1000 MESH per period

        return rewards

    def get_minimum_stake(self) -> float:
        """Get current minimum stake requirement"""
        # Dynamic based on network security needs
        base_stake = 1000.0

        # Increase if attacks detected
        if self._detect_attacks():
            base_stake *= 2.0

        return base_stake

    def update_reputation(self, node_id: str, success: bool):
        """Update node reputation based on behavior"""
        if node_id not in self.reputation_scores:
            self.reputation_scores[node_id] = 1.0

        if success:
            # Increase reputation (max 2.0)
            self.reputation_scores[node_id] = min(
                self.reputation_scores[node_id] * 1.01, 2.0
            )
        else:
            # Decrease reputation (min 0.0)
            self.reputation_scores[node_id] = max(
                self.reputation_scores[node_id] * 0.95, 0.0
            )

    async def _verify_node_work(self, node_id: str):
        """Verify accumulated work from a node"""
        proofs = self.proofs[node_id]

        # Sample random proofs for verification
        sample_size = min(3, len(proofs))
        sampled_proofs = proofs[-sample_size:]

        valid_count = 0
        for proof in sampled_proofs:
            if await self._verify_single_proof(proof):
                valid_count += 1

        # Update reputation based on verification
        success_rate = valid_count / sample_size
        if success_rate >= 0.9:
            self.update_reputation(node_id, True)
        else:
            self.update_reputation(node_id, False)

        # Clear old proofs
        self.proofs[node_id] = []

    async def _verify_single_proof(self, proof: ComputeProof) -> bool:
        """Verify a single proof of compute"""
        # In production, this would re-run computation or check against known results
        # For now, simulate verification
        await asyncio.sleep(0.1)  # Simulate verification time
        return proof.verify()

    async def _select_validators(self, exclude_node: str) -> List[str]:
        """Select validator nodes based on stake and reputation"""
        eligible_nodes = [
            node_id for node_id, score in self.reputation_scores.items()
            if node_id != exclude_node and score > 0.8
        ]

        # Sort by reputation and stake
        eligible_nodes.sort(
            key=lambda n: self.reputation_scores[n], 
            reverse=True
        )

        # Select top validators
        return eligible_nodes[:5]

    async def _collect_validations(self, 
                                 validators: List[str],
                                 model_name: str,
                                 input_data: Any,
                                 expected_output: Any) -> List[bool]:
        """Collect validation results from validators"""
        validations = []

        for validator in validators:
            # In production, send validation request to validator nodes
            # For now, simulate validation
            await asyncio.sleep(0.05)
            validations.append(True)  # Simulated result

        return validations

    def _reach_consensus(self, validations: List[bool]) -> bool:
        """Reach consensus from validation results"""
        if not validations:
            return False

        positive_votes = sum(validations)
        consensus_ratio = positive_votes / len(validations)

        return consensus_ratio >= self.verification_threshold

    def _penalize_node(self, node_id: str):
        """Penalize a node for invalid behavior"""
        self.update_reputation(node_id, False)

        # Increase stake requirement for repeat offenders
        if node_id not in self.stake_requirements:
            self.stake_requirements[node_id] = self.get_minimum_stake()
        else:
            self.stake_requirements[node_id] *= 1.5

    def _detect_attacks(self) -> bool:
        """Detect potential attacks on the network"""
        # Check for suspicious patterns
        recent_failures = sum(
            1 for scores in self.reputation_scores.values()
            if scores < 0.5
        )

        total_nodes = len(self.reputation_scores)
        if total_nodes > 0:
            failure_rate = recent_failures / total_nodes
            return failure_rate > 0.2

        return False
