"""
Federated Learning support for LLMESH Network
Enables collaborative model training without data sharing
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class FederationTask:
    """Federated learning task"""
    task_id: str
    model_id: str
    coordinator: str
    participants: List[str]
    rounds: int
    current_round: int
    aggregation_method: str
    min_participants: int
    created_at: float


@dataclass
class ModelUpdate:
    """Model update from participant"""
    task_id: str
    round_number: int
    participant_id: str
    weights: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    samples_trained: int


class FederatedLearning:
    """
    Federated learning coordinator and participant
    Manages distributed model training
    """

    def __init__(self, node_id: str):
        """Initialize federated learning module"""
        self.node_id = node_id
        self.active_tasks: Dict[str, FederationTask] = {}
        self.task_updates: Dict[str, List[ModelUpdate]] = {}
        self.local_models: Dict[str, Any] = {}

    async def create_task(self,
                         model_id: str,
                         rounds: int = 10,
                         min_participants: int = 3,
                         aggregation_method: str = "fedavg") -> str:
        """Create new federated learning task"""
        task_id = f"fed_{model_id}_{int(time.time())}"

        task = FederationTask(
            task_id=task_id,
            model_id=model_id,
            coordinator=self.node_id,
            participants=[],
            rounds=rounds,
            current_round=0,
            aggregation_method=aggregation_method,
            min_participants=min_participants,
            created_at=time.time()
        )

        self.active_tasks[task_id] = task
        self.task_updates[task_id] = []

        logger.info(f"Created federated learning task {task_id}")
        return task_id

    async def join_task(self, task_id: str) -> bool:
        """Join federated learning task as participant"""
        task = self.active_tasks.get(task_id)
        if not task:
            return False

        if self.node_id not in task.participants:
            task.participants.append(self.node_id)

        logger.info(f"Joined federated learning task {task_id}")
        return True

    async def submit_update(self, update: ModelUpdate):
        """Submit model update for aggregation"""
        task_id = update.task_id

        if task_id not in self.task_updates:
            self.task_updates[task_id] = []

        self.task_updates[task_id].append(update)

        # Check if ready to aggregate
        task = self.active_tasks.get(task_id)
        if task and task.coordinator == self.node_id:
            await self._check_aggregation(task_id)

    async def train_local_model(self, 
                              task_id: str,
                              training_data: Any,
                              epochs: int = 5) -> ModelUpdate:
        """Train model locally on private data"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Simulate local training
        logger.info(f"Training local model for task {task_id}")

        # In production, this would:
        # 1. Load the global model
        # 2. Train on local data
        # 3. Extract weight updates

        await asyncio.sleep(1.0)  # Simulate training time

        # Generate mock weight updates
        weights = {
            "layer1": np.random.randn(100, 50),
            "layer2": np.random.randn(50, 10)
        }

        metrics = {
            "loss": np.random.uniform(0.1, 0.5),
            "accuracy": np.random.uniform(0.7, 0.95)
        }

        update = ModelUpdate(
            task_id=task_id,
            round_number=task.current_round,
            participant_id=self.node_id,
            weights=weights,
            metrics=metrics,
            samples_trained=1000
        )

        return update

    async def aggregate_updates(self, 
                              task_id: str,
                              method: str = "fedavg") -> Dict[str, np.ndarray]:
        """Aggregate model updates from participants"""
        updates = self.task_updates.get(task_id, [])

        if not updates:
            raise ValueError("No updates to aggregate")

        if method == "fedavg":
            return self._federated_averaging(updates)
        elif method == "median":
            return self._median_aggregation(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _federated_averaging(self, 
                           updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Federated averaging aggregation"""
        # Calculate total samples
        total_samples = sum(u.samples_trained for u in updates)

        # Initialize aggregated weights
        aggregated = {}

        for layer_name in updates[0].weights:
            # Weighted average based on samples
            layer_sum = None

            for update in updates:
                weight = update.samples_trained / total_samples
                layer_weights = update.weights[layer_name]

                if layer_sum is None:
                    layer_sum = layer_weights * weight
                else:
                    layer_sum += layer_weights * weight

            aggregated[layer_name] = layer_sum

        return aggregated

    def _median_aggregation(self, 
                          updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Median aggregation (Byzantine-robust)"""
        aggregated = {}

        for layer_name in updates[0].weights:
            # Stack all weight matrices
            all_weights = np.stack([u.weights[layer_name] for u in updates])

            # Compute element-wise median
            aggregated[layer_name] = np.median(all_weights, axis=0)

        return aggregated

    async def _check_aggregation(self, task_id: str):
        """Check if ready to aggregate and proceed"""
        task = self.active_tasks[task_id]
        updates = self.task_updates.get(task_id, [])

        # Filter updates for current round
        current_updates = [u for u in updates 
                         if u.round_number == task.current_round]

        # Check if enough participants
        if len(current_updates) >= task.min_participants:
            # Aggregate updates
            aggregated_weights = await self.aggregate_updates(
                task_id, 
                task.aggregation_method
            )

            # Broadcast new global model
            await self._broadcast_global_model(task_id, aggregated_weights)

            # Move to next round
            task.current_round += 1

            # Clear updates for this round
            self.task_updates[task_id] = [u for u in updates 
                                        if u.round_number > task.current_round]

            logger.info(f"Completed round {task.current_round} for task {task_id}")

            # Check if training complete
            if task.current_round >= task.rounds:
                await self._complete_task(task_id)

    async def _broadcast_global_model(self, 
                                    task_id: str,
                                    weights: Dict[str, np.ndarray]):
        """Broadcast updated global model to participants"""
        task = self.active_tasks[task_id]

        # In production, this would send the model to all participants
        logger.info(f"Broadcasting global model for task {task_id}")

    async def _complete_task(self, task_id: str):
        """Complete federated learning task"""
        task = self.active_tasks[task_id]

        logger.info(f"Federated learning task {task_id} completed after {task.rounds} rounds")

        # Clean up
        del self.active_tasks[task_id]
        del self.task_updates[task_id]

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of federated learning task"""
        task = self.active_tasks.get(task_id)
        if not task:
            return {"status": "not_found"}

        updates = self.task_updates.get(task_id, [])
        current_round_updates = [u for u in updates 
                               if u.round_number == task.current_round]

        return {
            "status": "active",
            "current_round": task.current_round,
            "total_rounds": task.rounds,
            "participants": len(task.participants),
            "updates_received": len(current_round_updates),
            "min_participants": task.min_participants,
            "ready_to_aggregate": len(current_round_updates) >= task.min_participants
        }
