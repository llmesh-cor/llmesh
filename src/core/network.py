"""
LLMESH Network Orchestrator
Manages the overall network state and coordination
"""

import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import time
import logging

from .node import MeshNode, NodeInfo
from .consensus import ProofOfCompute

logger = logging.getLogger(__name__)


@dataclass
class NetworkStats:
    """Global network statistics"""
    total_nodes: int
    total_stake: float
    total_compute: float
    total_models: int
    avg_response_time: float
    network_health: float


class MeshNetwork:
    """
    LLMESH Network orchestrator
    Manages network state, consensus, and global operations
    """

    def __init__(self):
        """Initialize the MESH network"""
        self.nodes: Dict[str, MeshNode] = {}
        self.network_state: Dict[str, Any] = {}
        self.consensus = ProofOfCompute()
        self.start_time = time.time()
        self._is_running = False

    async def bootstrap(self, bootstrap_nodes: List[str] = None):
        """Bootstrap the network with initial nodes"""
        logger.info("Bootstrapping MESH network")

        if bootstrap_nodes:
            for node_addr in bootstrap_nodes:
                await self._connect_to_bootstrap_node(node_addr)

        self._is_running = True
        asyncio.create_task(self._network_monitor_loop())

    async def register_node(self, node: MeshNode) -> bool:
        """Register a new node in the network"""
        if node.stake < self.get_minimum_stake():
            logger.warning(f"Node {node.node_id} stake too low")
            return False

        # Verify node through consensus
        if await self.consensus.verify_node(node):
            self.nodes[node.node_id] = node
            logger.info(f"Node {node.node_id} registered successfully")
            return True

        return False

    async def handle_inference_request(self, model_name: str, input_data: Any) -> Any:
        """Handle inference request at network level"""
        # Find nodes hosting the model
        capable_nodes = self._find_capable_nodes(model_name)

        if not capable_nodes:
            raise ValueError(f"No nodes found hosting {model_name}")

        # Select optimal node based on load, latency, stake
        selected_node = self._select_optimal_node(capable_nodes)

        # Route request
        return await selected_node.request_inference(model_name, input_data)

    def get_network_stats(self) -> NetworkStats:
        """Get current network statistics"""
        total_stake = sum(node.stake for node in self.nodes.values())
        total_models = len(self._get_all_models())

        return NetworkStats(
            total_nodes=len(self.nodes),
            total_stake=total_stake,
            total_compute=self._calculate_total_compute(),
            total_models=total_models,
            avg_response_time=self._calculate_avg_response_time(),
            network_health=self._calculate_network_health()
        )

    def get_minimum_stake(self) -> float:
        """Get minimum stake requirement"""
        # Dynamic based on network size
        base_stake = 100.0
        network_factor = len(self.nodes) / 100
        return base_stake * (1 + network_factor)

    async def _network_monitor_loop(self):
        """Monitor network health and performance"""
        while self._is_running:
            await asyncio.sleep(30)

            # Check node health
            await self._check_node_health()

            # Update network metrics
            self._update_network_metrics()

            # Rebalance if needed
            if self._needs_rebalancing():
                await self._rebalance_network()

    async def _check_node_health(self):
        """Check health of all nodes"""
        unhealthy_nodes = []

        for node_id, node in self.nodes.items():
            if not await self._is_node_healthy(node):
                unhealthy_nodes.append(node_id)

        # Remove unhealthy nodes
        for node_id in unhealthy_nodes:
            logger.warning(f"Removing unhealthy node {node_id}")
            del self.nodes[node_id]

    async def _is_node_healthy(self, node: MeshNode) -> bool:
        """Check if a node is healthy"""
        # Simplified health check
        return node.is_running

    def _find_capable_nodes(self, model_name: str) -> List[MeshNode]:
        """Find nodes capable of serving a model"""
        capable = []
        for node in self.nodes.values():
            if model_name in node.hosted_models:
                capable.append(node)
        return capable

    def _select_optimal_node(self, nodes: List[MeshNode]) -> MeshNode:
        """Select optimal node based on multiple factors"""
        # Simplified selection - in production considers:
        # - Current load
        # - Network latency
        # - Stake amount
        # - Historical reliability
        return max(nodes, key=lambda n: n.stake)

    def _get_all_models(self) -> Set[str]:
        """Get all models in the network"""
        models = set()
        for node in self.nodes.values():
            models.update(node.hosted_models.keys())
        return models

    def _calculate_total_compute(self) -> float:
        """Calculate total network compute power"""
        # Simplified - in production measures actual FLOPS
        return len(self.nodes) * 100.0  # 100 TFLOPS per node average

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # Simplified - in production tracks actual latencies
        return 47.0  # ms

    def _calculate_network_health(self) -> float:
        """Calculate overall network health score"""
        # Factors: uptime, node distribution, response times
        uptime = min((time.time() - self.start_time) / 86400, 1.0)  # Days
        distribution = min(len(self.nodes) / 1000, 1.0)

        return (uptime + distribution) / 2

    def _update_network_metrics(self):
        """Update network performance metrics"""
        self.network_state["last_update"] = time.time()
        self.network_state["stats"] = self.get_network_stats()

    def _needs_rebalancing(self) -> bool:
        """Check if network needs rebalancing"""
        # Check for hotspots, uneven distribution, etc.
        return False  # Simplified

    async def _rebalance_network(self):
        """Rebalance network load"""
        logger.info("Rebalancing network")
        # In production: migrate models, adjust routing, etc.

    async def _connect_to_bootstrap_node(self, node_addr: str):
        """Connect to a bootstrap node"""
        logger.info(f"Connecting to bootstrap node {node_addr}")
        # In production: actual P2P connection
