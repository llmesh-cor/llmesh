"""
MESH Node Implementation
Each node in the network acts as both a model host and router
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a MESH node"""
    node_id: str
    address: str
    port: int
    stake: float
    compute_power: float
    models: List[str]
    reputation: float


class MeshNode:
    """
    Core MESH node implementation
    Handles model hosting, routing, and network participation
    """

    def __init__(self, node_id: Optional[str] = None, stake: float = 0.0):
        """Initialize a new MESH node"""
        self.node_id = node_id or self._generate_node_id()
        self.stake = stake
        self.peers: Dict[str, NodeInfo] = {}
        self.hosted_models: Dict[str, Any] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import uuid
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]

    async def start(self, port: int = 8080):
        """Start the node and join the network"""
        logger.info(f"Starting MESH node {self.node_id} on port {port}")
        self.is_running = True

        # Start core services
        self._tasks.extend([
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._routing_update_loop()),
        ])

        logger.info(f"Node {self.node_id} started successfully")

    async def stop(self):
        """Gracefully stop the node"""
        logger.info(f"Stopping node {self.node_id}")
        self.is_running = False

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def deploy_model(self, model_path: str, name: str, fee: float = 0.1):
        """Deploy an AI model to the node"""
        logger.info(f"Deploying model {name} with fee {fee} MESH")

        # In production, this would load the actual model
        self.hosted_models[name] = {
            "path": model_path,
            "fee": fee,
            "requests": 0,
            "earnings": 0.0
        }

        # Announce to network
        await self._announce_model(name)

    async def request_inference(self, model_name: str, input_data: Any) -> Any:
        """Request inference from a model in the network"""
        # Find best route to model
        route = await self._find_route(model_name)

        if not route:
            raise ValueError(f"Model {model_name} not found in network")

        # Route request through mesh
        return await self._route_request(route, model_name, input_data)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to peers"""
        while self.is_running:
            await asyncio.sleep(30)
            await self._send_heartbeat()

    async def _discovery_loop(self):
        """Discover new peers in the network"""
        while self.is_running:
            await asyncio.sleep(60)
            await self._discover_peers()

    async def _routing_update_loop(self):
        """Update routing tables based on network state"""
        while self.is_running:
            await asyncio.sleep(10)
            await self._update_routing_table()

    async def _send_heartbeat(self):
        """Send heartbeat to all peers"""
        message = {
            "type": "heartbeat",
            "node_id": self.node_id,
            "stake": self.stake,
            "models": list(self.hosted_models.keys()),
            "timestamp": asyncio.get_event_loop().time()
        }

        for peer_id in self.peers:
            # In production, send via P2P transport
            logger.debug(f"Heartbeat sent to {peer_id}")

    async def _discover_peers(self):
        """Discover new peers using DHT"""
        # Simplified discovery - in production uses Kademlia DHT
        logger.debug("Running peer discovery")

    async def _update_routing_table(self):
        """Update routing table with optimal paths"""
        # Simplified routing - in production uses advanced algorithms
        for model_name in self._get_network_models():
            self.routing_table[model_name] = self._calculate_route(model_name)

    async def _announce_model(self, model_name: str):
        """Announce model availability to network"""
        message = {
            "type": "model_announce",
            "node_id": self.node_id,
            "model": model_name,
            "fee": self.hosted_models[model_name]["fee"]
        }

        # Broadcast to network
        logger.info(f"Announced model {model_name} to network")

    async def _find_route(self, model_name: str) -> Optional[List[str]]:
        """Find optimal route to model"""
        return self.routing_table.get(model_name)

    async def _route_request(self, route: List[str], model_name: str, input_data: Any) -> Any:
        """Route inference request through mesh"""
        # Simplified routing - in production handles failures, retries, etc.
        logger.debug(f"Routing request for {model_name} via {route}")

        # Mock response
        return {"result": "inference_result", "route": route}

    def _get_network_models(self) -> List[str]:
        """Get all models available in network"""
        models = set(self.hosted_models.keys())
        for peer in self.peers.values():
            models.update(peer.models)
        return list(models)

    def _calculate_route(self, model_name: str) -> List[str]:
        """Calculate optimal route to model"""
        # Simplified - in production uses Dijkstra's algorithm with latency/stake weights
        return [self.node_id, "peer1", "target"]

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            "node_id": self.node_id,
            "stake": self.stake,
            "peers": len(self.peers),
            "models": len(self.hosted_models),
            "total_requests": sum(m["requests"] for m in self.hosted_models.values()),
            "total_earnings": sum(m["earnings"] for m in self.hosted_models.values())
        }
