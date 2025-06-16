"""
Peer discovery mechanism for MESH network
Implements DHT-based discovery with bootstrap nodes
"""

import asyncio
import random
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a peer"""
    node_id: str
    address: str
    port: int
    last_seen: float
    capabilities: List[str]
    reputation: float = 1.0


class PeerDiscovery:
    """
    DHT-based peer discovery
    Uses Kademlia-inspired protocol for decentralized discovery
    """

    def __init__(self, node_id: str, bootstrap_nodes: List[str] = None):
        """Initialize peer discovery"""
        self.node_id = node_id
        self.routing_table: Dict[int, List[PeerInfo]] = {}
        self.k_bucket_size = 20
        self.alpha = 3  # Parallelism parameter
        self.bootstrap_nodes = bootstrap_nodes or []
        self._running = False

        # Initialize k-buckets
        for i in range(256):  # 256-bit node IDs
            self.routing_table[i] = []

    async def start(self):
        """Start peer discovery"""
        self._running = True

        # Bootstrap from known nodes
        if self.bootstrap_nodes:
            await self._bootstrap()

        # Start maintenance loops
        asyncio.create_task(self._refresh_loop())
        asyncio.create_task(self._cleanup_loop())

        logger.info("Peer discovery started")

    async def find_node(self, target_id: str) -> List[PeerInfo]:
        """Find nodes closest to target ID"""
        # Calculate XOR distances
        closest_peers = []

        for bucket in self.routing_table.values():
            for peer in bucket:
                distance = self._xor_distance(peer.node_id, target_id)
                closest_peers.append((distance, peer))

        # Sort by distance and return k closest
        closest_peers.sort(key=lambda x: x[0])
        return [peer for _, peer in closest_peers[:self.k_bucket_size]]

    async def announce(self, capability: str, data: Dict[str, Any]):
        """Announce capability to network"""
        # Find nodes to store announcement
        storage_nodes = await self.find_node(self._hash_key(capability))

        announcement = {
            "node_id": self.node_id,
            "capability": capability,
            "data": data,
            "timestamp": time.time()
        }

        # Store on multiple nodes for redundancy
        for node in storage_nodes[:self.alpha]:
            await self._store_on_node(node, announcement)

    def add_peer(self, peer: PeerInfo):
        """Add peer to routing table"""
        bucket_idx = self._get_bucket_index(peer.node_id)
        bucket = self.routing_table[bucket_idx]

        # Check if peer already exists
        for i, existing in enumerate(bucket):
            if existing.node_id == peer.node_id:
                # Update existing peer
                bucket[i] = peer
                return

        # Add new peer if bucket not full
        if len(bucket) < self.k_bucket_size:
            bucket.append(peer)
        else:
            # Replace least recently seen peer
            oldest_idx = min(range(len(bucket)), 
                           key=lambda i: bucket[i].last_seen)
            bucket[oldest_idx] = peer
