"""
P2P Transport Layer for LLMESH Network
Handles encrypted peer-to-peer communication
"""

import asyncio
import json
import hashlib
from typing import Dict, Callable, Optional, Any
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """P2P message structure"""
    msg_id: str
    msg_type: str
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: float
    signature: Optional[str] = None


class P2PTransport:
    """
    Secure P2P transport layer
    Features: encryption, message routing, NAT traversal
    """

    def __init__(self, node_id: str, port: int = 8888):
        """Initialize P2P transport"""
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, asyncio.StreamWriter] = {}
        self.handlers: Dict[str, Callable] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        self.server: Optional[asyncio.Server] = None
        self._message_cache: Dict[str, float] = {}  # For deduplication

    async def start(self):
        """Start P2P server"""
        self.server = await asyncio.start_server(
            self._handle_connection,
            '0.0.0.0',
            self.port
        )

        logger.info(f"P2P transport started on port {self.port}")

    async def stop(self):
        """Stop P2P server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all peer connections
        for writer in self.peers.values():
            writer.close()
            await writer.wait_closed()

    async def connect_to_peer(self, peer_id: str, host: str, port: int):
        """Connect to a peer node"""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            self.peers[peer_id] = writer

            # Start handshake
            await self._handshake(peer_id, reader, writer)

            # Start message handler
            asyncio.create_task(self._handle_peer_messages(peer_id, reader))

            logger.info(f"Connected to peer {peer_id} at {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id}: {e}")

    async def send_message(self, recipient: str, msg_type: str, 
                          payload: Dict[str, Any]) -> bool:
        """Send message to a peer"""
        if recipient not in self.peers:
            logger.warning(f"Peer {recipient} not connected")
            return False

        message = Message(
            msg_id=self._generate_msg_id(),
            msg_type=msg_type,
            sender=self.node_id,
            recipient=recipient,
            payload=payload,
            timestamp=asyncio.get_event_loop().time()
        )

        # Sign message
        message.signature = self._sign_message(message)

        # Encrypt and send
        encrypted = self._encrypt_message(message, recipient)

        try:
            writer = self.peers[recipient]
            writer.write(encrypted + b'\n')
            await writer.drain()
            return True

        except Exception as e:
            logger.error(f"Failed to send message to {recipient}: {e}")
            return False

    def _generate_msg_id(self) -> str:
        """Generate unique message ID"""
        import uuid
        return str(uuid.uuid4())
