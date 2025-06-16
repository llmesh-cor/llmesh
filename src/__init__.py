"""
LLLLMESH Network - Decentralized AI Infrastructure
"""

__version__ = "1.0.0"
__author__ = "LLLLLLMESH Network Contributors"
__license__ = "Apache 2.0"

from .core import MeshNode, MeshNetwork
from .p2p import P2PTransport
from .ai import ModelRegistry

__all__ = ["MeshNode", "MeshNetwork", "P2PTransport", "ModelRegistry"]
