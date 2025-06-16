"""
Model Registry for LLMESH Network
Manages AI model storage, versioning, and distribution
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for registered models"""
    model_id: str
    name: str
    version: str
    format: str  # onnx, tensorflow, pytorch
    size: int
    hash: str
    owner: str
    fee: float
    description: str
    metrics: Dict[str, float]
    created_at: float
    updated_at: float


class ModelRegistry:
    """
    Decentralized model registry
    Tracks and manages AI models across the network
    """

    def __init__(self):
        """Initialize model registry"""
        self.models: Dict[str, ModelMetadata] = {}
        self.model_storage: Dict[str, bytes] = {}  # Simplified storage
        self.model_locations: Dict[str, List[str]] = {}  # Model ID -> Node IDs

    def register_model(self, 
                      name: str,
                      model_data: bytes,
                      format: str,
                      owner: str,
                      fee: float = 0.1,
                      description: str = "",
                      metrics: Dict[str, float] = None) -> str:
        """Register a new model"""
        # Generate model ID
        model_hash = hashlib.sha256(model_data).hexdigest()
        model_id = f"{name}_{model_hash[:8]}"

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version="1.0.0",
            format=format,
            size=len(model_data),
            hash=model_hash,
            owner=owner,
            fee=fee,
            description=description,
            metrics=metrics or {},
            created_at=time.time(),
            updated_at=time.time()
        )

        # Store model
        self.models[model_id] = metadata
        self.model_storage[model_id] = model_data

        logger.info(f"Model registered: {model_id}")
        return model_id

    def get_model(self, model_id: str) -> Optional[Tuple[ModelMetadata, bytes]]:
        """Retrieve model and metadata"""
        metadata = self.models.get(model_id)
        if not metadata:
            return None

        model_data = self.model_storage.get(model_id)
        if not model_data:
            # Try to fetch from network
            model_data = self._fetch_from_network(model_id)

        return (metadata, model_data) if model_data else None

    def search_models(self, 
                     name: Optional[str] = None,
                     owner: Optional[str] = None,
                     max_fee: Optional[float] = None) -> List[ModelMetadata]:
        """Search for models matching criteria"""
        results = []

        for metadata in self.models.values():
            # Apply filters
            if name and name.lower() not in metadata.name.lower():
                continue
            if owner and metadata.owner != owner:
                continue
            if max_fee is not None and metadata.fee > max_fee:
                continue

            results.append(metadata)

        # Sort by creation time (newest first)
        results.sort(key=lambda m: m.created_at, reverse=True)

        return results

    def update_model_metrics(self, model_id: str, 
                           metrics: Dict[str, float]):
        """Update model performance metrics"""
        if model_id not in self.models:
            return

        self.models[model_id].metrics.update(metrics)
        self.models[model_id].updated_at = time.time()

    def add_model_location(self, model_id: str, node_id: str):
        """Add node hosting a model"""
        if model_id not in self.model_locations:
            self.model_locations[model_id] = []

        if node_id not in self.model_locations[model_id]:
            self.model_locations[model_id].append(node_id)

    def get_model_locations(self, model_id: str) -> List[str]:
        """Get nodes hosting a model"""
        return self.model_locations.get(model_id, [])

    def calculate_model_reputation(self, model_id: str) -> float:
        """Calculate model reputation score"""
        metadata = self.models.get(model_id)
        if not metadata:
            return 0.0

        # Factors: accuracy, latency, usage count
        accuracy = metadata.metrics.get("accuracy", 0.5)
        latency_score = 1.0 / (1.0 + metadata.metrics.get("avg_latency", 100) / 100)
        usage_score = min(metadata.metrics.get("usage_count", 0) / 1000, 1.0)

        # Weighted average
        reputation = (accuracy * 0.5 + 
                     latency_score * 0.3 + 
                     usage_score * 0.2)

        return reputation

    def _fetch_from_network(self, model_id: str) -> Optional[bytes]:
        """Fetch model from network nodes"""
        # In production, this would query nodes storing the model
        logger.info(f"Fetching model {model_id} from network")
        return None

    def export_registry(self) -> Dict[str, Any]:
        """Export registry for synchronization"""
        return {
            "version": "1.0.0",
            "models": {
                model_id: {
                    "metadata": metadata.__dict__,
                    "locations": self.get_model_locations(model_id)
                }
                for model_id, metadata in self.models.items()
            },
            "timestamp": time.time()
        }

    def import_registry(self, registry_data: Dict[str, Any]):
        """Import registry data from another node"""
        for model_id, data in registry_data["models"].items():
            if model_id not in self.models:
                # Add new model metadata
                metadata_dict = data["metadata"]
                self.models[model_id] = ModelMetadata(**metadata_dict)

            # Update locations
            for node_id in data["locations"]:
                self.add_model_location(model_id, node_id)
