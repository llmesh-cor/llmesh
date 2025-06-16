"""
Complete Node Setup Example
Shows how to set up and configure a LLMESH node
"""

import asyncio
import logging
from pathlib import Path

from src.core import MeshNode
from src.p2p import P2PTransport, PeerDiscovery
from src.ai import ModelRegistry, InferenceEngine
from src.token import TokenEconomics, StakeManager

logger = logging.getLogger(__name__)


class LLMeshNodeOperator:
    """Helper class for node operators"""

    def __init__(self, config_path: str = "llmesh.config.yaml"):
        """Initialize node operator"""
        self.config_path = config_path
        self.node = None
        self.transport = None
        self.discovery = None
        self.registry = None
        self.inference = None
        self.economics = None

    async def setup_node(self):
        """Complete node setup process"""
        print("🔧 LLMESH Node Setup")
        print("=" * 50)

        # Step 1: Load configuration
        config = self.load_config()

        # Step 2: Initialize components
        print("\n📦 Initializing components...")

        # Token economics
        self.economics = TokenEconomics()
        stake_manager = StakeManager(self.economics)

        # Create node with stake
        stake_amount = config.get("stake_amount", 1000)
        self.node = MeshNode(stake=stake_amount)

        # Fund node (in production, tokens come from wallet)
        self.economics.balances[self.node.node_id] = stake_amount * 2
        self.economics.stake(self.node.node_id, stake_amount)

        print(f"✅ Node ID: {self.node.node_id}")
        print(f"💰 Staked: {stake_amount} MESH")

        # Step 3: Setup P2P transport
        print("\n🌐 Setting up P2P transport...")
        self.transport = P2PTransport(
            self.node.node_id,
            port=config.get("p2p_port", 8888)
        )

        # Register message handlers
        self.setup_message_handlers()

        await self.transport.start()
        print("✅ P2P transport started")

        # Step 4: Setup peer discovery
        print("\n🔍 Setting up peer discovery...")
        bootstrap_nodes = config.get("bootstrap_nodes", [])
        self.discovery = PeerDiscovery(
            self.node.node_id,
            bootstrap_nodes
        )

        await self.discovery.start()
        print("✅ Peer discovery started")

        # Step 5: Setup model registry
        print("\n📚 Setting up model registry...")
        self.registry = ModelRegistry()

        # Step 6: Setup inference engine
        print("\n🧠 Setting up inference engine...")
        self.inference = InferenceEngine(self.node.node_id)

        # Step 7: Start node
        print("\n🚀 Starting LLMESH node...")
        await self.node.start(port=config.get("api_port", 8080))

        # Step 8: Connect to bootstrap nodes
        if bootstrap_nodes:
            print("\n🔗 Connecting to network...")
            for bootstrap in bootstrap_nodes:
                host, port = bootstrap.split(":")
                await self.connect_to_peer("bootstrap", host, int(port))

        print("\n✅ Node setup complete!")

    def load_config(self):
        """Load node configuration"""
        # In production, load from YAML file
        return {
            "stake_amount": 5000,
            "p2p_port": 8888,
            "api_port": 8080,
            "bootstrap_nodes": [
                "bootstrap1.llmesh.network:8888",
                "bootstrap2.llmesh.network:8888"
            ],
            "models_dir": "models/",
            "data_dir": "data/"
        }

    def setup_message_handlers(self):
        """Setup P2P message handlers"""

        async def handle_inference_request(message):
            """Handle incoming inference request"""
            request_data = message.payload
            logger.info(f"Received inference request: {request_data['model_name']}")

            # Process inference
            # ... inference logic ...

        self.transport.register_handler("inference_request", handle_inference_request)

    async def connect_to_peer(self, peer_id: str, host: str, port: int):
        """Connect to a peer"""
        try:
            await self.transport.connect_to_peer(peer_id, host, port)
            print(f"✅ Connected to {peer_id} at {host}:{port}")
        except Exception as e:
            print(f"❌ Failed to connect to {peer_id}: {e}")

    async def deploy_llm(self, model_path: str, name: str, fee: float = 0.1):
        """Deploy an LLM to the network"""
        print(f"\n🤖 Deploying LLM: {name}")

        # Load model file
        model_data = Path(model_path).read_bytes()

        # Register in local registry
        model_id = self.registry.register_model(
            name=name,
            model_data=model_data,
            format="onnx",
            owner=self.node.node_id,
            fee=fee,
            description="Large Language Model"
        )

        # Load into inference engine
        await self.inference.load_model(model_id, model_data, "onnx")

        # Deploy to node
        await self.node.deploy_model(model_path, name, fee)

        print(f"✅ LLM deployed with ID: {model_id}")

    async def run(self):
        """Run the node"""
        print("\n🏃 Node is running...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Print periodic stats
                stats = self.node.get_stats()
                print(f"\r📊 Peers: {stats['peers']} | "
                      f"Models: {stats['models']} | "
                      f"Requests: {stats['total_requests']} | "
                      f"Earnings: {stats['total_earnings']:.2f} MESH", 
                      end="", flush=True)

                await asyncio.sleep(5)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")

    async def shutdown(self):
        """Shutdown the node"""
        if self.node:
            await self.node.stop()
        if self.transport:
            await self.transport.stop()

        print("✅ Node shutdown complete")


async def main():
    """Main entry point"""
    operator = LLMeshNodeOperator()

    try:
        # Setup node
        await operator.setup_node()

        # Deploy example LLM
        await operator.deploy_llm(
            "models/llama-mini.onnx",
            "llama-mini",
            fee=0.1
        )

        # Run node
        await operator.run()

    finally:
        await operator.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
