"""
Tests for core MESH components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.core import MeshNode, MeshNetwork, DynamicRouter
from src.token import TokenEconomics


class TestMeshNode:
    """Test MeshNode functionality"""

    @pytest.mark.asyncio
    async def test_node_creation(self):
        """Test node creation and initialization"""
        node = MeshNode(stake=1000)

        assert node.node_id is not None
        assert len(node.node_id) == 16
        assert node.stake == 1000
        assert not node.is_running

    @pytest.mark.asyncio
    async def test_node_start_stop(self):
        """Test node lifecycle"""
        node = MeshNode()

        # Start node
        await node.start(port=8080)
        assert node.is_running

        # Stop node
        await node.stop()
        assert not node.is_running

    @pytest.mark.asyncio
    async def test_model_deployment(self):
        """Test model deployment"""
        node = MeshNode()
        await node.start()

        # Deploy model
        await node.deploy_model(
            model_path="test.onnx",
            name="test-model",
            fee=0.1
        )

        assert "test-model" in node.hosted_models
        assert node.hosted_models["test-model"]["fee"] == 0.1

        await node.stop()

    @pytest.mark.asyncio
    async def test_inference_request(self):
        """Test inference request routing"""
        node = MeshNode()
        await node.start()

        # Mock route finding
        with patch.object(node, '_find_route', return_value=['node1', 'node2']):
            with patch.object(node, '_route_request', return_value={"result": "test"}):
                result = await node.request_inference("test-model", {"input": "data"})

        assert result["result"] == "test"

        await node.stop()


class TestMeshNetwork:
    """Test MeshNetwork functionality"""

    @pytest.mark.asyncio
    async def test_network_bootstrap(self):
        """Test network bootstrapping"""
        network = MeshNetwork()
        await network.bootstrap()

        assert network._is_running
        assert network.consensus is not None

    @pytest.mark.asyncio
    async def test_node_registration(self):
        """Test node registration"""
        network = MeshNetwork()
        await network.bootstrap()

        # Create node with sufficient stake
        node = MeshNode(stake=1000)

        # Mock consensus verification
        with patch.object(network.consensus, 'verify_node', return_value=True):
            result = await network.register_node(node)

        assert result is True
        assert node.node_id in network.nodes

    @pytest.mark.asyncio
    async def test_insufficient_stake(self):
        """Test node rejection for insufficient stake"""
        network = MeshNetwork()

        # Create node with insufficient stake
        node = MeshNode(stake=10)

        result = await network.register_node(node)
        assert result is False
        assert node.node_id not in network.nodes

    def test_network_stats(self):
        """Test network statistics calculation"""
        network = MeshNetwork()

        # Add test nodes
        for i in range(5):
            node = MeshNode(node_id=f"node_{i}", stake=1000)
            network.nodes[node.node_id] = node

        stats = network.get_network_stats()

        assert stats.total_nodes == 5
        assert stats.total_stake == 5000
        assert stats.network_health >= 0


class TestDynamicRouter:
    """Test routing functionality"""

    def test_route_calculation(self):
        """Test route calculation"""
        router = DynamicRouter()

        # Add node metrics
        router.update_node_metrics("node1", {
            "latency": 10,
            "reliability": 0.99,
            "load": 0.2
        })
        router.update_node_metrics("node2", {
            "latency": 20,
            "reliability": 0.95,
            "load": 0.5
        })

        # Find routes
        routes = router.find_routes("node1", "node2", k=1)

        assert len(routes) >= 1
        assert routes[0].path[0] == "node1"

    def test_node_failure_handling(self):
        """Test handling of node failures"""
        router = DynamicRouter()

        # Add nodes
        for i in range(3):
            router.update_node_metrics(f"node_{i}", {
                "latency": 10 + i * 5,
                "reliability": 0.95
            })

        # Simulate node failure
        router.handle_node_failure("node_1")

        assert "node_1" not in router.node_metrics


class TestTokenEconomics:
    """Test token economics"""

    def test_token_transfer(self):
        """Test token transfers"""
        economics = TokenEconomics()

        # Set initial balances
        economics.balances["alice"] = 1000
        economics.balances["bob"] = 500

        # Transfer tokens
        result = economics.transfer("alice", "bob", 100)

        assert result is True
        assert economics.get_balance("alice") == 900
        assert economics.get_balance("bob") == 600

    def test_insufficient_balance_transfer(self):
        """Test transfer with insufficient balance"""
        economics = TokenEconomics()
        economics.balances["alice"] = 50

        result = economics.transfer("alice", "bob", 100)

        assert result is False
        assert economics.get_balance("alice") == 50

    def test_staking(self):
        """Test token staking"""
        economics = TokenEconomics()
        economics.balances["alice"] = 5000

        # Stake tokens
        result = economics.stake("alice", 2000)

        assert result is True
        assert economics.get_balance("alice") == 3000
        assert economics.stakes["alice"].amount == 2000

    def test_reward_calculation(self):
        """Test staking reward calculation"""
        economics = TokenEconomics()

        # Setup stake
        economics.balances["alice"] = 5000
        economics.stake("alice", 1000)

        # Simulate time passing
        import time
        time.sleep(0.1)

        # Calculate rewards
        reward = economics.calculate_rewards(
            "alice",
            compute_contribution=0.5,
            uptime=1.0
        )

        assert reward > 0


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
