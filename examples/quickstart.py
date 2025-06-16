"""
LLLLMESH Network Quick Start Example
Shows basic usage of the LLMESH network
"""

import asyncio
import logging
from src.core import MeshNode, MeshNetwork
from src.token import TokenEconomics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Quick start example for LLMESH network"""

    print("🚀 LLLLMESH Network Quick Start")
    print("=" * 50)

    # Initialize token economics
    economics = TokenEconomics()

    # Create network
    network = MeshNetwork()
    await network.bootstrap()

    # Create and start a node
    node = MeshNode(stake=1000)

    # Fund the node
    economics.balances[node.node_id] = 10000  # Initial balance
    economics.stake(node.node_id, 1000)  # Stake tokens

    print(f"\n📍 Node ID: {node.node_id}")
    print(f"💰 Balance: {economics.get_balance(node.node_id)} MESH")
    print(f"🔒 Staked: {economics.stakes[node.node_id].amount} MESH")

    # Start the node
    await node.start(port=8080)

    # Register with network
    if await network.register_node(node):
        print("✅ Node registered with network")
    else:
        print("❌ Failed to register node")
        return

    # Deploy a model
    print("\n🤖 Deploying LLM model...")
    await node.deploy_model(
        model_path="models/llama-mini.onnx",
        name="llama-mini",
        fee=0.1
    )

    # Simulate some activity
    print("\n📊 Running for 30 seconds...")
    await asyncio.sleep(30)

    # Check earnings
    compute_contribution = 0.5  # 50% of network compute
    uptime = 1.0  # 100% uptime

    rewards = economics.calculate_rewards(
        node.node_id,
        compute_contribution,
        uptime
    )

    print(f"\n💎 Earned rewards: {rewards:.2f} MESH")

    # Claim rewards
    claimed = economics.claim_rewards(
        node.node_id,
        compute_contribution,
        uptime
    )

    print(f"✅ Claimed: {claimed:.2f} MESH")
    print(f"💰 New balance: {economics.get_balance(node.node_id):.2f} MESH")

    # Get node stats
    stats = node.get_stats()
    print(f"\n📈 Node Statistics:")
    print(f"  - Peers: {stats['peers']}")
    print(f"  - Models: {stats['models']}")
    print(f"  - Requests: {stats['total_requests']}")
    print(f"  - Earnings: {stats['total_earnings']:.2f} MESH")

    # Network stats
    net_stats = network.get_network_stats()
    print(f"\n🌐 Network Statistics:")
    print(f"  - Total Nodes: {net_stats.total_nodes}")
    print(f"  - Total Stake: {net_stats.total_stake:.0f} MESH")
    print(f"  - Total Compute: {net_stats.total_compute:.0f} TFLOPS")
    print(f"  - Network Health: {net_stats.network_health:.1%}")

    # Shutdown
    print("\n🛑 Shutting down...")
    await node.stop()

    print("\n✅ Quick start completed!")


if __name__ == "__main__":
    asyncio.run(main())
