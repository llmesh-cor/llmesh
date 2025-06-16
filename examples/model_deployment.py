"""
Model Deployment Example
Shows how to deploy and manage LLMs on LLMESH
"""

import asyncio
import numpy as np
from pathlib import Path

from src.core import MeshNode
from src.ai import ModelRegistry, InferenceEngine
from src.token import TokenEconomics


async def create_example_llm():
    """Create an example LLM model"""
    # In production, this would be a real LLM
    # For demo, we'll create mock model data

    print("🔨 Creating example LLM...")

    # Mock ONNX model data
    model_data = b"ONNX_LLM_MODEL_DATA_PLACEHOLDER" * 10000  # ~300KB

    # Save to file
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "llama_mini.onnx"
    model_path.write_bytes(model_data)

    print(f"✅ Model saved to {model_path}")
    return model_path


async def deploy_llm_example():
    """Example of deploying an LLM to LLMESH network"""

    print("🚀 LLMESH LLM Deployment Example")
    print("=" * 50)

    # Initialize components
    economics = TokenEconomics()
    node = MeshNode(stake=2000)
    registry = ModelRegistry()
    inference_engine = InferenceEngine(node.node_id)

    # Fund and stake
    economics.balances[node.node_id] = 5000
    economics.stake(node.node_id, 2000)

    # Start node
    await node.start()

    # Create example LLM
    model_path = await create_example_llm()

    # Deploy LLM
    print("\n📤 Deploying LLM to network...")

    model_data = model_path.read_bytes()

    # Register model
    model_id = registry.register_model(
        name="llama-mini-v1",
        model_data=model_data,
        format="onnx",
        owner=node.node_id,
        fee=0.05,  # 0.05 MESH per inference
        description="Compact LLM for text generation and chat",
        metrics={
            "parameters": "125M",
            "context_length": 2048,
            "languages": "en,es,fr,de",
            "avg_latency": 45.5  # ms
        }
    )

    print(f"✅ LLM registered: {model_id}")

    # Load model into inference engine
    await inference_engine.load_model(model_id, model_data, "onnx")

    # Deploy to node
    await node.deploy_model(
        str(model_path),
        "llama-mini-v1",
        fee=0.05
    )

    # Simulate inference requests
    print("\n🧪 Simulating LLM inference requests...")

    prompts = [
        "Tell me about artificial intelligence",
        "Write a poem about decentralization",
        "Explain blockchain in simple terms",
        "What is the future of AI?",
        "Generate a story about mesh networks"
    ]

    for i, prompt in enumerate(prompts):
        # Create inference request
        request = {
            "request_id": f"req_{i}",
            "model_id": model_id,
            "input_data": {
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7
            },
            "requester": f"client_{i}",
            "timestamp": asyncio.get_event_loop().time()
        }

        # Process inference
        print(f"\n  Processing request {i+1}/5: '{prompt[:30]}...'")

        try:
            # Simulate inference
            result = await node.request_inference(
                "llama-mini-v1",
                request["input_data"]
            )

            print(f"  ✅ Generated response")

            # Update model metrics
            registry.update_model_metrics(model_id, {
                "usage_count": i + 1,
                "total_tokens": (i + 1) * 100,
                "last_used": asyncio.get_event_loop().time()
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")

        await asyncio.sleep(1)

    # Check model stats
    print("\n📊 LLM Statistics:")
    metadata, _ = registry.get_model(model_id)

    print(f"  - Model ID: {metadata.model_id}")
    print(f"  - Owner: {metadata.owner[:16]}...")
    print(f"  - Fee: {metadata.fee} MESH")
    print(f"  - Size: {metadata.size / 1024:.1f} KB")
    print(f"  - Parameters: {metadata.metrics.get('parameters', 'N/A')}")
    print(f"  - Context Length: {metadata.metrics.get('context_length', 'N/A')}")
    print(f"  - Usage Count: {metadata.metrics.get('usage_count', 0)}")

    # Calculate reputation
    reputation = registry.calculate_model_reputation(model_id)
    print(f"  - Reputation Score: {reputation:.2f}/1.0")

    # Search for LLMs
    print("\n🔍 Searching for LLMs...")
    results = registry.search_models(name="llama", max_fee=0.1)

    print(f"Found {len(results)} LLMs:")
    for model in results:
        print(f"  - {model.name} (Fee: {model.fee} MESH)")

    # Export registry for sharing
    registry_data = registry.export_registry()
    print(f"\n💾 Registry exported ({len(registry_data['models'])} models)")

    # Shutdown
    await node.stop()
    print("\n✅ LLM deployment example completed!")


async def federated_llm_training_example():
    """Example of federated LLM training on LLMESH"""

    print("\n\n🤝 Federated LLM Training Example")
    print("=" * 50)

    from src.ai import FederatedLearning

    # Create coordinator node
    coordinator = MeshNode(node_id="coordinator", stake=5000)
    fed_learning = FederatedLearning(coordinator.node_id)

    # Create federated learning task for LLM fine-tuning
    task_id = await fed_learning.create_task(
        model_id="llama-mini-v1",
        rounds=3,
        min_participants=2,
        aggregation_method="fedavg"
    )

    print(f"\n✅ Created federated LLM training task: {task_id}")

    # Simulate participants with private data
    participants = []
    datasets = [
        "medical_conversations",
        "legal_documents", 
        "technical_support"
    ]

    for i, dataset in enumerate(datasets):
        participant = MeshNode(node_id=f"hospital_{i}", stake=1000)
        participant_fl = FederatedLearning(participant.node_id)

        # Join task
        await participant_fl.join_task(task_id)
        participants.append((participant_fl, dataset))

        print(f"✅ {dataset} dataset joined training")

    # Simulate training rounds
    for round_num in range(3):
        print(f"\n🔄 Training Round {round_num + 1}/3")

        # Each participant trains locally on private data
        for i, (participant, dataset) in enumerate(participants):
            print(f"  Training on {dataset}...")

            # Simulate local LLM fine-tuning
            update = await participant.train_local_model(
                task_id,
                training_data=dataset,  # Private data stays local
                epochs=5
            )

            # Submit update to coordinator
            await fed_learning.submit_update(update)

        # Check task status
        status = fed_learning.get_task_status(task_id)
        print(f"\n  Task Status: {status}")

        await asyncio.sleep(1)

    print("\n✅ Federated LLM training completed!")
    print("   The LLM has been fine-tuned on distributed private data")
    print("   without any data leaving the local nodes!")


async def main():
    """Run all examples"""
    await deploy_llm_example()
    await federated_llm_training_example()


if __name__ == "__main__":
    asyncio.run(main())
