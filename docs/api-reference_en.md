# MESH API Reference

## Node API

### Starting a Node

```python
from mesh import MeshNode

# Create node with stake
node = MeshNode(stake=1000)

# Start node
await node.start(port=8080)
```

### Deploying Models

```python
# Deploy a model
await node.deploy_model(
    model_path="path/to/model.onnx",
    name="my-model",
    fee=0.1  # MESH tokens per inference
)
```

### Requesting Inference

```python
# Request inference from network
result = await node.request_inference(
    model_name="sentiment-analyzer",
    input_data={"text": "Hello world"}
)
```

## REST API Endpoints

### Node Status

```
GET /api/v1/status
```

Response:
```json
{
    "node_id": "abc123...",
    "version": "1.0.0",
    "peers": 42,
    "models": 3,
    "stake": 5000,
    "earnings": 123.45
}
```

### Model Registry

```
GET /api/v1/models
```

Query parameters:
- `name`: Filter by model name
- `max_fee`: Maximum fee in MESH
- `limit`: Number of results

### Submit Inference

```
POST /api/v1/inference
```

Request body:
```json
{
    "model": "sentiment-analyzer",
    "input": {
        "text": "This is amazing!"
    },
    "timeout": 30
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Subscribe to Events

```javascript
ws.send(JSON.stringify({
    "type": "subscribe",
    "events": ["inference", "peer_update", "model_announce"]
}));
```

### Real-time Inference

```javascript
ws.send(JSON.stringify({
    "type": "inference_stream",
    "model": "gpt-mini",
    "input": "Tell me a story",
    "stream": true
}));
```

## Python SDK

### Installation

```bash
pip install mesh-ai-network
```

### Basic Usage

```python
from mesh import MeshClient

# Initialize client
client = MeshClient()

# Find models
models = await client.search_models(
    name="sentiment",
    max_fee=0.1
)

# Run inference
result = await client.inference(
    model="sentiment-analyzer-v2",
    input_data={"text": "Great product!"}
)
```

### Advanced Features

```python
# Batch inference
results = await client.batch_inference([
    {"model": "model1", "input": data1},
    {"model": "model2", "input": data2}
])

# Model deployment
model_id = await client.deploy_model(
    model_data=model_bytes,
    metadata={
        "name": "my-model",
        "version": "1.0.0",
        "description": "Custom model"
    }
)
```

## Error Codes

| Code | Description |
|------|-------------|
| 1001 | Model not found |
| 1002 | Insufficient stake |
| 1003 | Timeout exceeded |
| 1004 | Validation failed |
| 2001 | Network error |
| 2002 | Peer unreachable |
| 3001 | Insufficient balance |
| 3002 | Transaction failed |

## Rate Limits

- **Inference requests**: 100/minute per IP
- **Model deployments**: 10/hour per node
- **API calls**: 1000/hour per token

## Webhooks

Configure webhooks for events:

```python
client.register_webhook(
    url="https://example.com/webhook",
    events=["inference_complete", "model_deployed"],
    secret="webhook_secret"
)
```
