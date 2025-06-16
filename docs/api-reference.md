# LLMESH API 参考

## 节点 API

### 启动节点

```python
from llmesh import MeshNode

# 创建带有质押的节点
node = MeshNode(stake=1000)

# 启动节点
await node.start(port=8080)
```

### 部署模型

```python
# 部署模型
await node.deploy_model(
    model_path="path/to/model.onnx",
    name="my-model",
    fee=0.1  # 每次推理的 MESH 代币费用
)
```

### 请求推理

```python
# 从网络请求推理
result = await node.request_inference(
    model_name="sentiment-analyzer",
    input_data={"text": "你好世界"}
)
```

## REST API 端点

### 节点状态

```
GET /api/v1/status
```

响应：
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

### 模型注册表

```
GET /api/v1/models
```

查询参数：
- `name`：按模型名称过滤
- `max_fee`：最大费用（MESH）
- `limit`：结果数量

### 提交推理

```
POST /api/v1/inference
```

请求体：
```json
{
    "model": "sentiment-analyzer",
    "input": {
        "text": "这太棒了！"
    },
    "timeout": 30
}
```

## WebSocket API

### 连接

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### 订阅事件

```javascript
ws.send(JSON.stringify({
    "type": "subscribe",
    "events": ["inference", "peer_update", "model_announce"]
}));
```

### 实时推理

```javascript
ws.send(JSON.stringify({
    "type": "inference_stream",
    "model": "gpt-mini",
    "input": "给我讲个故事",
    "stream": true
}));
```

## Python SDK

### 安装

```bash
pip install llmesh-network
```

### 基本用法

```python
from llmesh import MeshClient

# 初始化客户端
client = MeshClient()

# 查找模型
models = await client.search_models(
    name="sentiment",
    max_fee=0.1
)

# 运行推理
result = await client.inference(
    model="sentiment-analyzer-v2",
    input_data={"text": "很棒的产品！"}
)
```

### 高级功能

```python
# 批量推理
results = await client.batch_inference([
    {"model": "model1", "input": data1},
    {"model": "model2", "input": data2}
])

# 模型部署
model_id = await client.deploy_model(
    model_data=model_bytes,
    metadata={
        "name": "my-model",
        "version": "1.0.0",
        "description": "自定义模型"
    }
)
```

## 错误代码

| 代码 | 描述 |
|------|-------------|
| 1001 | 未找到模型 |
| 1002 | 质押不足 |
| 1003 | 超时 |
| 1004 | 验证失败 |
| 2001 | 网络错误 |
| 2002 | 节点不可达 |
| 3001 | 余额不足 |
| 3002 | 交易失败 |

## 速率限制

- **推理请求**：每分钟 100 次/IP
- **模型部署**：每小时 10 次/节点
- **API 调用**：每小时 1000 次/令牌

## Webhook

配置事件的 webhook：

```python
client.register_webhook(
    url="https://example.com/webhook",
    events=["inference_complete", "model_deployed"],
    secret="webhook_secret"
)
```