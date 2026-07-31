# Deployment Guide

## Overview

Synapse is distributed inference infrastructure for Mixture-of-Experts (MoE) models. This guide covers deploying Synapse in various environments.

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA 12.0+ (for GPU inference)
- 16GB+ RAM
- 50GB+ disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Build the gateway
cargo build --release

# Run the gateway
./target/release/synapse-core
```

### Docker

```bash
# Build the Docker image
docker build -t synapse .

# Run the container
docker run -p 8000:8000 synapse
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_PORT` | `8000` | Gateway port |
| `SYNAPSE_HOST` | `0.0.0.0` | Gateway host |
| `SYNAPSE_LOG_LEVEL` | `info` | Log level (trace, debug, info, warn, error) |
| `SYNAPSE_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |

### Configuration File

Create `config/default.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8000

[ollama]
url = "http://localhost:11434"
model = "granite3.1-moe:3b"

[logging]
level = "info"
```

## Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "swarm_nodes": 0
}
```

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Response:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "kimi-k3",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }]
}
```

### Jobs API

#### Create Job

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3b-moe",
    "messages": [{"role": "user", "content": "hello"}],
    "priority": "normal"
  }'
```

Response:
```json
{
  "job_id": "uuid"
}
```

#### Poll Job

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

Response:
```json
{
  "job_id": "uuid",
  "object": "job",
  "status": "completed",
  "model": "granite-3b-moe",
  "result": {
    "text": "...",
    "tokens": 100
  },
  "created_at": "2026-07-31T12:00:00Z",
  "updated_at": "2026-07-31T12:00:01Z"
}
```

### Metrics (Prometheus)

```bash
curl http://localhost:8000/metrics
```

Response:
```
# HELP synapse_jobs_total Total jobs submitted
# TYPE synapse_jobs_total counter
synapse_jobs_total 42
```

### Models Catalog

```bash
curl http://localhost:8000/v1/models
```

Response:
```json
[
  {
    "id": "kimi-k3",
    "name": "Kimi K3",
    "parameters": "1T",
    "experts": 64
  }
]
```

## Monitoring

### Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'synapse'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana

Import the Synapse dashboard from `docs/grafana/synapse.json`.

## Multi-Machine Deployment

### TCP Transport

Workers can run on separate machines:

```bash
# Machine 1: Gateway
./target/release/synapse-core --port 8000

# Machine 2: Worker
./target/release/expert_worker model.gguf 0 1 2 --port 8001
```

### mDNS Discovery

Workers auto-discover on LAN:

```toml
[discovery]
enabled = true
mdns = true
```

## Security

### TLS (Future)

TLS support is planned for production deployment. See issue #62.

### Authentication (Future)

API key authentication is planned. See issue #63.

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   lsof -i :8000
   kill -9 <PID>
   ```

2. **CUDA not found**
   ```bash
   nvidia-smi
   export CUDA_HOME=/usr/local/cuda
   ```

3. **Out of memory**
   ```bash
   # Reduce batch size
   export SYNAPSE_BATCH_SIZE=1
   ```

### Logs

```bash
# Enable debug logging
export SYNAPSE_LOG_LEVEL=debug

# View logs
journalctl -u synapse -f
```

## Performance Tuning

### GPU Memory

- Mixtral 8x7B: ~16GB VRAM
- Mixtral 8x22B: ~48GB VRAM

### CPU Inference

For CPU-only inference:

```toml
[inference]
device = "cpu"
threads = 8
```

## Next Steps

- [ ] Add TLS support (#62)
- [ ] Add authentication (#63)
- [ ] Add dynamic expert loading (#64)
- [ ] Add P2P expert discovery (#65)
