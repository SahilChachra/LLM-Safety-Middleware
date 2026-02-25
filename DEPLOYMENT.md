# LLM Safety Middleware — Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

---

## Overview

The LLM Safety Middleware is a **stateless proxy service** that sits in front
of a remote LLM backend.  It validates input prompts, forwards safe ones to
the LLM via async HTTP, validates the response, and returns it.

### Architecture

```
                        ┌─────────────────────────────────┐
                        │      Load Balancer (optional)    │
                        └───────────────┬─────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
       ┌───────▼───────┐       ┌────────▼───────┐       ┌───────▼───────┐
       │  Middleware 1  │       │  Middleware 2  │       │  Middleware 3  │
       │ (api_server)   │       │ (api_server)   │       │ (api_server)   │
       └───────┬────────┘       └───────┬────────┘       └───────┬───────┘
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        │  async HTTP
                              ┌─────────▼──────────┐
                              │   Remote LLM        │
                              │ (Ollama / OpenAI /  │
                              │  LM Studio / vLLM)  │
                              └─────────────────────┘
```

---

## Prerequisites

### System requirements

| | Minimum (dev) | Recommended (prod) |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 20 GB+ |
| GPU | Optional | NVIDIA 8GB+ VRAM (for classifier on GPU) |

### Software
- Python 3.10+
- Docker 20.10+ (for containerised deployment)
- A running LLM backend (Ollama, OpenAI, LM Studio, vLLM, etc.)

---

## Local Development

### 1. Set up environment

```bash
git clone https://github.com/SahilChachra/LLM-Safety-Middleware.git
cd LLM-Safety-Middleware

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment variables

```bash
# .env (do not commit to git)
LLM_BACKEND_TYPE=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama2
LLM_TIMEOUT_SECONDS=60
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.7
ADMIN_API_KEY=dev-secret-change-in-prod
LOG_LEVEL=DEBUG
```

### 3. Start LLM backend (Ollama example)

```bash
# Install Ollama: https://ollama.ai
ollama serve
ollama pull llama2
```

### 4. Run safety middleware

```bash
# Load env vars and start
export $(cat .env | xargs)
uvicorn api_server:app --reload --port 8000
# or
python api_server.py

# Interactive API docs
open http://localhost:8000/docs
```

### 5. Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/backend

curl -X POST http://localhost:8000/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "api_server.py"]
```

### Build and run

```bash
docker build -t llm-safety-middleware:latest .

docker run -d \
  --name safety-middleware \
  -p 8000:8000 \
  -e LLM_BACKEND_TYPE=ollama \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  -e LLM_MODEL=llama2 \
  -e ADMIN_API_KEY=changeme \
  -e CONFIG_PATH=/app/config_production.json \
  -v $(pwd)/config_production.json:/app/config_production.json:ro \
  -v $(pwd)/logs:/app/logs \
  llm-safety-middleware:latest

docker logs -f safety-middleware
```

### Docker Compose (middleware + Ollama)

```yaml
# docker-compose.yml
version: "3.9"
services:

  safety-middleware:
    build: .
    ports:
      - "8000:8000"
    environment:
      LLM_BACKEND_TYPE: ollama
      LLM_BASE_URL: http://ollama:11434
      LLM_MODEL: llama2
      LLM_TIMEOUT_SECONDS: "60"
      ADMIN_API_KEY: "${ADMIN_API_KEY}"
      CONFIG_PATH: /app/config_production.json
      WORKERS: "2"
    volumes:
      - ./config_production.json:/app/config_production.json:ro
      - ./logs:/app/logs
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  ollama_data:
```

```bash
docker-compose up -d
docker-compose logs -f safety-middleware

# Pull the model into Ollama
docker-compose exec ollama ollama pull llama2
```

---

## Production Deployment

### 1. Server setup (Ubuntu 22.04)

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# docker-compose plugin
sudo apt-get install docker-compose-plugin
```

### 2. Production config

Create `config_production.json` (see `config_examples.txt`):
```json
{
  "safety_threshold": 0.80,
  "toxicity_threshold": 0.75,
  "enable_rate_limiting": true,
  "max_requests_per_minute": 60,
  "log_level": "WARNING",
  "save_reports": true,
  "reports_dir": "/var/log/safety_reports"
}
```

### 3. Secrets

Never commit secrets.  Use environment files or a secrets manager:

```bash
# .env.production (chmod 600, owned by deploy user)
ADMIN_API_KEY=<strong-random-secret>
LLM_API_KEY=sk-...
```

### 4. Systemd service (alternative to Docker)

```ini
# /etc/systemd/system/safety-middleware.service
[Unit]
Description=LLM Safety Middleware
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/safety-middleware
EnvironmentFile=/opt/safety-middleware/.env.production
ExecStart=/opt/safety-middleware/venv/bin/python api_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable safety-middleware
sudo systemctl start safety-middleware
sudo systemctl status safety-middleware
```

### 5. Nginx reverse proxy with TLS

```nginx
# /etc/nginx/sites-available/safety-middleware
upstream safety_middleware {
    server 127.0.0.1:8000;
    # Horizontal scaling: add more backend addresses here
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://safety_middleware;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 10s;
        proxy_send_timeout    120s;   # allow for slow LLM responses
        proxy_read_timeout    120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/safety-middleware /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# TLS via Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

---

## Cloud Deployment

### AWS — ECS / Fargate

```bash
# Push image to ECR
aws ecr create-repository --repository-name llm-safety-middleware
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com

docker tag llm-safety-middleware:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/llm-safety-middleware:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/llm-safety-middleware:latest
```

ECS task definition environment variables:
```json
[
  {"name": "LLM_BACKEND_TYPE", "value": "openai"},
  {"name": "LLM_BASE_URL",     "value": "https://api.openai.com"},
  {"name": "LLM_MODEL",        "value": "gpt-4o"},
  {"name": "LLM_API_KEY",      "valueFrom": "arn:aws:secretsmanager:..."},
  {"name": "ADMIN_API_KEY",    "valueFrom": "arn:aws:secretsmanager:..."}
]
```

### Google Cloud — Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/llm-safety-middleware

gcloud run deploy safety-middleware \
  --image gcr.io/YOUR_PROJECT/llm-safety-middleware \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 120s \
  --max-instances 10 \
  --set-env-vars LLM_BACKEND_TYPE=openai,LLM_MODEL=gpt-4o \
  --set-secrets LLM_API_KEY=llm-api-key:latest,ADMIN_API_KEY=admin-key:latest
```

### Azure — Container Instances

```bash
az acr create --resource-group myRG --name myRegistry --sku Basic
az acr build --registry myRegistry --image llm-safety-middleware .

az container create \
  --resource-group myRG \
  --name safety-middleware \
  --image myRegistry.azurecr.io/llm-safety-middleware \
  --cpu 2 --memory 8 \
  --port 8000 \
  --environment-variables LLM_BACKEND_TYPE=openai LLM_MODEL=gpt-4o \
  --secure-environment-variables LLM_API_KEY=sk-... ADMIN_API_KEY=secret
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-middleware
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safety-middleware
  template:
    metadata:
      labels:
        app: safety-middleware
    spec:
      containers:
      - name: safety-middleware
        image: llm-safety-middleware:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_BACKEND_TYPE
          value: "openai"
        - name: LLM_BASE_URL
          value: "https://api.openai.com"
        - name: LLM_MODEL
          value: "gpt-4o"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: api-key
        - name: ADMIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: admin-key
        - name: CONFIG_PATH
          value: "/app/config_production.json"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config_production.json
          subPath: config_production.json
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: safety-config
---
apiVersion: v1
kind: Service
metadata:
  name: safety-middleware
spec:
  selector:
    app: safety-middleware
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Monitoring & Maintenance

### Health checks

```bash
# Middleware health
curl http://localhost:8000/health

# Remote LLM reachability
curl http://localhost:8000/health/backend

# Shell health-check script
#!/bin/bash
STATUS=$(curl -sf http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
if [ "$STATUS" != "healthy" ]; then
    echo "Service unhealthy! Restarting..."
    systemctl restart safety-middleware
fi
```

### Statistics endpoint

```bash
curl http://localhost:8000/api/v1/statistics
```

Returns: total requests, accepted, rejected, rejection reasons, avg processing time, uptime.

### Logging

The server emits structured logs to stdout.  Pipe to your preferred collector:

```bash
# journald (systemd)
journalctl -u safety-middleware -f

# Docker
docker logs -f safety-middleware

# Ship to Loki / CloudWatch / Datadog via log driver
docker run ... --log-driver=awslogs \
  --log-opt awslogs-group=/safety-middleware ...
```

### Backup safety reports

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf /backups/safety-reports-$DATE.tar.gz /var/log/safety_reports
find /backups -name "safety-reports-*.tar.gz" -mtime +30 -delete
```

---

## Security Considerations

1. **Set `ADMIN_API_KEY`** in production — without it the `/statistics/reset` endpoint returns 503
2. **Use TLS** — run behind Nginx / ALB with a valid certificate; never expose plain HTTP
3. **Restrict CORS** — set `ALLOWED_ORIGINS` to your front-end domain(s); default is no cross-origin access
4. **Rate limiting** — enable `enable_rate_limiting=True` and tune `max_requests_per_minute`
5. **Firewall** — only expose port 443 (and 22 for SSH); block direct port 8000 from external traffic
6. **Secrets management** — use AWS Secrets Manager, GCP Secret Manager, Vault, or `.env` with `chmod 600`
7. **Least privilege** — run the container as a non-root user (see Dockerfile above)
8. **Model trust** — only use trusted HuggingFace model sources for the safety classifier
9. **Audit logs** — keep `save_reports=True` and archive reports for compliance
10. **Dependency updates** — pin and regularly update `requirements.txt` to patch vulnerabilities

---

## Troubleshooting

### Backend unreachable on startup

```bash
# Check Ollama is running and model is available
curl http://localhost:11434/api/tags

# Check connectivity from inside the container
docker exec safety-middleware curl http://host.docker.internal:11434/api/tags

# Probe via API
curl http://localhost:8000/health/backend
```

### Out of memory

```bash
# Check container memory usage
docker stats safety-middleware

# Force CPU mode for the safety classifier
# Add to config_production.json:
#   "device": "cpu"

# Or increase container memory limit
docker run -m 8g llm-safety-middleware:latest
```

### Slow responses

- Check `avg_processing_time` in `/api/v1/statistics`
- Disable semantic check and/or post-gen check if latency is critical:
  ```json
  {"enable_semantic_check": false, "enable_post_generation_check": false}
  ```
- Check LLM backend latency independently: `curl http://localhost:11434/api/generate ...`
- Scale horizontally: add more middleware replicas behind the load balancer

### High rejection rate

```bash
# See which layer is triggering rejections
curl http://localhost:8000/api/v1/statistics | jq '.rejection_reasons'

# Loosen thresholds for development / lower-risk environments
# (see config_examples.txt)
```

### Rate limit errors for legitimate traffic

```json
{
  "enable_rate_limiting": true,
  "max_requests_per_minute": 120
}
```

---

**Version**: 2.0.0
**Last Updated**: 2026-02-24
