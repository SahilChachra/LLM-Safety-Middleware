# Quick Start Guide — LLM Safety Middleware

Get up and running in 5 minutes!

## Architecture in one sentence

The middleware sits **between your client and a remote LLM**.  Every request
is safety-checked before being forwarded; every response is safety-checked
before being returned.

```
Client → Safety Middleware → Remote LLM (Ollama / OpenAI / custom)
```

---

## 🚀 Installation (2 minutes)

### Option 1: Local (uv recommended)

```bash
# 1. Clone repository
git clone https://github.com/SahilChachra/LLM-Safety-Middleware.git
cd LLM-Safety-Middleware

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Install spaCy language model
python -m spacy download en_core_web_sm
# or (if pip is unavailable in the venv):
# uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### Option 2: Docker

```bash
git clone https://github.com/SahilChachra/LLM-Safety-Middleware.git
cd LLM-Safety-Middleware
docker-compose up -d
```

---

## 📝 Basic Usage (3 minutes)

### Example 1: Input safety check only (sync)

```python
from llm_safety_pipeline import SafetyPipeline

pipeline = SafetyPipeline()

status, report = pipeline.process("What is artificial intelligence?")

print(f"Status: {status}")              # ACCEPTED / REJECTED
print(f"Safety Level: {report.safety_level.name}")
```

**Output:**
```
Status: ACCEPTED
Safety Level: SAFE
```

### Example 2: Full pipeline — check input, call LLM, check output (async)

```python
import asyncio
from llm_safety_pipeline import LLMBackendConfig, SafetyPipeline

backend = LLMBackendConfig(
    backend_type="ollama",          # "ollama" | "openai" | "custom"
    base_url="http://localhost:11434",
    model="llama2",
)
pipeline = SafetyPipeline(backend_config=backend)

async def main():
    status, report = await pipeline.async_process(
        "Write a paragraph about renewable energy.",
        generation_kwargs={"max_new_tokens": 150},
    )
    if status == "ACCEPTED":
        print(report.generated_text)
    else:
        print(f"Rejected: {report.rejection_reason.value}")
    await pipeline.async_close()

asyncio.run(main())
```

### Example 3: One-line safety check

```python
from llm_safety_pipeline import quick_check

result = quick_check("How to bake a cake?")
print(f"Safe: {result['is_safe']}, Score: {result['safety_score']}")
```

---

## 🔧 Configuration

### Safety layers (JSON file)

```python
from llm_safety_pipeline import SafetyConfig, SafetyPipeline

config = SafetyConfig(
    safety_threshold=0.80,          # Higher = stricter
    enable_rate_limiting=True,
    max_requests_per_minute=60,
)
pipeline = SafetyPipeline(config)
```

Load from file:
```python
config = SafetyConfig.from_json("config_production.json")
pipeline = SafetyPipeline(config)
```

**config_production.json:**
```json
{
  "safety_threshold": 0.80,
  "toxicity_threshold": 0.75,
  "enable_rate_limiting": true,
  "max_requests_per_minute": 60,
  "log_level": "WARNING",
  "save_reports": true
}
```

### LLM backend (env vars or code)

```python
from llm_safety_pipeline import LLMBackendConfig

# From code
backend = LLMBackendConfig(
    backend_type="openai",
    base_url="https://api.openai.com",
    model="gpt-4o",
    api_key="sk-...",
    max_new_tokens=512,
    temperature=0.7,
)

# From environment variables (used by api_server.py)
backend = LLMBackendConfig.from_env()
```

| Env var | Default | Description |
|---|---|---|
| `LLM_BACKEND_TYPE` | `ollama` | `ollama` / `openai` / `custom` |
| `LLM_BASE_URL` | `http://localhost:11434` | Remote LLM server URL |
| `LLM_MODEL` | `llama2` | Model name |
| `LLM_API_KEY` | — | Bearer token (OpenAI etc.) |
| `LLM_MAX_NEW_TOKENS` | `512` | Default token budget |
| `LLM_TEMPERATURE` | `0.7` | Default temperature |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on 5xx |

---

## 🌐 REST API

### Start the server

```bash
# Direct
python api_server.py

# With uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000

# With env vars
LLM_BACKEND_TYPE=ollama LLM_BASE_URL=http://localhost:11434 \
  LLM_MODEL=llama2 python api_server.py
```

### Endpoints

#### 1. Input safety check (no LLM call)
```bash
curl -X POST http://localhost:8000/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "client_id": "user_123"}'
```

**Response:**
```json
{
  "status": "ACCEPTED",
  "safety_level": "SAFE",
  "safety_score": 0.92,
  "rejection_reason": null,
  "matched_patterns": [],
  "processing_time": 0.045,
  "timestamp": "2026-02-24T12:00:00"
}
```

#### 2. Full generation (check → LLM → check)
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write about AI ethics.", "max_new_tokens": 100}'
```

**Response:**
```json
{
  "status": "ACCEPTED",
  "generated_text": "Artificial intelligence ethics is...",
  "safety_level": "SAFE",
  "rejection_reason": null,
  "generation_time": 1.23,
  "processing_time": 1.28,
  "timestamp": "2026-02-24T12:00:00"
}
```

#### 3. Backend health probe
```bash
curl http://localhost:8000/health/backend
```

#### 4. Statistics
```bash
curl http://localhost:8000/api/v1/statistics
```

#### 5. Reset statistics (requires `ADMIN_API_KEY`)
```bash
curl -X POST http://localhost:8000/api/v1/statistics/reset \
  -H "X-API-Key: <ADMIN_API_KEY>"
```

#### 6. Monitoring dashboard (browser)

Open **http://localhost:8000/dashboard** in your browser for a live visual
overview of all the above — no extra tools needed. The dashboard polls the API
every 10 seconds and shows request counts, accept rate, rejection reasons,
latency, and backend health.

![Dashboard preview](docs/dashboard-preview.png)

---

## 📊 Common Use Cases

### Content moderation

```python
from llm_safety_pipeline import SafetyPipeline

pipeline = SafetyPipeline()

def moderate_content(user_input: str) -> dict:
    status, report = pipeline.process(user_input)
    if status == "ACCEPTED":
        return {"allowed": True, "score": report.safety_score}
    return {
        "allowed": False,
        "reason": report.rejection_reason.value,
        "score": report.safety_score,
    }

print(moderate_content("How to bake a cake?"))
```

### Safe async chatbot

```python
import asyncio
from llm_safety_pipeline import LLMBackendConfig, SafetyPipeline

pipeline = SafetyPipeline(
    backend_config=LLMBackendConfig.from_env()
)

async def safe_reply(user_message: str) -> str:
    status, report = await pipeline.async_process(user_message)
    if status == "ACCEPTED":
        return report.generated_text or ""
    return "I cannot respond to that request."

# asyncio.run(safe_reply("Tell me about climate change."))
```

### Batch safety screening

```python
from llm_safety_pipeline import SafetyPipeline

pipeline = SafetyPipeline()

prompts = [
    "What is Python programming?",
    "Explain quantum computing.",
    "How do neural networks work?",
]

for prompt in prompts:
    status, report = pipeline.process(prompt)
    print(f"{'✅' if status == 'ACCEPTED' else '❌'} {prompt}: {status}")
```

---

## 🎯 Testing

### Run demo

```bash
python demo_safety_pipeline.py
```

### Run tests

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio respx

# Run all 74 tests
pytest test_safety_pipeline.py -v

# With coverage
pytest --cov=llm_safety_pipeline --cov-report=html
```

---

## 🔍 Monitoring

### Health check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/backend   # probes the remote LLM
```

### Statistics

```python
stats = pipeline.get_statistics()
print(f"Total: {stats['total_requests']}")
print(f"Accepted: {stats['accepted']}")
print(f"Rejected: {stats['rejected']}")
print(f"Rejection reasons: {stats['rejection_reasons']}")
```

---

## 🛠️ Customisation

### Custom banned patterns

```python
config = SafetyConfig(
    custom_banned_patterns=[r'\bcustom_dangerous_word\b'],
    custom_allowed_contexts=[r'\bsafe_research_context\b'],
)
pipeline = SafetyPipeline(config)
```

### Adjust thresholds

```python
config = SafetyConfig(
    safety_threshold=0.90,      # Very strict
    toxicity_threshold=0.85,
)
```

### Per-request generation overrides

```python
status, report = await pipeline.async_process(
    prompt,
    generation_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.9,
    },
)
```

---

## 🐛 Troubleshooting

### Backend unreachable
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check with the health endpoint
curl http://localhost:8000/health/backend
```

### Models not downloaded (safety classifier)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained("unitary/toxic-bert")
AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
```

### Out of memory (safety classifier on GPU)
```python
config = SafetyConfig(device="cpu")
```

### Disable slow checks for speed
```python
config = SafetyConfig(
    enable_semantic_check=False,
    enable_post_generation_check=False,
)
```

---

## 📚 Next Steps

1. Read the full documentation in `README.md`
2. Browse examples in `demo_safety_pipeline.py`
3. Follow `DEPLOYMENT.md` for production setup
4. Adjust `config_examples.txt` for your environment
5. Check `http://localhost:8000/docs` for interactive API docs

---

## ✅ Pre-production Checklist

- [ ] LLM backend configured and reachable (`/health/backend`)
- [ ] Appropriate safety thresholds set and tested
- [ ] Rate limiting enabled
- [ ] `ADMIN_API_KEY` set for statistics reset endpoint
- [ ] CORS configured via `ALLOWED_ORIGINS`
- [ ] Logging configured
- [ ] Monitoring / alerting set up
- [ ] Load tested

---

**Ready to build safe AI applications!**

- Full documentation: `README.md`
- Deployment guide: `DEPLOYMENT.md`
- Interactive API docs: `http://localhost:8000/docs`
