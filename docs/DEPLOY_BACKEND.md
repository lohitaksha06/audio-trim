# Deploying the Audelle backend

The backend is a long-running FastAPI service with native audio libraries,
PyTorch/Demucs models and a background job pool. **Do not deploy it to Vercel
Functions.** Vercel is the right place for `website/`; the API needs a container
host with a persistent disk and enough RAM for Demucs.

The repository now includes:

- `Dockerfile` — CPU-only Python 3.12 image, ffmpeg and libsndfile included
- `.dockerignore`
- `scripts/warm_models.py` — downloads Demucs and Whisper weights once
- `AUDIDELLE_CORS_ORIGINS` support in `server/main.py`

## What the backend needs

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB for 6-stem Demucs |
| Disk | 10 GB persistent | 20 GB |
| Runtime | Always on | Always on |
| Network | HTTPS | Custom domain + HTTPS |
| ffmpeg | included in image | included in image |

The image is large because it contains PyTorch and the Python audio stack. The
Demucs and Whisper weights are downloaded into `/data` on first use; that is why
the volume must persist.

## Option A — Render (simplest first deployment)

1. Push this repository to GitHub.
2. In Render: **New → Web Service → Build from a Git repository**.
3. Choose the repository and **Docker** runtime.
4. Set **Dockerfile path** to `./Dockerfile`.
5. Choose an instance with at least **4 GB RAM** (8 GB if you expect 6-stem
   separation).
6. Add a persistent disk:
   - Mount path: `/data`
   - Size: 10–20 GB
7. Add environment variables:

```text
AUDIO_STORAGE_DIR=/data/storage
TORCH_HOME=/data/torch
HF_HOME=/data/huggingface
XDG_CACHE_HOME=/data/cache
AUDIDELLE_CORS_ORIGINS=https://audelle.vercel.app
```

8. Deploy. The health check is:

```text
GET https://<service>.onrender.com/api/health
```

9. Warm the models once from the Render shell:

```bash
python scripts/warm_models.py
```

10. Point the website at the API:
    - Vercel → `audelle` → Settings → Environment Variables
    - Add `NEXT_PUBLIC_API_URL=https://<service>.onrender.com`
    - Redeploy the website.

Free instances that spin down are not suitable: a cold start plus a model
download makes the first edit look broken, and free plans typically do not
support a persistent disk.

## Option B — Fly.io

```powershell
fly launch --no-deploy --name audelle-api
fly volumes create audelle_data --size 10
fly scale memory 4096
fly secrets set AUDIDELLE_CORS_ORIGINS="https://audelle.vercel.app"
fly deploy
```

Add the volume mount to the generated `fly.toml`:

```toml
[mounts]
  source = "audelle_data"
  destination = "/data"
```

Set the same `/data` environment variables as in Option A. Then run the warm
step once:

```powershell
fly ssh console -C "python /app/scripts/warm_models.py"
```

## Option C — AWS EC2 (most control)

A `t3.large` (2 vCPU, 8 GB) with a 30 GB gp3 volume is a reasonable CPU
starting point. For heavy Demucs use, a compute-optimized instance is better.
No GPU is required; the code explicitly runs on CPU when CUDA is absent.

```bash
# Ubuntu 24.04, Docker installed
sudo mkdir -p /srv/audelle-data
docker build -t audelle-api .
docker run -d --restart unless-stopped \
  --name audelle-api \
  -p 8000:8000 \
  -v /srv/audelle-data:/data \
  -e AUDIO_STORAGE_DIR=/data/storage \
  -e TORCH_HOME=/data/torch \
  -e HF_HOME=/data/huggingface \
  -e AUDIDELLE_CORS_ORIGINS=https://audelle.vercel.app \
  audelle-api
```

Put Caddy or Nginx in front for TLS, and set a request-body limit appropriate
for audio/video uploads (for example 500 MB). Warm the models:

```bash
docker exec -it audelle-api python scripts/warm_models.py
```

## Option D — Modal / RunPod / GPU workers

Use these only after moving uploads and job state out of the single container:

- object storage for uploads and exports
- Redis/Celery or a hosted queue instead of the in-memory `JobManager`
- a GPU worker or a CPU worker with more RAM
- authentication/rate limits in front of the API

The current `server/services/jobs.py` is deliberately an in-process thread pool.
It works on one instance and does not survive a restart or scale across replicas.

## Frontend configuration

The website reads the API origin at **build time**:

```text
NEXT_PUBLIC_API_URL=https://api.example.com
```

Set it in Vercel, then redeploy the site. Without it, the site deliberately
uses its local DSP paths.

## Security checklist before exposing it publicly

The current backend has **no authentication**. Do not publish it as-is on a
public URL with unrestricted uploads.

- Add authentication and per-user quotas.
- Add rate limits at the reverse proxy or API layer.
- Set an upload size limit (reverse proxy and FastAPI).
- Restrict CORS to the real site origin.
- Keep `/data` on a private volume; never expose it as a static directory.
- Clean up old uploads/exports; the storage layer has no expiry job.
- Put HTTPS and a WAF/rate limiter in front.
- Pin and review model licences; the drum bank licence is still unresolved.
- Log job failures without logging audio contents.

## Smoke test

```bash
curl https://api.example.com/api/health
curl -X POST https://api.example.com/api/upload -F "file=@sample.wav"
```

Then open `/prompt` on the Vercel site, upload a file, and try a request that
needs the server, such as `Separate into 4 stems`.

## Cost expectation

Demucs on CPU is the dominant cost. It can take minutes per track, and the
backend runs at most two heavy jobs at once. Model download and first-load time
are separate from inference. Measure the actual latency on the chosen instance
before promising a turnaround time.
