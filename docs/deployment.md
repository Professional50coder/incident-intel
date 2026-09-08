# Deployment

This is two separate deployables, and they belong on two different kinds of host.

## Dashboard → Vercel

The Next.js dashboard (`dashboard/`) is a normal Vercel deployment: point Vercel at this repo
with `dashboard` as the project root, no extra config needed. Set one environment variable in
the Vercel project settings:

```
NEXT_PUBLIC_API_URL=https://<wherever-the-backend-ends-up>
```

## Backend → not Vercel

**The FastAPI/PyTorch backend should not be deployed to Vercel serverless functions.** This
isn't a preference, it's a real fit problem:

- **Package size.** Even the CPU-only PyTorch build is well over Vercel's ~250MB (unzipped)
  serverless function limit. The CUDA build used for local training is several GB.
- **Cold starts.** A serverless function loading a PyTorch model from scratch on every cold
  start adds real, visible latency to `/analyze` - the opposite of what a "real-time" feeling
  dashboard needs.
- **Ephemeral, read-only filesystem.** The backend persists incident history to a SQLite file
  on disk (`data/incident_intel.db`). Serverless functions don't guarantee a writable,
  persistent filesystem between invocations - every "recent incidents" list would be empty or
  inconsistent.
- **Execution time limits.** Sampling and scoring frames from a video upload can take longer
  than a serverless function's execution budget, especially on the Hobby tier.

**What to use instead:** the `Dockerfile` in the repo root builds a small, portable CPU-only
image (inference on this model doesn't need a GPU - only training benefited from one). Any
container host works: a VPS with Docker Compose, Railway, Render, or Fly.io. Example for a
plain VPS:

```bash
docker build -t incident-intel-api .
docker run -d -p 8000:8000 \
  -e DASHBOARD_ORIGIN=https://your-dashboard.vercel.app \
  -v $(pwd)/data:/app/data \
  incident-intel-api
```

The `-v` mount keeps `data/incident_intel.db` (incident history) persistent across container
restarts - without it, history resets every deploy. `DASHBOARD_ORIGIN` must match the deployed
dashboard's URL exactly, or the browser will block the API calls as cross-origin (it defaults
to `http://localhost:3000` for local dev).

Once the backend has a public URL, set it as `NEXT_PUBLIC_API_URL` in the Vercel project and
redeploy the dashboard.
