# Disease Prediction React Frontend

React dashboard for visualizing and testing prediction APIs.

## Run Locally

1. Start FastAPI backend.
2. In this `frontend` folder run:

```bash
npm install
npm run dev
```

3. Open `http://127.0.0.1:5173`.

## Backend Proxy

By default, Vite proxies `/predict/*` and `/health` to `http://127.0.0.1:8000`.

If your backend runs on a different port, create `.env` in this folder:

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8010
```
