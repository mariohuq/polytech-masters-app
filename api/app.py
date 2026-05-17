from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.generator import MockSeriesGenerator
from api.registry_routes import router as registry_router
from models import DetectorKind, UnifiedAnomalyDetector, list_models

# Параметры стрима (не влияют на воспроизводимость ряда — ряд задаёт только seed в генераторе)
_STREAM_WARMUP = 64
_STREAM_SLEEP_MS = 50.0
_STREAM_MAX_BUFFER = 512

app = FastAPI(title="Polytech masters — mock stream API", version="0.1.0")
app.include_router(registry_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _check_readiness() -> dict[str, Any]:
    probe = np.zeros((8, 2), dtype=float)
    checked: list[str] = []
    for kind in DetectorKind:
        det = UnifiedAnomalyDetector(kind)
        det.fit(probe)
        _ = det.predict(probe)
        _ = det.predict_proba(probe)
        checked.append(kind.value)
    return {"status": "ready", "models": checked}


@app.get("/readiness")
def readiness() -> dict[str, Any]:
    try:
        return _check_readiness()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "error": str(e)},
        ) from e


@app.get("/models")
def models_endpoint() -> dict[str, list[str]]:
    return {"models": list_models()}


@app.get("/mock/profile")
def mock_profile(seed: int = Query(..., description="Seed генератора")) -> dict[str, Any]:
    return MockSeriesGenerator.config_from_seed(seed).__dict__


async def get_anomaly_stream_engine(seed: int):
    """Mock-стрим: ряд из ``MockSeriesGenerator(seed)``, детектор — spikes по умолчанию."""
    gen = MockSeriesGenerator(seed)
    kinds = list_models()
    model = kinds[seed % len(kinds)]
    est_params: dict[str, Any] = {}
    if model == "bounds":
        est_params["bounds"] = {
            str(i): (-50.0 - i, 50.0 + i) for i in range(gen.n_features)
        }
    det = UnifiedAnomalyDetector(model, estimator_params=est_params)

    buf: list[np.ndarray] = []
    fit_done = False

    yield {"phase": "started", "seed": seed, "generator": gen.describe(), "model": model}

    while True:
        buf.append(gen.next_row())
        if len(buf) > _STREAM_MAX_BUFFER:
            buf.pop(0)

        if len(buf) < _STREAM_WARMUP:
            if _STREAM_SLEEP_MS > 0:
                await asyncio.sleep(_STREAM_SLEEP_MS / 1000.0)
            continue

        x = np.asarray(buf, dtype=float)
        if not fit_done:
            det.fit(x)
            fit_done = True
            yield {"phase": "ready", "seed": seed, "n_rows": len(buf)}

        proba = det.predict_proba(x)
        pred = det.predict(x)
        last = len(buf) - 1

        yield {
            "phase": "sample",
            "seed": seed,
            "t": gen.step - 1,
            "n_rows": len(buf),
            "x": buf[last].tolist(),
            "proba": proba[last].tolist(),
            "predict": pred[last].tolist(),
        }

        if _STREAM_SLEEP_MS > 0:
            await asyncio.sleep(_STREAM_SLEEP_MS / 1000.0)


@app.get("/mock/stream")
async def mock_stream(
    seed: int = Query(..., description="Единственный параметр: воспроизводимый seed стрима"),
) -> StreamingResponse:
    async def events():
        async for payload in get_anomaly_stream_engine(seed):
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/mock/ws/stream")
async def mock_stream_ws(
    websocket: WebSocket,
    seed: int = Query(..., description="Единственный параметр: воспроизводимый seed стрима"),
):
    await websocket.accept()
    try:
        async for payload in get_anomaly_stream_engine(seed):
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass