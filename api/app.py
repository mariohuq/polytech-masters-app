from __future__ import annotations

import asyncio
import datetime
import json
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.generator import MockSeriesGenerator
from api.registry_routes import router as registry_router
from models import DetectorKind, UnifiedAnomalyDetector, list_models

app = FastAPI(title="Polytech masters — mock stream API", version="0.1.0")
app.include_router(registry_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_config(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Невалидный JSON в config: {e}") from e


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



async def get_anomaly_stream_engine(
    model: str,
    n_features: int,
    seed: int | None,
    sleep_ms: float,
    max_buffer: int,
    warmup: int,
    max_events: int | None,
    config: str | None,
):
    """
    Shared core logic for both SSE and WebSocket endpoints.
    Yields dict payloads sequentially.
    """
    allowed = set(list_models()) # Assuming list_models() exists in your scope
    if model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестная model={model!r}. Доступно: {sorted(allowed)}",
        )
        
    extra = _parse_config(config) # Assuming _parse_config() exists in your scope
    det = UnifiedAnomalyDetector(model=model, estimator_params=extra)
    
    gen = MockSeriesGenerator(n_features=n_features, seed=seed)
    buf: list[np.ndarray] = []
    events_out = 0
    fit_done = False

    while max_events is None or events_out < max_events:
        buf.append(gen.next_row())
        if len(buf) > max_buffer:
            buf.pop(0)

        if len(buf) < warmup:
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)
            continue

        x = np.asarray(buf, dtype=float)
        if not fit_done:
            det.fit(x)
            fit_done = True

        proba = det.predict_proba(x)
        pred = det.predict(x)
        last = len(buf) - 1
        
        # Yield pure dictionary payload
        yield {
            "t": datetime.datetime.isoformat(datetime.datetime.now()),
            "n_rows": len(buf),
            "x": buf[last].tolist(),
            "proba": proba[last].tolist(),
            "predict": pred[last].tolist(),
        }
        
        events_out += 1
        if sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000.0)


@app.get("/mock/stream")
async def mock_stream(
    model: str = Query("spikes", description="bounds | spikes | glitch — см. GET /models"),
    n_features: int = Query(2, ge=1, le=32),
    seed: int | None = Query(None),
    sleep_ms: float = Query(50.0, ge=0.0, description="Пауза между событиями (мс); 0 — без задержки"),
    max_buffer: int = Query(512, ge=16, le=10_000),
    warmup: int = Query(64, ge=1, le=20_000, description="Точки перед первым fit"),
    max_events: int | None = Query(None, ge=1, description="Ограничение числа SSE-событий"),
    config: str | None = Query(None, description="JSON с гиперпараметрами"),
) -> StreamingResponse:

    async def events():
        # Consume the shared engine and format for SSE
        async for payload in get_anomaly_stream_engine(
            model, n_features, seed, sleep_ms, max_buffer, warmup, max_events, config
        ):
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.websocket("/mock/ws/stream")
async def mock_stream_ws(
    websocket: WebSocket,
    model: str = Query("robust_rolling"),
    n_features: int = Query(2, ge=1, le=32),
    seed: int | None = Query(None),
    sleep_ms: float = Query(50.0, ge=0.0),
    max_buffer: int = Query(512, ge=16, le=10_000),
    warmup: int = Query(64, ge=1, le=20_000),
    max_events: int | None = Query(None, ge=1),
    config: str | None = Query(None),
):
    await websocket.accept()
    try:
        # Consume the exact same shared engine and push over WebSocket frame
        async for payload in get_anomaly_stream_engine(
            model, n_features, seed, sleep_ms, max_buffer, warmup, max_events, config
        ):
            await websocket.send_json(payload)
            
    except WebSocketDisconnect:
        # Handled gracefully when Grafana panel closes or dashboard reloads
        pass