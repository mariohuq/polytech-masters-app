
# Init project

```
# Install uv if you haven't (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup and install deps
uv sync
```

# Run API

```
uv run uvicorn api.app:app --host 127.0.0.1 --port 8765
```

See API docs <http://localhost:8765/docs>

# Visualize mock

Start graphana on :3000

```shell
(cd infra; docker compose up -d)
```

Use this url in datasource (WebSocket API, Settings - WebSocket - Host):

```
ws://host.docker.internal:8765/mock/ws/stream
```
