# UI — контуры датчиков

React + Vite + TypeScript. Админка registry и просмотр mock-стрима.

## Запуск

В одном терминале — API:

```bash
cd ..
uv run uvicorn api.app:app --host 127.0.0.1 --port 8765
```

В другом — UI (прокси на API):

```bash
npm install
npm run dev
```

Открыть http://127.0.0.1:5173

## Сборка

```bash
npm run build
```

Артефакты в `dist/`.
