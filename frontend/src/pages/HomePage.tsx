import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";

export default function HomePage() {
  const [health, setHealth] = useState<"ok" | "err" | "loading">("loading");
  const [ready, setReady] = useState<{
    status: string;
    models?: string[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await api.health();
        if (!cancelled) setHealth(h.status === "ok" ? "ok" : "err");
        const r = await api.readiness();
        if (!cancelled) setReady(r);
      } catch (e) {
        if (!cancelled) {
          setHealth("err");
          setError(e instanceof Error ? e.message : "Ошибка API");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <header className="page-header">
        <h2>Обзор системы</h2>
        <p>Статус API, детекторы и быстрые ссылки для Grafana / mock-стрима.</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="grid-2">
        <div className="card">
          <h3>Health (liveness)</h3>
          {health === "loading" && <span className="badge neutral">проверка…</span>}
          {health === "ok" && <span className="badge ok">ok</span>}
          {health === "err" && <span className="badge err">недоступен</span>}
        </div>
        <div className="card">
          <h3>Readiness</h3>
          {!ready && health !== "err" && <span className="badge neutral">…</span>}
          {ready?.status === "ready" && (
            <>
              <span className="badge ok">ready</span>
              <p style={{ margin: "0.75rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
                Модели: {ready.models?.join(", ")}
              </p>
            </>
          )}
          {ready && ready.status !== "ready" && (
            <span className="badge warn">{ready.status}</span>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h3>Быстрый старт</h3>
        <ul style={{ margin: 0, paddingLeft: "1.2rem", color: "var(--muted)", lineHeight: 1.7 }}>
          <li>
            <Link to="/registries">Создать registry</Link> — датчики с границами
          </li>
          <li>
            <Link to="/stream">Mock-стрим</Link> — только параметр <code>seed</code>
          </li>
          <li>
            Swagger API:{" "}
            <a href="http://127.0.0.1:8765/docs" target="_blank" rel="noreferrer">
              /docs
            </a>
          </li>
        </ul>
      </div>
    </>
  );
}
