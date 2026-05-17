import { useEffect, useRef, useState } from "react";
import { api, streamUrl, type StreamEvent } from "../api/client";

type Sample = StreamEvent & { phase: "sample" };

export default function MockStreamPage() {
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [meta, setMeta] = useState<StreamEvent | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  async function loadProfile() {
    try {
      const p = await api.mockProfile(seed);
      setProfile(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка profile");
    }
  }

  function stop() {
    esRef.current?.close();
    esRef.current = null;
    setRunning(false);
  }

  function start() {
    stop();
    setError(null);
    setSamples([]);
    setMeta(null);
    setRunning(true);

    const es = new EventSource(streamUrl(seed));
    esRef.current = es;

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as StreamEvent;
        if (data.phase === "started") {
          setMeta(data);
          setProfile((data.generator as Record<string, unknown>) ?? null);
        } else if (data.phase === "sample") {
          setSamples((prev) => {
            const next = [...prev, data as Sample];
            return next.slice(-40);
          });
        }
      } catch {
        /* ignore parse errors */
      }
    };

    es.onerror = () => {
      setError("Соединение со стримом прервано");
      stop();
    };
  }

  const last = samples[samples.length - 1];
  const values = samples.map((s) => s.x?.[0] ?? 0);
  const maxVal = Math.max(...values, 1e-6);

  return (
    <>
      <header className="page-header">
        <h2>Mock-стрим</h2>
        <p>
          Воспроизводимый поток по одному параметру <strong>seed</strong> (SSE).
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        <div className="flex-between">
          <div className="field" style={{ flex: 1, maxWidth: 200, marginBottom: 0 }}>
            <label>Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              disabled={running}
            />
          </div>
          <div className="row-actions">
            <button type="button" className="btn btn-ghost" onClick={loadProfile} disabled={running}>
              Профиль
            </button>
            {!running ? (
              <button type="button" className="btn btn-primary" onClick={start}>
                Старт
              </button>
            ) : (
              <button type="button" className="btn btn-danger" onClick={stop}>
                Стоп
              </button>
            )}
          </div>
        </div>
        <p className="mono" style={{ margin: "1rem 0 0", fontSize: "0.8rem", color: "var(--muted)" }}>
          {streamUrl(seed)}
        </p>
      </div>

      <div className="stream-grid">
        <div className="card">
          <h3>Конфигурация</h3>
          {meta && (
            <p style={{ margin: "0 0 0.5rem" }}>
              Модель: <span className="badge neutral">{meta.model}</span>
              {running && <span className="badge ok" style={{ marginLeft: 8 }}>live</span>}
            </p>
          )}
          {profile ? (
            <pre
              className="mono"
              style={{
                margin: 0,
                fontSize: "0.78rem",
                background: "var(--surface-2)",
                padding: "0.75rem",
                borderRadius: 8,
                overflow: "auto",
              }}
            >
              {JSON.stringify(profile, null, 2)}
            </pre>
          ) : (
            <p className="empty" style={{ padding: "1rem" }}>
              Нажмите «Профиль» или «Старт»
            </p>
          )}
        </div>

        <div className="card">
          <h3>Канал 0</h3>
          {values.length === 0 && <p className="empty" style={{ padding: "1rem" }}>Нет данных</p>}
          {values.length > 0 && (
            <>
              <div className="sparkline">
                {values.map((v, i) => {
                  const alert = (samples[i]?.predict?.[0] ?? 0) > 0;
                  return (
                    <span
                      key={i}
                      className={alert ? "alert" : ""}
                      style={{ height: `${Math.min(100, (Math.abs(v) / maxVal) * 100)}%` }}
                      title={`t=${samples[i].t} x=${v.toFixed(3)}`}
                    />
                  );
                })}
              </div>
              {last && (
                <p style={{ margin: "0.75rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
                  t={last.t} · x=[{last.x?.map((n) => n.toFixed(3)).join(", ")}] · proba=[
                  {last.proba?.map((n) => n.toFixed(2)).join(", ")}]
                </p>
              )}
            </>
          )}
        </div>
      </div>

      {samples.length > 0 && (
        <div className="card">
          <h3>Последние события</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>t</th>
                  <th>x</th>
                  <th>proba</th>
                  <th>predict</th>
                </tr>
              </thead>
              <tbody>
                {[...samples].reverse().slice(0, 12).map((s) => (
                  <tr key={s.t}>
                    <td>{s.t}</td>
                    <td className="mono">{s.x?.map((n) => n.toFixed(3)).join(", ")}</td>
                    <td className="mono">{s.proba?.map((n) => n.toFixed(2)).join(", ")}</td>
                    <td>
                      {(s.predict ?? []).some((p) => p > 0) ? (
                        <span className="badge err">аномалия</span>
                      ) : (
                        <span className="badge ok">норма</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
