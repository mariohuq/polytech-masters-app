import { FormEvent, useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type ModelRow, type RegistryInfo } from "../api/client";

export default function RegistryDetailPage() {
  const { registryId } = useParams<{ registryId: string }>();
  const [info, setInfo] = useState<RegistryInfo | null>(null);
  const [models, setModels] = useState<ModelRow[]>([]);
  const [kinds, setKinds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [name, setName] = useState("");
  const [kind, setKind] = useState("spikes");
  const [setCurrentOnAdd, setSetCurrentOnAdd] = useState(true);

  const load = useCallback(async () => {
    if (!registryId) return;
    setLoading(true);
    setError(null);
    try {
      const [reg, mdl, kindsRes] = await Promise.all([
        api.getRegistry(registryId),
        api.listRegistryModels(registryId),
        api.modelKinds(),
      ]);
      setInfo(reg);
      setModels(mdl.models);
      setKinds(kindsRes.models);
      if (kindsRes.models.length) setKind(kindsRes.models[0]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка");
    } finally {
      setLoading(false);
    }
  }, [registryId]);

  useEffect(() => {
    load();
  }, [load]);

  async function addModel(e: FormEvent) {
    e.preventDefault();
    if (!registryId) return;
    try {
      await api.addModel(registryId, {
        name,
        kind,
        set_current: setCurrentOnAdd,
      });
      setName("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка");
    }
  }

  async function setCurrent(m: ModelRow) {
    if (!registryId) return;
    try {
      await api.setCurrentModel(registryId, m.name, m.kind);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка");
    }
  }

  async function removeModel(m: ModelRow) {
    if (!registryId || !confirm(`Удалить модель «${m.name}»?`)) return;
    try {
      await api.deleteModel(registryId, m.name, m.kind);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка");
    }
  }

  if (!registryId) return null;

  const origin = window.location.origin.replace(":5173", ":8765");
  const streamHint = `${origin}/mock/stream?seed=42`;

  return (
    <>
      <header className="page-header">
        <p style={{ margin: 0 }}>
          <Link to="/registries">← Реестры</Link>
        </p>
        <h2>{info?.title || registryId}</h2>
        <p className="mono" style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          {registryId}
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {loading && <p className="empty">Загрузка…</p>}

      {!loading && info && (
        <>
          <div className="grid-2">
            <div className="card">
              <h3>Датчики</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Имя</th>
                      <th>Нижняя</th>
                      <th>Верхняя</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.sensors.map((s) => (
                      <tr key={s.name}>
                        <td>{s.name}</td>
                        <td>{s.low}</td>
                        <td>{s.high}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <h3>Адреса для Grafana</h3>
              <p style={{ fontSize: "0.85rem", color: "var(--muted)", marginTop: 0 }}>
                Mock-поток данных (SSE), seed задаёт весь ряд:
              </p>
              <div className="url-box">{streamHint}</div>
              <p style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
                WebSocket: <span className="mono">/mock/ws/stream?seed=…</span>
              </p>
            </div>
          </div>

          <form className="card" onSubmit={addModel}>
            <h3>Новая модель</h3>
            <div className="grid-2">
              <div className="field">
                <label>Имя (уникальное)</label>
                <input value={name} onChange={(e) => setName(e.target.value)} required />
              </div>
              <div className="field">
                <label>Тип</label>
                <select value={kind} onChange={(e) => setKind(e.target.value)}>
                  {kinds.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", fontSize: "0.9rem" }}>
              <input
                type="checkbox"
                checked={setCurrentOnAdd}
                onChange={(e) => setSetCurrentOnAdd(e.target.checked)}
              />
              Сделать текущей
            </label>
            <button type="submit" className="btn btn-primary" style={{ marginTop: "0.75rem" }}>
              Добавить
            </button>
          </form>

          <div className="card">
            <h3>Модели в registry</h3>
            {models.length === 0 && <p className="empty">Пока нет моделей.</p>}
            {models.length > 0 && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Имя</th>
                      <th>Тип</th>
                      <th>Обучена</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.model_id}>
                        <td>
                          {m.name}
                          {m.is_current && (
                            <span className="badge ok" style={{ marginLeft: 8 }}>
                              текущая
                            </span>
                          )}
                        </td>
                        <td>{m.kind}</td>
                        <td className="mono" style={{ fontSize: "0.78rem" }}>
                          {m.trained_at ? "да" : "—"}
                        </td>
                        <td>
                          <div className="row-actions">
                            {!m.is_current && (
                              <button
                                type="button"
                                className="btn btn-ghost btn-sm"
                                onClick={() => setCurrent(m)}
                              >
                                Сделать текущей
                              </button>
                            )}
                            <button
                              type="button"
                              className="btn btn-danger btn-sm"
                              onClick={() => removeModel(m)}
                            >
                              Удалить
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
}
