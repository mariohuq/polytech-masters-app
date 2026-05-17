import { FormEvent, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  type CreateRegistryPayload,
  type RegistryInfo,
  type Sensor,
} from "../api/client";

const emptySensor = (): Sensor => ({ name: "", low: 0, high: 100 });

export default function RegistriesPage() {
  const [list, setList] = useState<RegistryInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  const [title, setTitle] = useState("");
  const [registryId, setRegistryId] = useState("");
  const [sensors, setSensors] = useState<Sensor[]>([
    { name: "T1", low: 0, high: 100 },
    { name: "P1", low: -1, high: 10 },
  ]);
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listRegistries();
      setList(data.registries);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const body: CreateRegistryPayload = {
        title,
        sensors: sensors.filter((s) => s.name.trim()),
      };
      if (registryId.trim()) body.registry_id = registryId.trim();
      await api.createRegistry(body);
      setShowForm(false);
      setTitle("");
      setRegistryId("");
      setSensors([emptySensor(), emptySensor()]);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка создания");
    } finally {
      setSubmitting(false);
    }
  }

  async function onDelete(id: string) {
    if (!confirm(`Пометить registry «${id}» удалённым?`)) return;
    try {
      await api.deleteRegistry(id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка удаления");
    }
  }

  return (
    <>
      <header className="page-header flex-between">
        <div>
          <h2>Реестры датчиков</h2>
          <p>Создание registry: имена датчиков и границы (по схеме «Контуры датчиков»).</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setShowForm((v) => !v)}>
          {showForm ? "Отмена" : "+ Новый registry"}
        </button>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {showForm && (
        <form className="card" onSubmit={onSubmit}>
          <h3>Создание registry</h3>
          <div className="field">
            <label>Название источника</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Plant A" />
          </div>
          <div className="field">
            <label>ID (опционально)</label>
            <input
              value={registryId}
              onChange={(e) => setRegistryId(e.target.value)}
              placeholder="plant-a"
            />
          </div>
          <h4 style={{ margin: "0 0 0.5rem", fontSize: "0.9rem" }}>Датчики</h4>
          {sensors.map((s, i) => (
            <div key={i} className="sensor-row">
              <input
                placeholder="имя"
                value={s.name}
                onChange={(e) => {
                  const next = [...sensors];
                  next[i] = { ...s, name: e.target.value };
                  setSensors(next);
                }}
              />
              <input
                type="number"
                step="any"
                value={s.low}
                onChange={(e) => {
                  const next = [...sensors];
                  next[i] = { ...s, low: Number(e.target.value) };
                  setSensors(next);
                }}
              />
              <input
                type="number"
                step="any"
                value={s.high}
                onChange={(e) => {
                  const next = [...sensors];
                  next[i] = { ...s, high: Number(e.target.value) };
                  setSensors(next);
                }}
              />
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setSensors(sensors.filter((_, j) => j !== i))}
                disabled={sensors.length <= 1}
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            style={{ marginBottom: "1rem" }}
            onClick={() => setSensors([...sensors, emptySensor()])}
          >
            + датчик
          </button>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? "Создаём…" : "Создать"}
          </button>
        </form>
      )}

      <div className="card">
        <h3>Список</h3>
        {loading && <p className="empty">Загрузка…</p>}
        {!loading && list.length === 0 && (
          <p className="empty">Нет registry — создайте первый.</p>
        )}
        {!loading && list.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Название</th>
                  <th>Датчиков</th>
                  <th>Текущая модель</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {list.map((r) => (
                  <tr key={r.registry_id}>
                    <td className="mono">{r.registry_id}</td>
                    <td>{r.title || "—"}</td>
                    <td>{r.sensors?.length ?? 0}</td>
                    <td>
                      {r.current ? (
                        <span className="badge neutral">
                          {r.current.kind} / {r.current.name}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      <div className="row-actions">
                        <Link to={`/registries/${r.registry_id}`} className="btn btn-ghost btn-sm">
                          Открыть
                        </Link>
                        <button
                          type="button"
                          className="btn btn-danger btn-sm"
                          onClick={() => onDelete(r.registry_id)}
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
  );
}
