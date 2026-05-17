const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      const j = JSON.parse(body) as { detail?: unknown };
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep text */
    }
    throw new Error(detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  readiness: () => request<{ status: string; models: string[] }>("/readiness"),
  modelKinds: () => request<{ models: string[] }>("/models"),
  mockProfile: (seed: number) =>
    request<Record<string, unknown>>(`/mock/profile?seed=${seed}`),

  listRegistries: () =>
    request<{ registries: RegistryInfo[] }>("/registry"),
  createRegistry: (body: CreateRegistryPayload) =>
    request<RegistryInfo>("/registry", { method: "POST", body: JSON.stringify(body) }),
  getRegistry: (id: string) => request<RegistryInfo>(`/registry/${id}`),
  deleteRegistry: (id: string) =>
    request<{ status: string }>(`/registry/${id}`, { method: "DELETE" }),
  listRegistryModels: (id: string) =>
    request<{ models: ModelRow[]; current: CurrentModel | null }>(`/registry/${id}/models`),
  addModel: (id: string, body: AddModelPayload) =>
    request<ModelRow>(`/registry/${id}/models`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteModel: (id: string, name: string, kind?: string) => {
    const q = kind ? `?kind=${encodeURIComponent(kind)}` : "";
    return request<{ status: string }>(`/registry/${id}/models/${encodeURIComponent(name)}${q}`, {
      method: "DELETE",
    });
  },
  setCurrentModel: (id: string, name: string, kind: string) =>
    request<{ kind: string; name: string }>(`/registry/${id}/models/current`, {
      method: "PUT",
      body: JSON.stringify({ name, kind }),
    }),
};

export type Sensor = { name: string; low: number; high: number };

export type RegistryInfo = {
  registry_id: string;
  title: string;
  sensors: Sensor[];
  models: { name: string; kind: string }[];
  current: { kind: string; name: string } | null;
  created_at: string;
};

export type ModelRow = {
  name: string;
  kind: string;
  hyperparameters: Record<string, unknown>;
  model_id: string;
  trained_at: string | null;
  created_at: string;
  is_current: boolean;
};

export type CurrentModel = {
  name: string;
  kind: string;
  hyperparameters: Record<string, unknown>;
  model_id: string;
  trained_at: string | null;
};

export type CreateRegistryPayload = {
  title: string;
  registry_id?: string;
  sensors: Sensor[];
};

export type AddModelPayload = {
  name: string;
  kind: string;
  hyperparameters?: Record<string, unknown>;
  set_current?: boolean;
};

export function streamUrl(seed: number): string {
  return `${API_BASE}/mock/stream?seed=${seed}`;
}

export function wsStreamUrl(seed: number): string {
  const base = API_BASE || window.location.origin.replace(":5173", ":8765");
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = base.replace(/^https?:\/\//, "").replace(/^wss?:\/\//, "");
  return `${proto}//${host}/mock/ws/stream?seed=${seed}`;
}

export type StreamEvent = {
  phase: string;
  seed?: number;
  model?: string;
  generator?: Record<string, unknown>;
  t?: number;
  n_rows?: number;
  x?: number[];
  proba?: number[];
  predict?: number[];
};
