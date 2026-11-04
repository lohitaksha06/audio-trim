/**
 * Small persistence layer for the on-device studio.
 *
 * Audio blobs live in IndexedDB (localStorage cannot hold them), while tiny
 * JSON records live in localStorage. Every function is a no-op when the APIs
 * are unavailable (SSR, private mode) so the UI can keep working without
 * persistence instead of breaking.
 */

const DB_NAME = "audelle-project";
const DB_VERSION = 1;
const FILE_STORE = "files";
const RESULT_STORE = "results";
const CURRENT_FILE_KEY = "current-source";
const CURRENT_OUTPUT_KEY = "current-output";
const META_KEY = "audelle:project:meta";

export interface StoredFile {
  blob: Blob;
  name: string;
  type: string;
  lastModified: number;
}

export interface StoredResult {
  blob: Blob;
  name: string;
}

function canUseIdb(): boolean {
  return typeof window !== "undefined" && typeof indexedDB !== "undefined";
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(FILE_STORE)) db.createObjectStore(FILE_STORE);
      if (!db.objectStoreNames.contains(RESULT_STORE)) db.createObjectStore(RESULT_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
  });
}

async function idbPut(store: string, key: string, value: unknown): Promise<void> {
  if (!canUseIdb()) return;
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(store, "readwrite");
      tx.objectStore(store).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("IndexedDB write failed"));
    });
    db.close();
  } catch {
    /* Persistence is a convenience; never break an edit for it. */
  }
}

async function idbGet<T>(store: string, key: string): Promise<T | null> {
  if (!canUseIdb()) return null;
  try {
    const db = await openDb();
    const value = await new Promise<T | null>((resolve, reject) => {
      const tx = db.transaction(store, "readonly");
      const request = tx.objectStore(store).get(key);
      request.onsuccess = () => resolve((request.result as T | undefined) ?? null);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB read failed"));
    });
    db.close();
    return value;
  } catch {
    return null;
  }
}

async function idbDelete(store: string, key: string): Promise<void> {
  if (!canUseIdb()) return;
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(store, "readwrite");
      tx.objectStore(store).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("IndexedDB delete failed"));
    });
    db.close();
  } catch {
    /* ignore */
  }
}

export async function saveProjectFile(file: File): Promise<void> {
  await idbPut(FILE_STORE, CURRENT_FILE_KEY, {
    blob: file,
    name: file.name,
    type: file.type,
    lastModified: file.lastModified,
  } satisfies StoredFile);
  saveProjectMeta({ name: file.name, type: file.type, lastModified: file.lastModified });
}

export async function loadProjectFile(): Promise<File | null> {
  const stored = await idbGet<StoredFile>(FILE_STORE, CURRENT_FILE_KEY);
  if (!stored?.blob) return null;
  return new File([stored.blob], stored.name, {
    type: stored.type || "audio/*",
    lastModified: stored.lastModified || Date.now(),
  });
}

export async function clearProjectFile(): Promise<void> {
  await idbDelete(FILE_STORE, CURRENT_FILE_KEY);
  await idbDelete(RESULT_STORE, CURRENT_OUTPUT_KEY);
  removeProjectMeta();
}

export async function saveProjectOutput(blob: Blob, name: string): Promise<void> {
  await idbPut(RESULT_STORE, CURRENT_OUTPUT_KEY, { blob, name } satisfies StoredResult);
}

export async function loadProjectOutput(): Promise<StoredResult | null> {
  return idbGet<StoredResult>(RESULT_STORE, CURRENT_OUTPUT_KEY);
}

export async function clearProjectOutput(): Promise<void> {
  await idbDelete(RESULT_STORE, CURRENT_OUTPUT_KEY);
}

export interface ProjectMeta {
  name: string;
  type: string;
  lastModified: number;
  /** Server upload response, kept so a refresh does not upload twice. */
  serverUpload?: unknown;
  /** Last prompt/result summary, for the empty state after a refresh. */
  lastAction?: string;
}

export function saveProjectMeta(meta: ProjectMeta): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(META_KEY, JSON.stringify(meta));
  } catch {
    /* ignore */
  }
}

export function loadProjectMeta(): ProjectMeta | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(META_KEY);
    return raw ? (JSON.parse(raw) as ProjectMeta) : null;
  } catch {
    return null;
  }
}

export function removeProjectMeta(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(META_KEY);
  } catch {
    /* ignore */
  }
}

export function saveJson(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

export function loadJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function removeJson(key: string): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

/** Trigger a browser download from a Blob, with a real DOM click. */
export function downloadBlob(blob: Blob, filename: string): void {
  if (typeof document === "undefined") return;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Keep the URL alive for large WAVs and for Safari's deferred read.
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export function safeFilename(base: string, extension = "wav"): string {
  const cleaned = base
    .replace(/\.(mp3|wav|flac|m4a|aac|ogg)(?=[^.]*$)/i, "")
    .replace(/\.[a-z0-9]{2,5}$/i, "")
    .replace(/[^a-z0-9._-]+/gi, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return `${cleaned || "audelle"}.${extension}`;
}
