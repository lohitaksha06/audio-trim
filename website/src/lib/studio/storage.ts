/**
 * Studio project persistence. The project document is JSON and tiny; audio
 * assets are stored as blobs in IndexedDB. Both are keyed separately so a
 * failed asset write never corrupts the edit state.
 */

import type { StudioProject } from "./types";

const DB_NAME = "audelle-studio";
const DB_VERSION = 1;
const PROJECT_STORE = "project";
const ASSET_STORE = "assets";
const PROJECT_KEY = "current";

interface StoredAsset {
  blob: Blob;
  name: string;
  type: string;
}

function canUseIdb(): boolean {
  return typeof window !== "undefined" && typeof indexedDB !== "undefined";
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(PROJECT_STORE)) db.createObjectStore(PROJECT_STORE);
      if (!db.objectStoreNames.contains(ASSET_STORE)) db.createObjectStore(ASSET_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
  });
}

async function put(store: string, key: string, value: unknown): Promise<void> {
  if (!canUseIdb()) return;
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(store, "readwrite");
      tx.objectStore(store).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    /* persistence is optional */
  }
}

async function get<T>(store: string, key: string): Promise<T | null> {
  if (!canUseIdb()) return null;
  try {
    const db = await openDb();
    const value = await new Promise<T | null>((resolve, reject) => {
      const tx = db.transaction(store, "readonly");
      const req = tx.objectStore(store).get(key);
      req.onsuccess = () => resolve((req.result as T | undefined) ?? null);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return value;
  } catch {
    return null;
  }
}

async function remove(store: string, key: string): Promise<void> {
  if (!canUseIdb()) return;
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(store, "readwrite");
      tx.objectStore(store).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    /* ignore */
  }
}

export async function saveStudioProject(project: StudioProject): Promise<void> {
  await put(PROJECT_STORE, PROJECT_KEY, project);
}

export async function loadStudioProject(): Promise<StudioProject | null> {
  return get<StudioProject>(PROJECT_STORE, PROJECT_KEY);
}

export async function saveStudioAsset(id: string, file: File): Promise<void> {
  await put(ASSET_STORE, id, { blob: file, name: file.name, type: file.type } satisfies StoredAsset);
}

export async function loadStudioAsset(id: string): Promise<File | null> {
  const stored = await get<StoredAsset>(ASSET_STORE, id);
  if (!stored?.blob) return null;
  return new File([stored.blob], stored.name, {
    type: stored.type || "audio/wav",
    lastModified: Date.now(),
  });
}

export async function deleteStudioAsset(id: string): Promise<void> {
  await remove(ASSET_STORE, id);
}

export async function clearStudioProject(): Promise<void> {
  await remove(PROJECT_STORE, PROJECT_KEY);
}
