import AsyncStorage from "@react-native-async-storage/async-storage";

const QUEUE_KEY = "audelle.queue";

export type QueuedStatus = "queued" | "in_progress" | "done" | "failed";

export interface QueuedAsset {
  uri: string;
  name: string;
  mimeType?: string;
}

export interface QueueTask {
  id: string;
  prompt: string;
  audioName: string;
  asset: QueuedAsset;
  audioPath?: string;
  status: QueuedStatus;
  error?: string;
  resultKey?: string;
  createdAt: number;
}

async function readAll(): Promise<QueueTask[]> {
  try {
    const raw = await AsyncStorage.getItem(QUEUE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function writeAll(tasks: QueueTask[]): Promise<void> {
  try {
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(tasks));
  } catch {
    // ignore — queue is best-effort
  }
}

export async function getQueue(): Promise<QueueTask[]> {
  return readAll();
}

export async function getPendingCount(): Promise<number> {
  const tasks = await readAll();
  return tasks.filter((t) => t.status === "queued" || t.status === "in_progress").length;
}

export async function enqueue(
  input: Pick<QueueTask, "prompt" | "audioName" | "asset"> & { audioPath?: string },
): Promise<QueueTask> {
  const task: QueueTask = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    prompt: input.prompt,
    audioName: input.audioName,
    asset: input.asset,
    audioPath: input.audioPath,
    status: "queued",
    createdAt: Date.now(),
  };
  const tasks = await readAll();
  tasks.push(task);
  await writeAll(tasks);
  return task;
}

export async function updateTask(id: string, patch: Partial<QueueTask>): Promise<QueueTask[]> {
  const tasks = await readAll();
  const next = tasks.map((t) => (t.id === id ? { ...t, ...patch } : t));
  await writeAll(next);
  return next;
}

export async function removeTask(id: string): Promise<QueueTask[]> {
  const tasks = await readAll();
  const next = tasks.filter((t) => t.id !== id);
  await writeAll(next);
  return next;
}

export async function clearFinished(): Promise<QueueTask[]> {
  const tasks = await readAll();
  const next = tasks.filter((t) => t.status !== "done");
  await writeAll(next);
  return next;
}