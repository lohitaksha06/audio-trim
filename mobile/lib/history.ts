import AsyncStorage from "@react-native-async-storage/async-storage";

const HISTORY_KEY = "audelle.history";
const MAX_ENTRIES = 50;

export interface HistoryEntry {
  id: string;
  prompt: string;
  audioName: string;
  intent: string;
  createdAt: number;
  summary: string;
}

function nowId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function readAll(): Promise<HistoryEntry[]> {
  try {
    const raw = await AsyncStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const entries = await readAll();
  return entries.sort((a, b) => b.createdAt - a.createdAt);
}

export async function appendHistory(entry: Omit<HistoryEntry, "id" | "createdAt">): Promise<HistoryEntry[]> {
  const entries = await readAll();
  const full: HistoryEntry = { ...entry, id: nowId(), createdAt: Date.now() };
  entries.push(full);
  const trimmed = entries.slice(-MAX_ENTRIES);
  try {
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch {
    // ignore storage failures — history is best-effort
  }
  return trimmed.sort((a, b) => b.createdAt - a.createdAt);
}

export async function clearHistory(): Promise<void> {
  try {
    await AsyncStorage.removeItem(HISTORY_KEY);
  } catch {
    // ignore
  }
}