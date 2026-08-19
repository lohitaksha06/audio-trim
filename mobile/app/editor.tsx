import { useCallback, useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import * as DocumentPicker from "expo-document-picker";
import NetInfo from "@react-native-community/netinfo";
import { useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import {
  AnalysisResult,
  ProcessResponse,
  GenreInfo,
  uploadFile,
  understandAudio,
  submitJob,
  getJob,
  downloadUrl,
} from "../lib/api";
import { enqueue, getQueue, getPendingCount, updateTask, QueueTask } from "../lib/queue";
import { appendHistory, getHistory, clearHistory, HistoryEntry } from "../lib/history";
import { notifyJobDone } from "../lib/notifications";
import { downloadAndShare } from "../lib/export";

const MAX_POLLS = 160;

const FEATURES = [
  { category: "Source Separation", items: ["Isolate vocals", "Extract drums", "Remove bass", "Separate all stems"] },
  { category: "Smart Editing", items: ["Trim section", "Remove section", "Cut intro", "Highlight reel"] },
  { category: "Mood & Style", items: ["Make darker", "More energetic", "Add reverb", "Add fade"] },
  { category: "Format Conversion", items: ["WAV to MP3", "WAV to FLAC", "MP3 to WAV", "Any to OGG"] },
];

type ChatMsg = { role: "user" | "ai"; text: string };

export default function Editor() {
  const [file, setFile] = useState<DocumentPicker.DocumentPickerAsset | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<"idle" | "uploading" | "analyzed" | "processing" | "queued" | "completed">("idle");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [genre, setGenre] = useState<GenreInfo | null>(null);
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [queueCount, setQueueCount] = useState(0);
  const [resultKey, setResultKey] = useState<string | null>(null);
  const [expandedCat, setExpandedCat] = useState<string | null>(null);
  const [playerPlay, setPlayerPlay] = useState(false);

  const resultUrl = resultKey ? downloadUrl(resultKey) : null;
  const player = useAudioPlayer(resultUrl);
  const status = useAudioPlayerStatus(player);

  const refreshQueueCount = useCallback(async () => {
    setQueueCount(await getPendingCount());
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      const [h, q] = await Promise.all([getHistory(), getQueue()]);
      if (cancelled) return;
      setHistory(h);
      setQueueCount(q.filter((t) => t.status === "queued" || t.status === "in_progress").length);
    })();
    const unsub = NetInfo.addEventListener((s) => {
      if (s.isConnected && s.isInternetReachable !== false) flushQueue();
    });
    return () => {
      cancelled = true;
      unsub();
    };
  }, []);

  useEffect(() => {
    if (resultKey && player && resultUrl) {
      player.replace(resultUrl);
    }
  }, [resultKey, resultUrl, player]);

  useEffect(() => {
    if (status.playing) {
      setPlayerPlay(true);
    } else if (status.didJustFinish) {
      setPlayerPlay(false);
    }
  }, [status.playing, status.didJustFinish]);

  const addUser = (text: string) => setChat((c) => [...c, { role: "user", text }]);
  const addAi = (text: string) => setChat((c) => [...c, { role: "ai", text }]);

  const pickFile = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ["audio/*", "video/*"],
      copyToCacheDirectory: true,
    });

    if (result.canceled || !result.assets[0]) return;
    const asset = result.assets[0];
    setFile(asset);
    setState("uploading");
    setChat([]);
    setResultKey(null);

    try {
      const data = await uploadFile({ uri: asset.uri, name: asset.name, mimeType: asset.mimeType });
      setAudioPath(data.audio_path);
      setAnalysis(data.analysis);
      setState("analyzed");
      understandAudio(data.audio_path)
        .then((u) => setGenre(u.genre ?? null))
        .catch(() => setGenre(null));
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to upload file";
      Alert.alert("Upload failed", message);
      setState("idle");
    }
  };

  const pollJob = async (jobId: string, promptText: string) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      let job;
      try {
        job = await getJob(jobId);
      } catch {
        continue;
      }
      if (!job.ready) continue;

      if (job.success && job.result) {
        const res = job.result as unknown as ProcessResponse;
        const key = res.download_key || nilToNull(res.output_path);
        if (key) setResultKey(key);
        const intent = res.intent || "processed";
        addAi(`Done! ${cap(intent)} applied to ${file?.name ?? "your audio"}.`);
        await appendHistory({ prompt: promptText, audioName: file?.name ?? "", intent, summary: cap(intent) });
        await refreshHistory();
        setState("completed");
        notifyJobDone("Audelle ready", `"${promptText}" is ready`);
        return true;
      }

      if (job.error) {
        addAi(`Sorry, that failed on the server: ${job.error}`);
        setState("analyzed");
        return false;
      }
    }
    addAi("This one is taking a while — still working in the background.");
    setState("analyzed");
    return false;
  };

  const runPrompt = async (promptText: string) => {
    if (!promptText.trim() || !file || !audioPath) return;
    setPrompt("");
    addUser(promptText);
    setState("processing");
    setResultKey(null);

    try {
      const { job_id } = await submitJob("process", { audio_path: audioPath, prompt: promptText });
      await pollJob(job_id, promptText);
    } catch {
      // Offline or server unreachable → offline queue.
      await enqueue({ prompt: promptText, audioName: file.name, asset: { uri: file.uri, name: file.name, mimeType: file.mimeType }, audioPath });
      await refreshQueueCount();
      addAi(`No connection — added to the offline queue. It will process automatically when you're back online.`);
      setState("queued");
      flushQueue();
    }
  };

  const flushQueue = useCallback(async () => {
    const tasks = await getQueue();
    for (const task of tasks) {
      if (task.status === "done" || task.status === "in_progress") continue;
      await updateTask(task.id, { status: "in_progress" });
      try {
        let path = task.audioPath;
        if (!path) {
          const up = await uploadFile({ uri: task.asset.uri, name: task.asset.name, mimeType: task.asset.mimeType });
          path = up.audio_path;
        }
        const { job_id } = await submitJob("process", { audio_path: path, prompt: task.prompt });
        let finished = false;
        for (let i = 0; i < MAX_POLLS && !finished; i++) {
          await new Promise((r) => setTimeout(r, 3000));
          const job = await getJob(job_id);
          if (!job.ready) continue;
          if (job.success) {
            const res = job.result as unknown as ProcessResponse;
            await updateTask(task.id, { status: "done", resultKey: res.download_key || nilToNull(res.output_path) || undefined });
            await appendHistory({ prompt: task.prompt, audioName: task.audioName, intent: res.intent || "processed", summary: cap(res.intent || "processed") });
            notifyJobDone("Audelle ready", `"${task.prompt}" is ready`);
          } else {
            await updateTask(task.id, { status: "failed", error: job.error || "Failed" });
          }
          finished = true;
        }
      } catch {
        await updateTask(task.id, { status: "queued" });
        return; // still offline — stop flushing further tasks
      }
    }
    await refreshQueueCount();
    await refreshHistory();
  }, []);

  const refreshHistory = async () => setHistory(await getHistory());

  const reset = async () => {
    setState("idle");
    setFile(null);
    setAudioPath(null);
    setPrompt("");
    setAnalysis(null);
    setChat([]);
    setResultKey(null);
  };

  const clearHist = async () => {
    await clearHistory();
    await refreshHistory();
  };

  const shareResult = async () => {
    if (!resultKey) return;
    const ok = await downloadAndShare(downloadUrl(resultKey), `audelle-${Date.now()}.mp3`);
    if (!ok) Alert.alert("Export", "Could not download the result right now.");
  };

  const togglePlay = () => {
    if (!resultUrl) return;
    if (status.playing) {
      player.pause();
    } else {
      player.seekTo(0);
      player.play();
    }
  };

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "Done");

  const nilToNull = (v?: string | null) => (v ? v : null);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={reset}>
          <Text style={styles.headerBack}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Audelle</Text>
        <TouchableOpacity onPress={clearHist}>
          <Text style={[styles.headerBack, { width: 60, textAlign: "right" }]}>Clear</Text>
        </TouchableOpacity>
      </View>

      {state === "idle" && (
        <View style={styles.centerContent}>
          <Text style={styles.title}>Edit with AI</Text>
          <Text style={styles.subtitle}>Upload audio or video and describe what you want.</Text>

          <TouchableOpacity style={styles.uploadZone} onPress={pickFile} activeOpacity={0.7}>
            <View style={styles.uploadIcon}>
              <Text style={{ fontSize: 28, color: "rgba(255,255,255,0.4)" }}>+</Text>
            </View>
            <Text style={styles.uploadText}>Tap to upload audio or video</Text>
            <Text style={styles.uploadHint}>MP3, WAV, FLAC, M4A, MP4, MOV</Text>
          </TouchableOpacity>

          {queueCount > 0 && (
            <View style={styles.queueBanner}>
              <Text style={styles.queueBannerText}>{queueCount} queued offline</Text>
              <TouchableOpacity onPress={flushQueue} style={styles.queueSyncBtn}>
                <Text style={styles.queueSyncText}>Sync now</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}

      {state === "uploading" && (
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#00d4ff" />
          <Text style={[styles.subtitle, { marginTop: 16 }]}>Analyzing your file...</Text>
        </View>
      )}

      {state === "queued" && (
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#ffb020" />
          <Text style={[styles.subtitle, { marginTop: 16, color: "rgba(255,255,255,0.6)" }]}>
            Queued offline — will process when connected
          </Text>
          <TouchableOpacity onPress={() => setState("analyzed")} style={styles.queueSyncBtn}>
            <Text style={styles.queueSyncText}>Continue browsing</Text>
          </TouchableOpacity>
        </View>
      )}

      {(state === "analyzed" || state === "processing" || state === "completed") && file && (
        <View style={styles.editorLayout}>
          <View style={styles.fileBar}>
            <View style={styles.fileIcon}>
              <Text style={{ fontSize: 16, color: "#00d4ff" }}>
                {file.mimeType?.startsWith("video/") ? "V" : "A"}
              </Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.fileName} numberOfLines={1}>{file.name}</Text>
              {analysis && (
                <Text style={styles.fileMeta}>
                  {formatDuration(analysis.duration_seconds)} · {analysis.bpm.toFixed(0)} BPM · {analysis.key}
                  {genre?.genre ? ` · ${genre.genre}` : ""}
                </Text>
              )}
            </View>
            {state === "completed" && (
              <View style={styles.doneBadge}>
                <Text style={styles.doneText}>Done</Text>
              </View>
            )}
          </View>

          {queueCount > 0 && (
            <View style={styles.queuedLine}>
              <Text style={styles.queuedLineText}>{queueCount} queued offline</Text>
              <TouchableOpacity onPress={flushQueue}>
                <Text style={styles.queuedLineSync}>Sync now</Text>
              </TouchableOpacity>
            </View>
          )}

          {state === "completed" && resultKey && (
            <View style={styles.resultBar}>
              <TouchableOpacity style={styles.playBtn} onPress={togglePlay}>
                <Text style={styles.playText}>{status.playing ? "Pause" : "Play"}</Text>
              </TouchableOpacity>
              <Text style={styles.resultLabel} numberOfLines={1}>Audelle result</Text>
              <TouchableOpacity style={styles.shareBtn} onPress={shareResult}>
                <Text style={styles.shareText}>Save / Share</Text>
              </TouchableOpacity>
            </View>
          )}

          <ScrollView horizontal style={styles.featuresBar} showsHorizontalScrollIndicator={false}>
            {FEATURES.map((cat) => (
              <TouchableOpacity
                key={cat.category}
                style={[styles.catChip, expandedCat === cat.category && styles.catChipActive]}
                onPress={() => setExpandedCat(expandedCat === cat.category ? null : cat.category)}
              >
                <Text style={[styles.catChipText, expandedCat === cat.category && styles.catChipTextActive]}>
                  {cat.category}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>

          {expandedCat && (
            <ScrollView horizontal style={styles.itemsBar} showsHorizontalScrollIndicator={false}>
              {FEATURES.find((c) => c.category === expandedCat)?.items.map((item) => (
                <TouchableOpacity key={item} style={styles.itemChip} onPress={() => setPrompt(item)}>
                  <Text style={styles.itemChipText}>{item}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          )}

          {genre?.suggested_actions?.length ? (
            <ScrollView horizontal style={styles.itemsBar} showsHorizontalScrollIndicator={false}>
              {genre.suggested_actions.slice(0, 4).map((item) => (
                <TouchableOpacity key={item} style={styles.itemChip} onPress={() => setPrompt(item)}>
                  <Text style={styles.itemChipText}>{item}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          ) : null}

          <ScrollView style={styles.historyBar} horizontal showsHorizontalScrollIndicator={false}>
            {history.length > 0 && (
              <>
                <Text style={styles.historyLabel}>History</Text>
                {history.slice(0, 12).map((h) => (
                  <TouchableOpacity key={h.id} style={styles.historyChip} onPress={() => setPrompt(h.prompt)}>
                    <Text style={styles.historyChipText} numberOfLines={1}>{h.prompt}</Text>
                  </TouchableOpacity>
                ))}
              </>
            )}
          </ScrollView>

          <ScrollView style={styles.chatArea} contentContainerStyle={styles.chatContent}>
            {chat.length === 0 && (
              <View style={styles.emptyChat}>
                <Text style={styles.emptyChatText}>What do you want to do?</Text>
                <Text style={styles.emptyChatHint}>Type a prompt or select a feature above</Text>
              </View>
            )}
            {chat.map((m, i) => (
              <View key={i} style={[styles.message, m.role === "user" ? styles.userMsg : styles.aiMsg]}>
                <Text style={[styles.messageText, m.role === "user" ? styles.userMsgText : styles.aiMsgText]}>{m.text}</Text>
              </View>
            ))}
            {state === "processing" && (
              <View style={[styles.message, styles.aiMsg]}>
                <ActivityIndicator size="small" color="#00d4ff" />
              </View>
            )}
          </ScrollView>

          <View style={styles.inputBar}>
            <TextInput
              style={styles.input}
              value={prompt}
              onChangeText={setPrompt}
              placeholder='e.g. "Remove the vocals"'
              placeholderTextColor="rgba(255,255,255,0.2)"
              editable={state !== "processing"}
              onSubmitEditing={() => runPrompt(prompt)}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={[styles.sendBtn, !prompt.trim() && styles.sendBtnDisabled]}
              onPress={() => runPrompt(prompt)}
              disabled={!prompt.trim() || state === "processing"}
            >
              <LinearGradient
                colors={prompt.trim() ? ["#00d4ff", "#9b59b6"] : ["#333", "#333"]}
                style={styles.sendBtnGradient}
              >
                <Text style={styles.sendBtnText}>{state === "processing" ? "..." : "Go"}</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 56,
    paddingHorizontal: 20,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.05)",
  },
  headerBack: { fontSize: 15, color: "#00d4ff", width: 50 },
  headerTitle: { fontSize: 17, fontWeight: "600", color: "#fff" },
  centerContent: { flex: 1, justifyContent: "center", alignItems: "center", paddingHorizontal: 32 },
  title: { fontSize: 28, fontWeight: "700", color: "#fff", marginBottom: 8 },
  subtitle: { fontSize: 15, color: "rgba(255,255,255,0.4)", textAlign: "center", marginBottom: 32 },
  uploadZone: {
    width: "100%",
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: "rgba(255,255,255,0.15)",
    borderRadius: 20,
    padding: 40,
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.02)",
  },
  uploadIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "rgba(255,255,255,0.05)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  uploadText: { fontSize: 16, color: "rgba(255,255,255,0.6)", marginBottom: 4 },
  uploadHint: { fontSize: 13, color: "rgba(255,255,255,0.25)" },
  editorLayout: { flex: 1 },
  fileBar: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.05)",
    gap: 12,
  },
  fileIcon: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: "rgba(0,212,255,0.15)",
    justifyContent: "center",
    alignItems: "center",
  },
  fileName: { fontSize: 15, fontWeight: "500", color: "#fff" },
  fileMeta: { fontSize: 12, color: "rgba(255,255,255,0.4)", marginTop: 2 },
  doneBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12, backgroundColor: "rgba(34,197,94,0.15)" },
  doneText: { fontSize: 12, color: "#22c55e", fontWeight: "500" },
  queuedLine: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: "rgba(255,176,32,0.08)",
  },
  queuedLineText: { fontSize: 12, color: "#ffb020" },
  queuedLineSync: { fontSize: 12, color: "#00d4ff", fontWeight: "600" },
  resultBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 10,
    gap: 12,
    backgroundColor: "rgba(0,212,255,0.06)",
  },
  playBtn: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: "#00d4ff",
  },
  playText: { fontSize: 13, fontWeight: "700", color: "#000" },
  resultLabel: { flex: 1, fontSize: 13, color: "rgba(255,255,255,0.7)" },
  shareBtn: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12, borderWidth: 1, borderColor: "rgba(0,212,255,0.5)" },
  shareText: { fontSize: 13, color: "#00d4ff", fontWeight: "600" },
  featuresBar: { paddingHorizontal: 12, paddingVertical: 10, maxHeight: 50 },
  catChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.05)",
    marginHorizontal: 4,
  },
  catChipActive: { backgroundColor: "rgba(0,212,255,0.15)" },
  catChipText: { fontSize: 13, color: "rgba(255,255,255,0.5)" },
  catChipTextActive: { color: "#00d4ff" },
  itemsBar: { paddingHorizontal: 12, paddingBottom: 10, maxHeight: 50 },
  itemChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 14,
    backgroundColor: "rgba(0,212,255,0.08)",
    marginHorizontal: 4,
    borderWidth: 1,
    borderColor: "rgba(0,212,255,0.2)",
  },
  itemChipText: { fontSize: 12, color: "#00d4ff" },
  historyBar: { paddingHorizontal: 16, paddingBottom: 8, alignItems: "center", gap: 8 },
  historyLabel: { fontSize: 12, color: "rgba(255,255,255,0.35)", marginRight: 4 },
  historyChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    marginHorizontal: 3,
    maxWidth: 160,
  },
  historyChipText: { fontSize: 12, color: "rgba(255,255,255,0.6)" },
  chatArea: { flex: 1 },
  chatContent: { padding: 16, gap: 10 },
  emptyChat: { flex: 1, justifyContent: "center", alignItems: "center", paddingTop: 80 },
  emptyChatText: { fontSize: 17, color: "rgba(255,255,255,0.3)", marginBottom: 4 },
  emptyChatHint: { fontSize: 13, color: "rgba(255,255,255,0.2)" },
  message: { maxWidth: "85%", paddingHorizontal: 16, paddingVertical: 12, borderRadius: 16 },
  userMsg: { alignSelf: "flex-end", backgroundColor: "rgba(0,212,255,0.15)", borderBottomRightRadius: 4 },
  aiMsg: { alignSelf: "flex-start", backgroundColor: "rgba(255,255,255,0.05)", borderBottomLeftRadius: 4 },
  messageText: { fontSize: 14, lineHeight: 20 },
  userMsgText: { color: "rgba(255,255,255,0.9)" },
  aiMsgText: { color: "rgba(255,255,255,0.6)" },
  queueBanner: {
    marginTop: 24,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    backgroundColor: "rgba(255,176,32,0.08)",
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  queueBannerText: { fontSize: 13, color: "#ffb020", flex: 1 },
  queueSyncBtn: {
    marginTop: 24,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(0,212,255,0.5)",
  },
  queueSyncText: { fontSize: 14, color: "#00d4ff", fontWeight: "600" },
  inputBar: {
    flexDirection: "row",
    padding: 12,
    gap: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(255,255,255,0.05)",
    paddingBottom: 32,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.1)",
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 15,
    color: "#fff",
    backgroundColor: "rgba(255,255,255,0.03)",
  },
  sendBtn: { borderRadius: 14, overflow: "hidden" },
  sendBtnDisabled: { opacity: 0.5 },
  sendBtnGradient: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 14 },
  sendBtnText: { fontSize: 15, fontWeight: "600", color: "#000" },
});