import { useState } from "react";
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

const API_BASE = "http://localhost:8000";

const FEATURES = [
  { category: "Source Separation", items: ["Isolate vocals", "Extract drums", "Remove bass", "Separate all stems"] },
  { category: "Smart Editing", items: ["Trim section", "Remove section", "Cut intro", "Highlight reel"] },
  { category: "Mood & Style", items: ["Make darker", "More energetic", "Add reverb", "Add fade"] },
  { category: "Format Conversion", items: ["WAV to MP3", "WAV to FLAC", "MP3 to WAV", "Any to OGG"] },
];

export default function Editor() {
  const [file, setFile] = useState<DocumentPicker.DocumentPickerAsset | null>(null);
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<"idle" | "uploading" | "analyzed" | "processing" | "completed">("idle");
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);
  const [expandedCat, setExpandedCat] = useState<string | null>(null);

  const pickFile = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ["audio/*", "video/*"],
      copyToCacheDirectory: true,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      setFile(asset);
      setState("uploading");

      try {
        const formData = new FormData();
        formData.append("file", {
          uri: asset.uri,
          name: asset.name,
          type: asset.mimeType || "audio/mpeg",
        } as unknown as Blob);

        const res = await fetch(`${API_BASE}/api/upload`, {
          method: "POST",
          body: formData,
          headers: { "Content-Type": "multipart/form-data" },
        });

        if (!res.ok) throw new Error("Upload failed");
        const data = await res.json();
        setAnalysis(data.analysis);
        setState("analyzed");
      } catch {
        Alert.alert("Error", "Failed to upload file");
        setState("idle");
      }
    }
  };

  const process = () => {
    if (!prompt.trim()) return;
    setHistory((prev) => [...prev, { role: "user", text: prompt }]);
    setState("processing");
    setTimeout(() => {
      setHistory((prev) => [...prev, { role: "ai", text: `Done! Applied: "${prompt}"` }]);
      setState("completed");
    }, 2000);
  };

  const reset = () => {
    setState("idle");
    setFile(null);
    setPrompt("");
    setAnalysis(null);
    setHistory([]);
  };

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={reset}>
          <Text style={styles.headerBack}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Audelle</Text>
        <View style={{ width: 50 }} />
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
        </View>
      )}

      {state === "uploading" && (
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#00d4ff" />
          <Text style={[styles.subtitle, { marginTop: 16 }]}>Analyzing your file...</Text>
        </View>
      )}

      {(state === "analyzed" || state === "processing" || state === "completed") && file && (
        <View style={styles.editorLayout}>
          {/* File info */}
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
                  {formatDuration(analysis.duration_seconds as number)} · {(analysis.bpm as number)?.toFixed(0)} BPM · {analysis.key as string}
                </Text>
              )}
            </View>
            {state === "completed" && (
              <View style={styles.doneBadge}>
                <Text style={styles.doneText}>Done</Text>
              </View>
            )}
          </View>

          {/* Features sidebar (scrollable horizontally) */}
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
                <TouchableOpacity
                  key={item}
                  style={styles.itemChip}
                  onPress={() => setPrompt(item)}
                >
                  <Text style={styles.itemChipText}>{item}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          )}

          {/* Chat */}
          <ScrollView style={styles.chatArea} contentContainerStyle={styles.chatContent}>
            {history.length === 0 && (
              <View style={styles.emptyChat}>
                <Text style={styles.emptyChatText}>What do you want to do?</Text>
                <Text style={styles.emptyChatHint}>Type a prompt or select a feature above</Text>
              </View>
            )}
            {history.map((h, i) => (
              <View key={i} style={[styles.message, h.role === "user" ? styles.userMsg : styles.aiMsg]}>
                <Text style={[styles.messageText, h.role === "user" ? styles.userMsgText : styles.aiMsgText]}>
                  {h.text}
                </Text>
              </View>
            ))}
            {state === "processing" && (
              <View style={[styles.message, styles.aiMsg]}>
                <ActivityIndicator size="small" color="#00d4ff" />
              </View>
            )}
          </ScrollView>

          {/* Input */}
          <View style={styles.inputBar}>
            <TextInput
              style={styles.input}
              value={prompt}
              onChangeText={setPrompt}
              placeholder='e.g. "Remove the vocals"'
              placeholderTextColor="rgba(255,255,255,0.2)"
              editable={state !== "processing"}
              onSubmitEditing={process}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={[styles.sendBtn, !prompt.trim() && styles.sendBtnDisabled]}
              onPress={process}
              disabled={!prompt.trim() || state === "processing"}
            >
              <LinearGradient
                colors={prompt.trim() ? ["#00d4ff", "#9b59b6"] : ["#333", "#333"]}
                style={styles.sendBtnGradient}
              >
                <Text style={styles.sendBtnText}>
                  {state === "processing" ? "..." : "Go"}
                </Text>
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
  doneBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: "rgba(34,197,94,0.15)",
  },
  doneText: { fontSize: 12, color: "#22c55e", fontWeight: "500" },
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
