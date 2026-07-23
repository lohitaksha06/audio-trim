import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import Svg, { Path } from "react-native-svg";

function WaveBackground() {
  return (
    <View style={StyleSheet.absoluteFill}>
      <Svg width="100%" height="100%" viewBox="0 0 400 800" preserveAspectRatio="none">
        <Path
          d="M0 400 Q100 350 200 400 T400 400 L400 800 L0 800 Z"
          fill="rgba(0,212,255,0.08)"
        />
        <Path
          d="M0 420 Q100 370 200 420 T400 420 L400 800 L0 800 Z"
          fill="rgba(155,89,182,0.06)"
        />
        <Path
          d="M0 440 Q100 390 200 440 T400 440 L400 800 L0 800 Z"
          fill="rgba(0,212,255,0.04)"
        />
      </Svg>
    </View>
  );
}

export default function Home() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <WaveBackground />

      <View style={styles.content}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>AI-POWERED AUDIO EDITOR</Text>
        </View>

        <Text style={styles.title}>
          Describe what{"\n"}you want.
        </Text>
        <Text style={styles.gradientTitle}>
          Let AI handle{"\n"}the rest.
        </Text>

        <Text style={styles.subtitle}>
          Upload audio or video. The AI analyzes every element{"\n"}— instruments, vocals, beats, structure.{"\n"}Then just type what you want.
        </Text>

        <TouchableOpacity
          style={styles.button}
          onPress={() => router.push("/editor")}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={["#00d4ff", "#9b59b6"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.buttonGradient}
          >
            <Text style={styles.buttonText}>Start Editing</Text>
          </LinearGradient>
        </TouchableOpacity>

        <Text style={styles.hint}>
          No account needed
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  content: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 32,
  },
  badge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "rgba(0,212,255,0.3)",
    backgroundColor: "rgba(0,212,255,0.1)",
    marginBottom: 32,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: "600",
    color: "#00d4ff",
    letterSpacing: 1.5,
  },
  title: {
    fontSize: 36,
    fontWeight: "700",
    color: "#fff",
    textAlign: "center",
    lineHeight: 44,
    marginBottom: 4,
  },
  gradientTitle: {
    fontSize: 36,
    fontWeight: "700",
    color: "#00d4ff",
    textAlign: "center",
    lineHeight: 44,
    marginBottom: 24,
  },
  subtitle: {
    fontSize: 15,
    color: "rgba(255,255,255,0.5)",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 40,
  },
  button: {
    width: "100%",
    maxWidth: 280,
    borderRadius: 16,
    overflow: "hidden",
    marginBottom: 16,
  },
  buttonGradient: {
    paddingVertical: 16,
    alignItems: "center",
  },
  buttonText: {
    fontSize: 17,
    fontWeight: "600",
    color: "#000",
  },
  hint: {
    fontSize: 13,
    color: "rgba(255,255,255,0.3)",
  },
});
