import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

let granted = false;

export async function ensureNotificationPermission(): Promise<boolean> {
  if (granted) return true;
  try {
    const current = await Notifications.getPermissionsAsync();
    if (current.granted) {
      granted = true;
      return true;
    }
    if (Platform.OS === "android") {
      await Notifications.setNotificationChannelAsync("jobs", {
        name: "Job status",
        importance: Notifications.AndroidImportance.DEFAULT,
      });
    }
    const requested = await Notifications.requestPermissionsAsync();
    granted = requested.granted;
    return requested.granted;
  } catch {
    return false;
  }
}

export async function notifyJobDone(title: string, body: string): Promise<void> {
  try {
    if (!(await ensureNotificationPermission())) return;
    await Notifications.scheduleNotificationAsync({
      content: { title, body, sound: true },
      trigger: null,
    });
  } catch {
    // notifications are best-effort
  }
}

export async function scheduleJobDone(title: string, body: string, seconds: number): Promise<void> {
  try {
    if (!(await ensureNotificationPermission())) return;
    await Notifications.scheduleNotificationAsync({
      content: { title, body, sound: true },
      trigger: { type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL, seconds },
    });
  } catch {
    // best-effort
  }
}