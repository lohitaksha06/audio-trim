import { useEffect } from "react";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { ensureNotificationPermission } from "../lib/notifications";

export default function RootLayout() {
  useEffect(() => {
    ensureNotificationPermission();
  }, []);
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: "#000" },
        }}
      />
    </>
  );
}
