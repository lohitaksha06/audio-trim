import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";

export async function downloadAndShare(url: string, suggestedName: string): Promise<boolean> {
  try {
    const name = suggestedName.replace(/[^a-zA-Z0-9._-]/g, "_");
    const file = new FileSystem.File(FileSystem.Paths.cache, name);
    await FileSystem.File.downloadFileAsync(url, file);
    if (!file.exists) return false;
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(file.uri, {
        mimeType: "audio/mpeg",
        dialogTitle: "Save processed audio",
      });
      return true;
    }
    return file.exists;
  } catch {
    return false;
  }
}