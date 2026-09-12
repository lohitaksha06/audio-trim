import { describe, expect, test } from "bun:test";
import { describeResult, isOutputFollowUp } from "./followUp";

describe("isOutputFollowUp", () => {
  const followUps = [
    "now show me the output file to download with drums in the background",
    "show me the output file",
    "send me the download",
    "where is my file?",
    "play the result",
    "download the song",
    "please send me the drums stem",
    "let me hear the mix",
  ];
  for (const p of followUps) {
    test(`follow-up: ${p.slice(0, 40)}`, () => {
      expect(isOutputFollowUp(p)).toBe(true);
    });
  }

  const newEdits = [
    "can you add drums in the background",
    "add funky drums",
    "remove the vocals",
    "trim from 0:05 to 0:10",
    "make voices clearer",
    "convert to house style",
    "hello world",
  ];
  for (const p of newEdits) {
    test(`new edit (not follow-up): ${p.slice(0, 40)}`, () => {
      expect(isOutputFollowUp(p)).toBe(false);
    });
  }
});

describe("describeResult", () => {
  test("drums summary", () => {
    expect(
      describeResult("add_instrument", { added_instrument: "drums", groove: "funky", tempo_bpm: 99.4, hits: 32 }),
    ).toBe("added drums · funky groove · 99 BPM · 32 hits");
  });
  test("combine summary", () => {
    expect(
      describeResult("combine", { combined: ["drums", "bass"], groove: "default", tempo_bpm: 99.4, hits: 71 }),
    ).toBe("added drums + bass · 99 BPM · 71 hits");
  });
  test("boost summary", () => {
    expect(describeResult("boost", { boosted: "drums" })).toBe("turned up drums");
  });
  test("null meta", () => {
    expect(describeResult("trim", null)).toBeNull();
  });
});
