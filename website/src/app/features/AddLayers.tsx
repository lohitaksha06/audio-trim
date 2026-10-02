"use client";

import { ActionButton, Panel } from "./FeaturePanels";
import {
  BEAT_STYLES,
  DSP_INSTRUMENTS,
  TRAINED_INSTRUMENTS,
} from "./instrumentCatalog";

/**
 * Real "add a layer" controls.
 *
 * The instrument list comes from `instrumentCatalog.ts`, which a backend test
 * keeps in sync with the trained timbre model and the recorded drum bank — so
 * every button here corresponds to something that genuinely renders.
 *
 * Trained instruments are labelled as trained; DSP-only sounds say so in their
 * hint rather than pretending to be learned.
 */

function Grid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {children}
    </div>
  );
}

export function AddBeats() {
  return (
    <Panel title="Beats & drum patterns">
      <p className="text-xs text-white/58">
        Every hit is a real recorded drum sample, tempo-locked to your track&apos;s
        measured BPM. Set a target with &ldquo;set tempo to 128 BPM&rdquo; first.
      </p>
      <Grid>
        {BEAT_STYLES.map((b) => (
          <ActionButton
            key={b.prompt}
            label={b.label}
            hint={b.blurb}
            prompt={b.prompt}
            tone="accent"
          />
        ))}
      </Grid>
    </Panel>
  );
}

export function AddInstruments() {
  return (
    <Panel title="Instruments (trained timbre model)">
      <p className="text-xs text-white/58">
        Timbre is learned from real instrument recordings and rendered by a
        synthesiser. Notes follow your track&apos;s detected key, so the layer
        sits in tune.
      </p>
      <Grid>
        {TRAINED_INSTRUMENTS.map((i) => (
          <ActionButton
            key={i.prompt}
            label={i.label}
            hint={i.blurb}
            prompt={i.prompt}
            tone="accent"
          />
        ))}
      </Grid>
    </Panel>
  );
}

export function AddSfx() {
  return (
    <Panel title="Synth sounds (rule-based DSP)">
      <p className="text-xs text-white/58">
        These are oscillator-and-filter synthesis, not trained models. They are
        listed separately so nothing is oversold.
      </p>
      <Grid>
        {DSP_INSTRUMENTS.map((i) => (
          <ActionButton key={i.prompt} label={i.label} hint={i.blurb} prompt={i.prompt} />
        ))}
      </Grid>
    </Panel>
  );
}