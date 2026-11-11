'use client';

import { Job } from '@prisma/client';
import { useEffect, useMemo, useState } from 'react';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { openImageProfile } from './ImageProfileModal';

// Visualises what extensions/training_insights writes: `insights/*` scalar
// metrics (through the normal loss log), the PCA'd weight trajectory and the
// CLIP map of samples vs. dataset images. The point is to catch a run going
// bad from the weights and samples themselves, before it shows up as burnt
// images or a climbing loss.

interface Props {
  job: Job;
}

interface Point {
  step: number;
  value: number;
}

interface Resume {
  step: number;
  from_step: number;
  jump: number;
  typical_step: number;
  optimizer_restored: boolean;
}

interface Trajectory {
  resumes?: Resume[];
  steps: number[];
  window_losses: (number | null)[];
  coords: number[][];
  explained_variance: number[];
  anchor_steps: number[];
  sample_dim: number;
  total_params: number;
  path_length: number;
  displacement: number;
}

interface MapSample {
  step: number;
  prompt_idx: number;
  path: string;
  x: number;
  y: number;
  base_drift: number;
  nearest_dataset_idx?: number;
  nearest_similarity?: number;
}

interface EmbeddingMapData {
  model: string;
  space?: 'latent' | 'clip';
  dataset: { path: string; x: number; y: number }[];
  samples: MapSample[];
  // near-identical training images [i, j, similarity] (indices into dataset)
  duplicates?: [number, number, number][];
}

interface CapacityModule {
  rank_cap: number;
  erank: number;
  saturation: number;
  stable_rank: number;
  w1_saturation?: number;
  w2_saturation?: number;
  sv: number[];
  captured?: number;
  capture_curve?: [number, number][];
  grad_erank?: number;
  grad_erank_ratio?: number;
  agreement?: number;
  probe_step?: number;
}

interface Capacity {
  kind: 'lora' | 'lokr' | null;
  // per training image: how much of its own gradient the adapter can follow
  files?: Record<string, { captured: number | null; n: number; probes: number }>;
  spectrum_step?: number;
  modules: Record<string, CapacityModule>;
  probe: { step: number; start_step: number; modules: string[]; capture_curve: [number, number][] } | null;
}

interface CaptionProbeData {
  by_band: Record<string, { gain: number; specificity: number | null; n: number }>;
  files: Record<string, { n: number; gain?: number; specificity?: number; trigger_gain?: number }>;
  // what the probe has been doing (also written when it produced no numbers)
  status?: {
    attempts: number;
    recorded: number;
    last_step: number | null;
    variants: string[];
    skipped: string | null;
    error: string | null;
  };
}

function CaptionStatus({ data }: { data: CaptionProbeData | null }) {
  const st = data?.status;
  if (!st) {
    return (
      <div className="text-[11px] text-gray-500 mt-2">
        No caption probe output for this job yet. The probe needs the updated training_insights files and starts with
        the job (restart a running job after copying them); its first numbers appear on the first probe step.
      </div>
    );
  }
  return (
    <div className={classNames('text-[11px] mt-2', st.error ? 'text-red-300' : 'text-gray-500')}>
      Caption probe: {st.recorded} of {st.attempts} probes measured
      {st.last_step != null && `, last at step ${st.last_step}`}
      {st.variants.length > 0 && ` · compares the real caption with: ${st.variants.join(', ')}`}
      {st.skipped && <div className="text-yellow-300">Last skipped: {st.skipped}</div>}
      {st.error && <div>Last error: {st.error}</div>}
    </div>
  );
}

interface InsightEvent {
  step: number;
  time: number;
  reasons: string[];
  action: string;
}

interface Memorized {
  step: number;
  files: { path: string; rel_loss: number; vs_median: number }[];
}

type Status = 'ok' | 'warn' | 'bad' | 'info' | 'none';

interface Verdict {
  status: Status;
  note: string;
}

// timestep bands logged by extensions/training_insights/sample_losses.py,
// least noise (fine detail) first
const NOISE_BANDS = ['00-20', '20-40', '40-60', '60-80', '80-100'];
const NOISE_BAND_COLORS = ['#38bdf8', '#34d399', '#a3e635', '#fbbf24', '#f87171'];

const METRIC_KEYS = [
  'insights/update_ratio',
  'insights/grad_norm',
  'insights/grad_norm_max',
  'insights/grad_clip_fraction',
  'insights/nonfinite_grad_steps',
  'insights/direction_consistency',
  'insights/path_efficiency',
  'insights/weight_norm',
  'insights/distance_from_start',
  'insights/sample_diversity',
  'insights/sample_dataset_similarity',
  'insights/sample_drift_from_base',
  ...NOISE_BANDS.map(b => `insights/loss_noise/${b}`),
  'insights/file_loss_spread',
  'insights/memorized_fraction',
  'val/loss',
  'insights/capacity/saturation_median',
  'insights/capacity/saturation_p90',
  'insights/capacity/saturated_fraction',
  'insights/capacity/w2_saturation_median',
  'insights/capacity/grad_captured',
  'insights/capacity/headroom_4x',
  'insights/capacity/capture_1x',
  'insights/capacity/grad_erank_ratio',
  'insights/capacity/sample_agreement',
  'insights/capacity/grad_captured_signal',
  'insights/capacity/signal_share',
  'insights/concept_strength',
  'insights/control_concept',
  'insights/concept_bleed',
  'insights/control_drift',
  'insights/concept_drift',
  'insights/caption/gain',
  'insights/caption/specificity',
  'insights/caption/trigger_gain',
];
// fetched with a stride: only needed to line up with val/loss, and long runs
// have more steps than one request returns
const TRAIN_LOSS_KEY = 'loss/loss';
const TRAIN_LOSS_STRIDE = 5;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function imgSrc(p: string, thumb = true) {
  return `/api/img/${encodeURIComponent(p)}${thumb ? '?thumb=1' : ''}`;
}

function basename(p: string) {
  return p.replace(/^.*[\\/]/, '');
}

function fmt(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (v === 0) return '0';
  if (abs < 1e-3 || abs >= 1e5) return v.toExponential(2);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(3);
  return v.toPrecision(3);
}

function median(xs: number[]) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mean(xs: number[]) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}

function last<T>(xs: T[], n: number) {
  return xs.slice(Math.max(0, xs.length - n));
}

// purple -> blue -> green -> yellow, for "early -> late"
function progressColor(t: number) {
  const c = Math.max(0, Math.min(1, t));
  return `hsl(${270 - 220 * c} 80% 60%)`;
}

const PROMPT_COLORS = [
  '#60a5fa',
  '#f472b6',
  '#34d399',
  '#fbbf24',
  '#a78bfa',
  '#f87171',
  '#22d3ee',
  '#a3e635',
  '#fb923c',
  '#e879f9',
];

const STATUS_STYLES: Record<Status, string> = {
  ok: 'bg-green-500',
  warn: 'bg-yellow-400',
  bad: 'bg-red-500',
  info: 'bg-blue-400',
  none: 'bg-gray-600',
};

// ---------------------------------------------------------------------------
// heuristics — deliberately conservative, they are hints not verdicts
// ---------------------------------------------------------------------------

type Series = Record<string, Point[]>;
const vals = (s: Series, k: string) => (s[k] ?? []).map(p => p.value);

// "this sample is a copy" thresholds (warn, bad). Pooled VAE latents only see
// layout and colour, so unrelated images already score higher there than in CLIP.
function copyThresholds(space?: string): [number, number] {
  return space === 'clip' ? [0.9, 0.95] : [0.93, 0.97];
}

const NOT_YET: Verdict = { status: 'none', note: 'Waiting for data.' };

// the run's own early level, so warnings don't depend on the optimizer/model:
// median of the first 20% (at least 3 points)
function baseline(v: number[]) {
  return median(v.slice(0, Math.max(3, Math.floor(v.length / 5))));
}

function judgeUpdateRatio(s: Series): Verdict {
  const v = vals(s, 'insights/update_ratio');
  if (!v.length) return NOT_YET;
  const m = median(last(v, 10));
  // absolute sanity bounds first, they hold for any optimizer
  if (m > 5e-2)
    return { status: 'bad', note: 'Each step changes the weights by >5% — very likely diverging. Lower the LR.' };
  if (m < 1e-7)
    return {
      status: 'warn',
      note: 'Weights are barely moving — LR too low, or the trainable params are not being updated.',
    };
  if (v.length >= 10) {
    const r = m / baseline(v);
    if (r > 10) return { status: 'bad', note: `Steps are ${r.toFixed(0)}× larger than early in this run.` };
    if (r > 4) return { status: 'warn', note: `Steps are ${r.toFixed(1)}× larger than early in this run.` };
    if (r < 0.05)
      return {
        status: 'info',
        note: `Steps shrank to ${(r * 100).toFixed(0)}% of their early size (LR decay, or stalling).`,
      };
    return { status: 'ok', note: `${r.toFixed(2)}× this run's early step size.` };
  }
  if (m > 1e-2)
    return { status: 'warn', note: 'Steps are large (>1% of the weights). Fine briefly, risky if it persists.' };
  return { status: 'ok', note: 'Within the usual 1e-7 – 1e-2 band.' };
}

function judgeGradNorm(s: Series): Verdict {
  const g = vals(s, 'insights/grad_norm');
  const gmax = vals(s, 'insights/grad_norm_max');
  if (g.length < 3) return NOT_YET;
  const base = median(g);
  const spike = Math.max(...last(gmax.length ? gmax : g, 5)) / base;
  if (spike > 20) return { status: 'bad', note: `Recent gradient spike ${spike.toFixed(0)}× the run median.` };
  if (spike > 5) return { status: 'warn', note: `Recent gradient spike ${spike.toFixed(1)}× the run median.` };
  const n = Math.max(2, Math.floor(g.length / 5));
  const growth = median(last(g, n)) / median(g.slice(0, n));
  if (g.length >= 20 && growth > 3)
    return { status: 'warn', note: `Gradients are ${growth.toFixed(1)}× larger than early in the run and rising.` };
  return { status: 'ok', note: 'No spikes; the faint line is the per-window max.' };
}

function judgeClip(s: Series): Verdict {
  const v = vals(s, 'insights/grad_clip_fraction');
  if (!v.length) return { status: 'none', note: 'Only reported when gradient clipping is active.' };
  const m = mean(last(v, 10));
  if (m > 0.9)
    return {
      status: 'bad',
      note: 'Almost every step is clipped — the LR is likely too high (or max_grad_norm too low).',
    };
  if (m > 0.5) return { status: 'warn', note: 'Most steps are being clipped.' };
  return { status: 'ok', note: `${(m * 100).toFixed(0)}% of recent steps clipped.` };
}

function judgeNonFinite(s: Series): Verdict {
  const v = vals(s, 'insights/nonfinite_grad_steps');
  if (!v.length) return NOT_YET;
  const recent = last(v, 10).reduce((a, b) => a + b, 0);
  const total = v.reduce((a, b) => a + b, 0);
  if (recent > 0) return { status: 'bad', note: `${recent} recent steps had NaN/Inf gradients.` };
  if (total > 0) return { status: 'warn', note: `${total} NaN/Inf gradient steps earlier in the run.` };
  return { status: 'ok', note: 'No NaN/Inf gradients.' };
}

function judgeDirection(s: Series): Verdict {
  const v = vals(s, 'insights/direction_consistency');
  if (v.length < 3) return NOT_YET;
  const m = mean(last(v, 10));
  // compared with this run's own early level as well as absolutely
  const drop = v.length >= 10 ? baseline(v) - m : 0;
  if (m < -0.5 || drop > 0.6)
    return {
      status: 'bad',
      note: 'Consecutive updates point in opposite directions — bouncing across a valley. LR too high.',
    };
  if (m < -0.2 || drop > 0.35)
    return { status: 'warn', note: 'Updates are starting to reverse each other (oscillation).' };
  if (m < 0.05) return { status: 'info', note: 'Close to a random walk — normal late in training / near convergence.' };
  return { status: 'ok', note: 'Updates keep heading the same way.' };
}

function judgeEfficiency(s: Series): Verdict {
  const v = vals(s, 'insights/path_efficiency');
  if (!v.length) return NOT_YET;
  return {
    status: 'info',
    note: '1 = straight line from the start, near 0 = lots of wandering for little net movement.',
  };
}

function judgeWeightNorm(s: Series): Verdict {
  const v = vals(s, 'insights/weight_norm');
  if (v.length < 12) return NOT_YET;
  const q = Math.floor(v.length / 4);
  const a = v[v.length - 1 - 2 * q],
    b = v[v.length - 1 - q],
    c = v[v.length - 1];
  const prev = (b - a) / Math.abs(a || 1);
  const cur = (c - b) / Math.abs(b || 1);
  if (cur > 0.1 && cur > 2 * Math.max(prev, 0.01))
    return {
      status: 'warn',
      note: `Weight norm growth is accelerating (+${(cur * 100).toFixed(0)}% in the last quarter of the run).`,
    };
  return { status: 'info', note: 'Steady growth is normal; accelerating growth often precedes overcooking.' };
}

function judgeDistance(s: Series): Verdict {
  const v = vals(s, 'insights/distance_from_start');
  if (!v.length) return NOT_YET;
  return { status: 'info', note: 'L2 distance from the starting weights (estimated from the sampled coordinates).' };
}

function judgeDiversity(s: Series): Verdict {
  const v = vals(s, 'insights/sample_diversity');
  if (v.length < 2) return { status: 'none', note: 'Needs ≥2 sample prompts and ≥2 sample steps.' };
  const drop = 1 - v[v.length - 1] / v[0];
  if (drop > 0.5)
    return {
      status: 'bad',
      note: `Samples are ${(drop * 100).toFixed(0)}% less varied than at the start — prompts are collapsing into one look.`,
    };
  if (drop > 0.3) return { status: 'warn', note: `Sample variety down ${(drop * 100).toFixed(0)}% from the start.` };
  return { status: 'ok', note: 'Different prompts still produce different images.' };
}

function judgeSimilarity(s: Series, space?: string): Verdict {
  const v = vals(s, 'insights/sample_dataset_similarity');
  if (!v.length) return NOT_YET;
  const x = v[v.length - 1];
  const [warn, bad] = copyThresholds(space);
  if (x > bad) return { status: 'bad', note: 'Samples are near-copies of training images (memorisation).' };
  if (x > warn) return { status: 'warn', note: 'Samples are getting very close to individual training images.' };
  return {
    status: 'ok',
    note: `Rising = learning the look of the dataset. Above ~${warn} starts to look like copying.`,
  };
}

function judgeDrift(s: Series): Verdict {
  const v = vals(s, 'insights/sample_drift_from_base');
  if (v.length < 2) return NOT_YET;
  return { status: 'info', note: 'How far samples moved from the base model. Should rise, then level off.' };
}

// band series normalised to each band's own early level, so the lines are comparable
function bandLines(s: Series): { points: Point[]; color: string }[] {
  return NOISE_BANDS.map((b, i) => {
    const pts = s[`insights/loss_noise/${b}`] ?? [];
    const base = pts.length ? baseline(pts.map(p => p.value)) : 1;
    return { points: pts.map(p => ({ step: p.step, value: p.value / (base || 1) })), color: NOISE_BAND_COLORS[i] };
  });
}

function judgeNoiseBands(s: Series): Verdict {
  const lines = bandLines(s);
  if (lines.some(l => l.points.length < 10)) return NOT_YET;
  const recent = lines.map(l => median(last(l.points, 10).map(p => p.value)));
  const low = recent[0],
    high = recent[recent.length - 1];
  if (low < 0.75 && high > 0.95)
    return {
      status: 'warn',
      note: `Detail (low-noise) loss is down to ${(low * 100).toFixed(0)}% while composition (high-noise) loss has stalled — a typical overfitting pattern.`,
    };
  return {
    status: 'ok',
    note: 'Each line is a noise band vs its own early level. Watch for blue (fine detail) falling while red (composition) stops.',
  };
}

function judgeSpread(s: Series): Verdict {
  const v = vals(s, 'insights/file_loss_spread');
  if (v.length < 5) return { status: 'none', note: 'Needs a few epochs of per-file losses.' };
  const r = median(last(v, 5)) / baseline(v);
  if (r > 1.5)
    return {
      status: 'warn',
      note: `Per-file losses are ${r.toFixed(1)}× more spread out than early on — some images are treated very differently.`,
    };
  return { status: 'info', note: 'How differently the model treats individual images (noise-adjusted).' };
}

function judgeMemorized(s: Series): Verdict {
  const v = vals(s, 'insights/memorized_fraction');
  if (!v.length) return { status: 'none', note: 'Needs a few epochs of per-file losses.' };
  const x = median(last(v, 5));
  if (x > 0.3)
    return {
      status: 'bad',
      note: `${(x * 100).toFixed(0)}% of images are far easier than the rest — likely memorised.`,
    };
  if (x > 0.15)
    return { status: 'warn', note: `${(x * 100).toFixed(0)}% of images are far easier than the rest (listed below).` };
  return { status: 'ok', note: 'Share of images with under half the median noise-adjusted loss.' };
}

// val/loss divided by the mean train loss since the previous validation. The
// two use different timesteps, so only the trend of the ratio means anything.
function validationGap(s: Series): Point[] {
  const val = s['val/loss'] ?? [];
  const train = s[TRAIN_LOSS_KEY] ?? [];
  if (!val.length || !train.length) return [];
  const out: Point[] = [];
  let j = 0,
    prev = -Infinity;
  for (const v of val) {
    let sum = 0,
      n = 0;
    while (j < train.length && train[j].step <= v.step) {
      if (train[j].step > prev) {
        sum += train[j].value;
        n++;
      }
      j++;
    }
    if (n > 0 && sum > 0) out.push({ step: v.step, value: v.value / (sum / n) });
    prev = v.step;
  }
  return out;
}

function judgeValidationGap(s: Series): Verdict {
  const g = validationGap(s).map(p => p.value);
  if (!(s['val/loss'] ?? []).length) return { status: 'none', note: 'Turn on Validation in the job to get this.' };
  if (g.length < 4) return NOT_YET;
  const r = median(last(g, 3)) / baseline(g);
  if (r > 1.5)
    return {
      status: 'bad',
      note: `Validation loss is ${((r - 1) * 100).toFixed(0)}% worse relative to training than early on — overfitting.`,
    };
  if (r > 1.25)
    return { status: 'warn', note: `Validation is falling behind training (+${((r - 1) * 100).toFixed(0)}% gap).` };
  return { status: 'ok', note: 'Validation vs training loss ratio; a rising line means overfitting.' };
}

function latest(s: Series, k: string): number | null {
  const v = s[k] ?? [];
  return v.length ? v[v.length - 1].value : null;
}

// Capture of the probe gradient, preferring the noise-corrected value. The plain
// value of a noise-dominated gradient (diffusion: random timestep + noise per
// sample) is ~k/N of isotropic noise, which looks like "far too small" for any
// adapter, so without a clear shared signal we say we can't tell.
function capturedSignal(s: Series): { value: number | null; noisy: boolean } {
  const signal = latest(s, 'insights/capacity/grad_captured_signal');
  if (signal != null) return { value: signal, noisy: false };
  const share = latest(s, 'insights/capacity/signal_share');
  return { value: null, noisy: share != null && share < 0.2 };
}

function judgeCaptured(s: Series): Verdict {
  const raw = latest(s, 'insights/capacity/grad_captured');
  if (raw == null) return { status: 'none', note: 'First gradient probe runs after ~250 steps.' };
  const { value: x, noisy } = capturedSignal(s);
  if (x == null)
    return {
      status: 'info',
      note: noisy
        ? `The probe gradient is mostly per-batch noise (signal ${((latest(s, 'insights/capacity/signal_share') ?? 0) * 100).toFixed(0)}%), so capacity can't be judged yet. Raise probe_span (e.g. 16-32) for a clearer signal.`
        : 'Not enough shared signal in the probe yet to judge capacity.',
    };
  // Low capture alone is not a problem: the full gradient also asks to reproduce
  // every training image exactly, and an adapter that can't follow all of that
  // is doing its job. It only matters when the samples are missing the concept.
  if (x < 0.35)
    return {
      status: 'info',
      note: `The adapter can follow ${(x * 100).toFixed(0)}% of the noise-corrected gradient per step. Normal for small adapters (most of the gradient is image-specific detail); only a capacity problem if the samples are still missing the concept.`,
    };
  return { status: 'ok', note: `The adapter can follow ${(x * 100).toFixed(0)}% of the noise-corrected gradient.` };
}

function judgeHeadroom(s: Series): Verdict {
  // a multiple (capture at 4x / at 1x): an absolute difference looks small
  // exactly when the adapter is hopelessly small (4% -> 15% is 3.8x)
  const x = latest(s, 'insights/capacity/headroom_4x');
  const base = latest(s, 'insights/capacity/capture_1x');
  if (x == null) return { status: 'none', note: 'Comes with the gradient probe.' };
  if (x > 2 && (base == null || base < 0.5))
    return {
      status: 'warn',
      note: `4× the adapter structure would capture ${x.toFixed(1)}× as much of the gradient — a bigger adapter would help.`,
    };
  return {
    status: 'ok',
    note: `4× the structure would capture only ${x.toFixed(1)}× as much — size is not the bottleneck.`,
  };
}

function judgeSaturation(s: Series): Verdict {
  const x = latest(s, 'insights/capacity/saturation_p90');
  if (x == null) return NOT_YET;
  if (x > 0.9)
    return { status: 'warn', note: 'Most modules use nearly all the rank they have — no room left to grow.' };
  return { status: 'ok', note: `90% of modules use under ${(x * 100).toFixed(0)}% of their rank.` };
}

function judgeAgreement(s: Series): Verdict {
  const x = latest(s, 'insights/capacity/sample_agreement');
  if (x == null) return { status: 'none', note: 'Comes with the gradient probe.' };
  if (x < -0.02)
    return {
      status: 'warn',
      note: 'Gradients from different samples cancel out — images/captions pull in opposite directions (a data problem, not capacity).',
    };
  return {
    status: 'info',
    note: "Mean cosine between samples' gradients. Near 0 is normal for diffusion (noise dominates); negative = conflicting data.",
  };
}

function judgeCaptionGain(s: Series): Verdict {
  const x = latest(s, 'insights/caption/gain');
  if (x == null)
    return { status: 'none', note: 'Needs a few caption probes (one every 25 steps) and a blank-caption embedding.' };
  if (x < 0.01)
    return {
      status: 'warn',
      note: `A blank caption predicts almost as well (${(x * 100).toFixed(1)}%) — the model is not using the captions, or they carry no information.`,
    };
  return { status: 'ok', note: `The real caption lowers the loss by ${(x * 100).toFixed(1)}% vs a blank one.` };
}

function judgeCaptionSpecificity(s: Series): Verdict {
  const x = latest(s, 'insights/caption/specificity');
  const gain = latest(s, 'insights/caption/gain');
  if (x == null) return { status: 'none', note: 'Needs a few caption probes.' };
  if (x < 0.005 && (gain == null || gain > 0.01))
    return {
      status: 'warn',
      note: `Another image's caption works almost as well (${(x * 100).toFixed(1)}%) — captions are too generic to tell the images apart.`,
    };
  return { status: 'ok', note: `Each image's own caption beats another image's by ${(x * 100).toFixed(1)}%.` };
}

function judgeTrigger(s: Series): Verdict {
  const x = latest(s, 'insights/caption/trigger_gain');
  if (x == null)
    return { status: 'none', note: 'Needs a trigger word and a loaded text encoder (not cached text embeddings).' };
  if (x < 0.003)
    return {
      status: 'warn',
      note: 'Removing the trigger word changes almost nothing — it is not bound to the concept (yet).',
    };
  return { status: 'ok', note: `The trigger word lowers the loss by ${(x * 100).toFixed(1)}%.` };
}

function judgeConceptStrength(s: Series): Verdict {
  const v = vals(s, 'insights/concept_strength');
  if (!v.length)
    return {
      status: 'none',
      note: 'Set a Concept Prompt in the job\'s Insights settings (e.g. "translucent slime skin") to score the concept itself.',
    };
  const x = v[v.length - 1];
  const control = latest(s, 'insights/control_concept');
  if (control != null && x > 0.2 && control > 0.6 * x)
    return {
      status: 'warn',
      note: `The concept also shows up without the trigger (${(control * 100).toFixed(0)}% vs ${(x * 100).toFixed(0)}%) — it is not bound to the trigger word.`,
    };
  return {
    status: x >= 0.5 ? 'ok' : 'info',
    note: `${(x * 100).toFixed(0)}% of the way from the first samples to the dataset's level of the concept (faint line: prompts without the trigger).`,
  };
}

function judgeBleed(s: Series): Verdict {
  const x = latest(s, 'insights/concept_bleed');
  if (x == null) return { status: 'none', note: 'Add a sample prompt without the trigger word to measure this.' };
  if (x > 0.7)
    return {
      status: 'warn',
      note: `Prompts without the trigger change ${(x * 100).toFixed(0)}% as much as with it — the whole model is drifting toward the dataset, not just the concept (try Differential Output Preservation, better captions, a lower LR).`,
    };
  return {
    status: 'ok',
    note: `Prompts without the trigger change ${(x * 100).toFixed(0)}% as much as with it — the change stays mostly on the trigger.`,
  };
}

interface CardDef {
  key: string;
  overlayKey?: string;
  title: string;
  judge: (s: Series, space?: string) => Verdict;
  logScale?: boolean;
  group: 'Weights & gradients' | 'Samples' | 'Dataset fit' | 'Capacity' | 'Captions';
  // plotted instead of series[key] when given
  derive?: (s: Series) => Point[];
  lines?: (s: Series) => { points: Point[]; color: string }[];
}

const GROUPS = ['Weights & gradients', 'Samples', 'Dataset fit', 'Captions', 'Capacity'] as const;

const CARDS: CardDef[] = [
  {
    key: 'insights/update_ratio',
    title: 'Update / weight ratio',
    judge: judgeUpdateRatio,
    logScale: true,
    group: 'Weights & gradients',
  },
  {
    key: 'insights/grad_norm',
    overlayKey: 'insights/grad_norm_max',
    title: 'Gradient norm (pre-clip)',
    judge: judgeGradNorm,
    logScale: true,
    group: 'Weights & gradients',
  },
  { key: 'insights/grad_clip_fraction', title: 'Clipped steps', judge: judgeClip, group: 'Weights & gradients' },
  {
    key: 'insights/nonfinite_grad_steps',
    title: 'NaN / Inf gradients',
    judge: judgeNonFinite,
    group: 'Weights & gradients',
  },
  {
    key: 'insights/direction_consistency',
    title: 'Direction consistency',
    judge: judgeDirection,
    group: 'Weights & gradients',
  },
  { key: 'insights/path_efficiency', title: 'Path efficiency', judge: judgeEfficiency, group: 'Weights & gradients' },
  { key: 'insights/weight_norm', title: 'Weight norm', judge: judgeWeightNorm, group: 'Weights & gradients' },
  {
    key: 'insights/distance_from_start',
    title: 'Distance from start',
    judge: judgeDistance,
    group: 'Weights & gradients',
  },
  { key: 'insights/sample_diversity', title: 'Sample diversity', judge: judgeDiversity, group: 'Samples' },
  {
    key: 'insights/sample_dataset_similarity',
    title: 'Similarity to dataset',
    judge: judgeSimilarity,
    group: 'Samples',
  },
  { key: 'insights/sample_drift_from_base', title: 'Drift from base model', judge: judgeDrift, group: 'Samples' },
  {
    key: 'insights/loss_noise/00-20',
    title: 'Loss by noise level',
    judge: judgeNoiseBands,
    lines: bandLines,
    group: 'Dataset fit',
  },
  { key: 'insights/file_loss_spread', title: 'Per-file loss spread', judge: judgeSpread, group: 'Dataset fit' },
  { key: 'insights/memorized_fraction', title: 'Memorised images', judge: judgeMemorized, group: 'Dataset fit' },
  {
    key: 'val/loss',
    title: 'Validation gap',
    judge: judgeValidationGap,
    derive: validationGap,
    group: 'Dataset fit',
  },
  { key: 'insights/capacity/grad_captured', title: 'Gradient captured', judge: judgeCaptured, group: 'Capacity' },
  { key: 'insights/capacity/headroom_4x', title: 'Headroom at 4× size', judge: judgeHeadroom, group: 'Capacity' },
  {
    key: 'insights/capacity/saturation_p90',
    overlayKey: 'insights/capacity/saturation_median',
    title: 'Rank saturation (p90)',
    judge: judgeSaturation,
    group: 'Capacity',
  },
  {
    key: 'insights/capacity/sample_agreement',
    title: 'Sample agreement',
    judge: judgeAgreement,
    group: 'Capacity',
  },
  {
    key: 'insights/concept_strength',
    overlayKey: 'insights/control_concept',
    title: 'Concept strength',
    judge: judgeConceptStrength,
    group: 'Samples',
  },
  {
    key: 'insights/concept_bleed',
    title: 'Concept bleed',
    judge: judgeBleed,
    group: 'Samples',
  },
  { key: 'insights/caption/gain', title: 'Caption helps', judge: judgeCaptionGain, group: 'Captions' },
  {
    key: 'insights/caption/specificity',
    title: 'Caption fits its image',
    judge: judgeCaptionSpecificity,
    group: 'Captions',
  },
  { key: 'insights/caption/trigger_gain', title: 'Trigger word bound', judge: judgeTrigger, group: 'Captions' },
];

// ---------------------------------------------------------------------------
// data hooks
// ---------------------------------------------------------------------------

function useInsightMetrics(jobID: string) {
  const [series, setSeries] = useState<Series>({});
  usePollLoop(
    async () => {
      // one request for just the keys this tab uses (columnar steps/values)
      const res = await apiClient.get(`/api/jobs/${jobID}/loss`, {
        params: { keys: [...METRIC_KEYS, TRAIN_LOSS_KEY].join(',') },
      });
      const incoming: Record<string, { steps: number[]; values: (number | null)[] }> = res.data?.series ?? {};
      const out: Series = {};
      for (const [key, col] of Object.entries(incoming)) {
        // only needed to line up with val/loss: thin it out
        const stride = key === TRAIN_LOSS_KEY ? TRAIN_LOSS_STRIDE : 1;
        const pts: Point[] = [];
        for (let i = 0; i < col.steps.length; i += stride) {
          const v = col.values[i];
          if (v != null && Number.isFinite(v)) pts.push({ step: col.steps[i], value: v });
        }
        out[key] = pts;
      }
      setSeries(out);
    },
    10000,
    [jobID],
  );
  return series;
}

function useInsightFiles(jobID: string) {
  const [data, setData] = useState<{
    trajectory: Trajectory | null;
    embeddingMap: EmbeddingMapData | null;
    memorized: Memorized | null;
    capacity: Capacity | null;
    captionProbe: CaptionProbeData | null;
    events: { events: InsightEvent[] } | null;
  } | null>(null);
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${jobID}/insights`)
        .then(res => setData(res.data))
        .catch(() => {}),
    15000,
    [jobID],
  );
  return data;
}

// ---------------------------------------------------------------------------
// health cards
// ---------------------------------------------------------------------------

function downsample(points: Point[], max = 160): Point[] {
  if (points.length <= max) return points;
  const bucket = points.length / max;
  const out: Point[] = [];
  for (let i = 0; i < max; i++) {
    const chunk = points.slice(Math.floor(i * bucket), Math.floor((i + 1) * bucket));
    out.push({ step: chunk[chunk.length - 1].step, value: mean(chunk.map(p => p.value)) });
  }
  return out;
}

function Sparkline({
  points,
  overlay,
  logScale,
  marks = [],
  lines,
}: {
  points: Point[];
  overlay?: Point[];
  logScale?: boolean;
  marks?: number[];
  // several equal lines instead of main + overlay
  lines?: { points: Point[]; color: string }[];
}) {
  const W = 240,
    H = 44;
  const tf = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-12)) : v);
  const multi = (lines ?? [])
    .map(l => ({ points: downsample(l.points), color: l.color }))
    .filter(l => l.points.length > 1);
  const main = lines ? (multi[0]?.points ?? []) : downsample(points);
  const over = lines ? multi.slice(1).flatMap(l => l.points) : overlay ? downsample(overlay) : [];
  if (main.length < 2)
    return <div className="h-11 text-[11px] text-gray-600 italic flex items-center">no data yet</div>;
  const all = [...main, ...over];
  const x0 = Math.min(...all.map(p => p.step)),
    x1 = Math.max(...all.map(p => p.step));
  const ys = all.map(p => tf(p.value));
  let y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  if (y0 === y1) {
    y0 -= 1;
    y1 += 1;
  }
  const path = (pts: Point[]) =>
    pts
      .map((p, i) => {
        const x = ((p.step - x0) / Math.max(1, x1 - x0)) * W;
        const y = H - 2 - ((tf(p.value) - y0) / (y1 - y0)) * (H - 4);
        return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-11">
      {marks
        .filter(m => m > x0 && m < x1)
        .map(m => {
          const x = ((m - x0) / Math.max(1, x1 - x0)) * W;
          return (
            <line
              key={m}
              x1={x}
              x2={x}
              y1={0}
              y2={H}
              stroke="#f59e0b"
              strokeOpacity={0.6}
              strokeDasharray="3 3"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      {lines &&
        multi.map((l, i) => (
          <path
            key={i}
            d={path(l.points)}
            fill="none"
            stroke={l.color}
            strokeWidth={1.25}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      {!lines && over.length > 1 && (
        <path
          d={path(over)}
          fill="none"
          stroke="#94a3b8"
          strokeOpacity={0.35}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      )}
      <path d={path(main)} fill="none" stroke="#60a5fa" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function HealthCard({ def, series, space, marks }: { def: CardDef; series: Series; space?: string; marks: number[] }) {
  const points = def.derive ? def.derive(series) : (series[def.key] ?? []);
  const verdict = def.judge(series, space);
  const latest = points.length ? points[points.length - 1] : null;
  return (
    <div className="bg-gray-900 rounded-lg p-3 flex flex-col gap-1 min-w-0">
      <div className="flex items-center gap-2">
        <span className={classNames('w-2.5 h-2.5 rounded-full shrink-0', STATUS_STYLES[verdict.status])} />
        <span className="text-xs uppercase tracking-wide text-gray-400 truncate">{def.title}</span>
        <span className="flex-1" />
        <span className="text-sm font-mono text-gray-100">{fmt(latest?.value)}</span>
      </div>
      <Sparkline
        points={points}
        overlay={def.overlayKey ? series[def.overlayKey] : undefined}
        logScale={def.logScale}
        marks={marks}
        lines={def.lines ? def.lines(series) : undefined}
      />
      <div className="text-[11px] leading-snug text-gray-400 min-h-[2.2em]">{verdict.note}</div>
    </div>
  );
}

function resumeProblems(resumes: Resume[]): { status: Status; text: string }[] {
  const out: { status: Status; text: string }[] = [];
  for (const r of resumes) {
    if (!r.optimizer_restored) {
      out.push({
        status: 'bad',
        text: `resumed at step ${r.step} without optimizer state — the optimizer started over (adaptive ones like Prodigy re-learn their step size)`,
      });
    }
    const ratio = r.typical_step > 0 ? r.jump / r.typical_step : 0;
    if (ratio > 3) {
      out.push({
        status: 'warn',
        text: `weights jumped ${ratio.toFixed(1)}× a normal step when resuming at step ${r.step} — it resumed from different weights than it stopped with (EMA weights are what get saved when EMA is on)`,
      });
    }
  }
  return out;
}

function MemorizedList({ memorized, jobID }: { memorized: Memorized | null; jobID: string }) {
  if (!memorized || !memorized.files.length) return null;
  return (
    <div className="bg-gray-900 rounded-lg p-3 mt-3">
      <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">
        Images far easier than the rest (step {memorized.step})
      </div>
      <div className="flex flex-wrap gap-3">
        {memorized.files.map(f => (
          <button
            key={f.path}
            type="button"
            onClick={() => openImageProfile(jobID, f.path)}
            className="w-20 text-left text-[10px] text-gray-400"
            title={f.path}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={imgSrc(f.path)} alt="" className="w-20 h-20 object-contain bg-black rounded" />
            <div className="truncate mt-0.5">{basename(f.path)}</div>
            <div className="font-mono text-sky-300">{(f.vs_median * 100).toFixed(0)}% of median</div>
          </button>
        ))}
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        Noise-adjusted loss under half the dataset median: the model reproduces these much more easily than the rest,
        which usually means they are memorised (duplicates, very simple images, or too many repeats).
      </p>
    </div>
  );
}

function BucketsPanel({ jobID }: { jobID: string }) {
  const [buckets, setBuckets] = useState<
    { width: number; height: number; samples: number; files: number; difficulty: number }[]
  >([]);
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${jobID}/file_loss`, { params: { view: 'buckets' } })
        .then(res => setBuckets(res.data?.buckets ?? []))
        .catch(() => {}),
    30000,
    [jobID],
  );
  if (buckets.length < 2) return null;
  const max = Math.max(1.5, ...buckets.map(b => b.difficulty));
  return (
    <div className="bg-gray-900 rounded-lg p-3 mt-3">
      <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Difficulty by training resolution</div>
      <div className="space-y-1">
        {buckets.map(b => (
          <div key={`${b.width}x${b.height}`} className="flex items-center gap-2 text-[11px]">
            <span className="w-24 shrink-0 font-mono text-gray-300">
              {b.width}×{b.height}
            </span>
            <div className="flex-1 h-2.5 bg-gray-950 rounded relative">
              <div
                className={classNames('h-full rounded', b.difficulty > 1.2 ? 'bg-orange-400' : 'bg-blue-400')}
                style={{ width: `${(b.difficulty / max) * 100}%` }}
              />
              <div className="absolute top-0 h-full border-l border-gray-500" style={{ left: `${(1 / max) * 100}%` }} />
            </div>
            <span className="w-10 text-right font-mono">{b.difficulty.toFixed(2)}</span>
            <span className="w-24 text-gray-500">
              {b.files} files · {b.samples}
            </span>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        Noise-adjusted loss per bucket (1.0 = typical, grey line). One resolution clearly harder than the rest points at
        cropping/resolution, not captions.
      </p>
    </div>
  );
}

const BAND_NAMES = ['0-20% (detail)', '20-40%', '40-60%', '60-80%', '80-100% (composition)'];

function CaptionBands({ data }: { data: CaptionProbeData | null }) {
  if (!data || !Object.keys(data.by_band ?? {}).length) return null;
  return (
    <div className="bg-gray-900 rounded-lg p-3 mt-3">
      <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Caption effect by noise level</div>
      <table className="text-[11px] text-gray-300">
        <thead className="text-gray-500">
          <tr>
            <th className="text-left font-normal pr-4">noise</th>
            <th className="text-left font-normal pr-4">caption helps</th>
            <th className="text-left font-normal pr-4">fits its image</th>
            <th className="text-left font-normal">probes</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.by_band).map(([b, v]) => (
            <tr key={b}>
              <td className="pr-4">{BAND_NAMES[Number(b)]}</td>
              <td className="pr-4 font-mono">{(v.gain * 100).toFixed(1)}%</td>
              <td className="pr-4 font-mono">{v.specificity == null ? '—' : `${(v.specificity * 100).toFixed(1)}%`}</td>
              <td className="font-mono text-gray-500">{v.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-gray-500 mt-2">
        The caption should matter most at high noise, where the composition is decided. If it barely matters there
        either, the model is not learning from the captions.
      </p>
    </div>
  );
}

function HealthSection({
  series,
  space,
  resumes,
  memorized,
  jobID,
  captionProbe,
  events,
}: {
  series: Series;
  space?: string;
  resumes: Resume[];
  memorized: Memorized | null;
  jobID: string;
  captionProbe: CaptionProbeData | null;
  events: InsightEvent[];
}) {
  const verdicts = CARDS.map(c => ({ c, v: c.judge(series, space) }));
  const problems = verdicts.filter(({ v }) => v.status === 'bad' || v.status === 'warn');
  const resumeIssues = resumeProblems(resumes);
  const marks = resumes.map(r => r.step);
  const hasAny = Object.entries(series).some(([k, p]) => k.startsWith('insights/') && p.length > 0);
  const anyBad = problems.some(p => p.v.status === 'bad') || resumeIssues.some(r => r.status === 'bad');
  const recentEvents = events.filter(e => e.time > Date.now() / 1000 - 6 * 3600);
  const anyIssue = problems.length > 0 || resumeIssues.length > 0 || recentEvents.length > 0;
  return (
    <div className="space-y-3">
      <div
        className={classNames(
          'rounded-lg px-4 py-2 text-sm',
          !hasAny
            ? 'bg-gray-900 text-gray-400'
            : anyBad
              ? 'bg-red-950/60 text-red-200 border border-red-800'
              : anyIssue
                ? 'bg-yellow-950/50 text-yellow-100 border border-yellow-800'
                : 'bg-green-950/40 text-green-200 border border-green-900',
        )}
      >
        {!hasAny
          ? 'No insights logged yet. They come from extensions/training_insights and start after the first few steps.'
          : !anyIssue
            ? 'Nothing looks off right now.'
            : problems.length > 0 && `Watch: ${problems.map(p => p.c.title).join(', ')}`}
        {events.slice(-3).map(e => (
          <div key={`${e.step}-${e.time}`} className="mt-1">
            Step {e.step}: {e.reasons.join('; ')}
            {e.action !== 'none' && <span className="opacity-70"> → {e.action}</span>}
          </div>
        ))}
        {resumeIssues.map((r, i) => (
          <div key={i} className={problems.length || i ? 'mt-1' : ''}>
            {r.text.charAt(0).toUpperCase() + r.text.slice(1)}.
          </div>
        ))}
        {marks.length > 0 && hasAny && (
          <div className="mt-1 text-xs opacity-70">
            Dashed orange lines in the charts below mark where the job was resumed.
          </div>
        )}
      </div>
      {GROUPS.map(group => (
        <div key={group}>
          <h3 className="text-xs uppercase text-gray-500 mb-2">{group}</h3>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {CARDS.filter(c => c.group === group).map(c => (
              <HealthCard key={c.key} def={c} series={series} space={space} marks={marks} />
            ))}
          </div>
          {group === 'Captions' && (
            <>
              <CaptionStatus data={captionProbe} />
              <CaptionBands data={captionProbe} />
            </>
          )}
          {group === 'Dataset fit' && (
            <>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[11px] text-gray-500">
                <span>Noise bands:</span>
                {NOISE_BANDS.map((b, i) => (
                  <span key={b} className="flex items-center gap-1">
                    <span className="inline-block w-3 h-0.5" style={{ background: NOISE_BAND_COLORS[i] }} />
                    {b}%{i === 0 ? ' (detail)' : i === NOISE_BANDS.length - 1 ? ' (composition)' : ''}
                  </span>
                ))}
              </div>
              <MemorizedList memorized={memorized} jobID={jobID} />
              <BucketsPanel jobID={jobID} />
            </>
          )}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// scatter plumbing
// ---------------------------------------------------------------------------

const PW = 640,
  PH = 440,
  PAD = 24;

function makeScale(xs: number[], ys: number[]) {
  let x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  // equal aspect so distances along both axes are comparable
  const span = Math.max(x1 - x0, y1 - y0, 1e-12);
  const cx = (x0 + x1) / 2,
    cy = (y0 + y1) / 2;
  const half = (span / 2) * 1.05;
  const sx = (PW - 2 * PAD) / (2 * half);
  const sy = (PH - 2 * PAD) / (2 * half);
  const s = Math.min(sx, sy);
  return {
    x: (v: number) => PW / 2 + (v - cx) * s,
    y: (v: number) => PH / 2 - (v - cy) * s,
  };
}

interface Hover {
  x: number;
  y: number;
  content: React.ReactNode;
}

function Tooltip({ hover }: { hover: Hover | null }) {
  if (!hover) return null;
  return (
    <div
      className="absolute pointer-events-none z-20 bg-gray-950/95 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 shadow-lg"
      style={{
        left: `${(hover.x / PW) * 100}%`,
        top: `${(hover.y / PH) * 100}%`,
        transform: `translate(${hover.x > PW * 0.6 ? 'calc(-100% - 10px)' : '10px'}, -50%)`,
      }}
    >
      {hover.content}
    </div>
  );
}

// ---------------------------------------------------------------------------
// trajectory
// ---------------------------------------------------------------------------

const AXIS_PAIRS: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 2],
];

function TrajectoryPlot({ trajectory }: { trajectory: Trajectory | null }) {
  const [pair, setPair] = useState(0);
  const [colorBy, setColorBy] = useState<'step' | 'loss'>('step');
  const [hover, setHover] = useState<Hover | null>(null);

  const dims = trajectory?.coords?.[0]?.length ?? 0;
  const [a, b] = AXIS_PAIRS[pair];

  const plot = useMemo(() => {
    if (!trajectory || trajectory.coords.length < 3 || dims <= b) return null;
    const xs = trajectory.coords.map(c => c[a]);
    const ys = trajectory.coords.map(c => c[b]);
    const scale = makeScale(xs, ys);
    const losses = trajectory.window_losses
      .filter((l): l is number => l != null && Number.isFinite(l))
      .sort((p, q) => p - q);
    const lo = losses[Math.floor(losses.length * 0.05)] ?? 0;
    const hi = losses[Math.floor(losses.length * 0.95)] ?? 1;
    const n = trajectory.steps.length;
    const color = (i: number) => {
      if (colorBy === 'loss') {
        const l = trajectory.window_losses[i];
        // high loss = purple, low loss = yellow
        return l == null ? '#6b7280' : progressColor(1 - (l - lo) / Math.max(1e-12, hi - lo));
      }
      return progressColor(i / Math.max(1, n - 1));
    };
    return { xs, ys, scale, color, n };
  }, [trajectory, a, b, dims, colorBy]);

  if (!trajectory || !plot) {
    return (
      <div className="text-sm text-gray-500 py-8 text-center">The trajectory appears after a few dozen steps.</div>
    );
  }

  const { xs, ys, scale, color, n } = plot;
  const ev = trajectory.explained_variance;
  const efficiency = trajectory.path_length > 0 ? trajectory.displacement / trajectory.path_length : 0;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {AXIS_PAIRS.map(([p, q], i) =>
          dims > q ? (
            <button
              key={i}
              onClick={() => setPair(i)}
              className={classNames(
                'px-2 py-0.5 rounded',
                pair === i ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-400',
              )}
            >
              PC{p + 1} × PC{q + 1}
            </button>
          ) : null,
        )}
        <span className="mx-2 text-gray-600">|</span>
        <span className="text-gray-500">color by</span>
        {(['step', 'loss'] as const).map(m => (
          <button
            key={m}
            onClick={() => setColorBy(m)}
            className={classNames(
              'px-2 py-0.5 rounded',
              colorBy === m ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-400',
            )}
          >
            {m}
          </button>
        ))}
      </div>
      <div className="relative w-full max-w-3xl">
        <svg
          viewBox={`0 0 ${PW} ${PH}`}
          className="w-full h-auto bg-gray-950 rounded-lg"
          onMouseLeave={() => setHover(null)}
        >
          {xs.slice(1).map((_, i) => (
            <line
              key={`s${i}`}
              x1={scale.x(xs[i])}
              y1={scale.y(ys[i])}
              x2={scale.x(xs[i + 1])}
              y2={scale.y(ys[i + 1])}
              stroke={color(i + 1)}
              strokeWidth={1.5}
              strokeOpacity={0.8}
            />
          ))}
          {xs.map((x, i) => (
            <circle
              key={`p${i}`}
              cx={scale.x(x)}
              cy={scale.y(ys[i])}
              r={3}
              fill={color(i)}
              onMouseEnter={() =>
                setHover({
                  x: scale.x(x),
                  y: scale.y(ys[i]),
                  content: (
                    <div>
                      <div>step {trajectory.steps[i]}</div>
                      <div className="text-gray-400">window loss {fmt(trajectory.window_losses[i])}</div>
                    </div>
                  ),
                })
              }
            />
          ))}
          <circle cx={scale.x(xs[0])} cy={scale.y(ys[0])} r={7} fill="none" stroke="#e5e7eb" strokeWidth={1.5} />
          <text x={scale.x(xs[0]) + 10} y={scale.y(ys[0]) + 4} className="fill-gray-300 text-[11px]">
            {trajectory.steps[0] === 0 ? 'start' : `first tracked · step ${trajectory.steps[0]}`}
          </text>
          {(trajectory.resumes ?? []).map(r => {
            const i = trajectory.steps.findIndex(st => st >= r.step);
            if (i <= 0) return null;
            const cx = scale.x(xs[i]),
              cy = scale.y(ys[i]);
            const c = r.optimizer_restored ? '#f59e0b' : '#ef4444';
            return (
              <g
                key={r.step}
                onMouseEnter={() =>
                  setHover({
                    x: cx,
                    y: cy,
                    content: (
                      <div>
                        <div>resumed at step {r.step}</div>
                        <div className="text-gray-400">
                          jump on load {fmt(r.jump)} ({fmt(r.typical_step > 0 ? r.jump / r.typical_step : 0)}× a normal
                          step)
                        </div>
                        <div className={r.optimizer_restored ? 'text-gray-400' : 'text-red-300'}>
                          optimizer state {r.optimizer_restored ? 'restored' : 'NOT restored'}
                        </div>
                      </div>
                    ),
                  })
                }
              >
                <rect
                  x={cx - 6}
                  y={cy - 6}
                  width={12}
                  height={12}
                  transform={`rotate(45 ${cx} ${cy})`}
                  fill="none"
                  stroke={c}
                  strokeWidth={2}
                />
                <text x={cx + 10} y={cy - 8} fill={c} className="text-[11px]">
                  resumed
                </text>
              </g>
            );
          })}
          <circle cx={scale.x(xs[n - 1])} cy={scale.y(ys[n - 1])} r={7} fill="none" stroke="#fde047" strokeWidth={2} />
          <text x={scale.x(xs[n - 1]) + 10} y={scale.y(ys[n - 1]) + 4} className="fill-yellow-200 text-[11px]">
            step {trajectory.steps[n - 1]}
          </text>
        </svg>
        <Tooltip hover={hover} />
      </div>
      <div className="text-xs text-gray-400 flex flex-wrap gap-x-6 gap-y-1">
        <span>variance explained: {ev.map((v, i) => `PC${i + 1} ${(v * 100).toFixed(0)}%`).join(' · ')}</span>
        <span>path length {fmt(trajectory.path_length)}</span>
        <span>net distance {fmt(trajectory.displacement)}</span>
        <span>efficiency {fmt(efficiency)}</span>
        <span>
          {trajectory.sample_dim.toLocaleString()} of {trajectory.total_params.toLocaleString()} params tracked
        </span>
      </div>
      <p className="text-xs text-gray-500 max-w-3xl">
        Axes are fitted on {trajectory.anchor_steps.length} evenly spaced snapshots. A healthy run usually moves in a
        smooth arc that slows down as the loss settles. Look out for sudden jumps (a bad batch or LR spike), tight
        zig-zags (LR too high), or the path still speeding up late in the run (overfitting/overcooking). Hover a point
        to see its step.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// embedding map
// ---------------------------------------------------------------------------

// easy (blue) -> typical (grey) -> hard (orange), for noise-adjusted difficulty
function difficultyColor(v: number | undefined) {
  if (v == null) return '#4b5563';
  if (v >= 1) return `hsl(28 ${Math.min(95, (v - 1) * 250)}% ${62 - Math.min(10, (v - 1) * 25)}%)`;
  return `hsl(200 ${Math.min(95, (1 - v) * 250)}% 62%)`;
}

function EmbeddingMapPlot({ data, jobID }: { data: EmbeddingMapData | null; jobID: string }) {
  const steps = useMemo(() => [...new Set((data?.samples ?? []).map(s => s.step))].sort((p, q) => p - q), [data]);
  const [stepIdx, setStepIdx] = useState<number | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [byDifficulty, setByDifficulty] = useState(true);
  // noise-adjusted difficulty per training image (live_learning_monitor)
  const [difficulty, setDifficulty] = useState<Record<string, number>>({});
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${jobID}/file_loss`, { params: { adjusted: 1 } })
        .then(res => {
          const out: Record<string, number> = {};
          for (const r of res.data?.rows ?? []) if (r.difficulty != null) out[r.file_path] = r.difficulty;
          setDifficulty(out);
        })
        .catch(() => {}),
    30000,
    [jobID],
  );
  const dupIdx = useMemo(() => {
    const set = new Set<number>();
    for (const [i, j] of data?.duplicates ?? []) {
      set.add(i);
      set.add(j);
    }
    return set;
  }, [data]);
  const hasDifficulty = Object.keys(difficulty).length > 0;

  const plot = useMemo(() => {
    if (!data || !data.samples.length) return null;
    const xs = [...data.dataset.map(d => d.x), ...data.samples.map(s => s.x)];
    const ys = [...data.dataset.map(d => d.y), ...data.samples.map(s => s.y)];
    const byPrompt = new Map<number, MapSample[]>();
    for (const s of data.samples) {
      if (!byPrompt.has(s.prompt_idx)) byPrompt.set(s.prompt_idx, []);
      byPrompt.get(s.prompt_idx)!.push(s);
    }
    return { scale: makeScale(xs, ys), byPrompt };
  }, [data]);

  if (!data || !plot || !steps.length) {
    return (
      <div className="text-sm text-gray-500 py-8 text-center">
        The map appears after the first sample step (image samples only).
      </div>
    );
  }

  const idx = stepIdx == null || stepIdx >= steps.length ? steps.length - 1 : stepIdx;
  const step = steps[idx];
  const { scale, byPrompt } = plot;
  const current = data.samples.filter(s => s.step === step);
  const pc = (i: number) => PROMPT_COLORS[i % PROMPT_COLORS.length];
  const [warnSim, badSim] = copyThresholds(data.space);

  const sampleTip = (s: MapSample) => (
    <div className="space-y-1">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imgSrc(s.path)} alt="" className="w-32 h-32 object-contain bg-black rounded" />
      <div>
        prompt {s.prompt_idx + 1} · step {s.step}
      </div>
      {s.nearest_similarity != null && (
        <div className="text-gray-400">closest training image {fmt(s.nearest_similarity)}</div>
      )}
    </div>
  );

  return (
    <div className="space-y-3">
      {hasDifficulty && (
        <label className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer select-none w-fit">
          <input
            type="checkbox"
            checked={byDifficulty}
            onChange={e => setByDifficulty(e.target.checked)}
            className="accent-blue-500"
          />
          Colour training images by difficulty (blue easy · orange hard)
        </label>
      )}
      <div className="flex items-center gap-3 text-xs max-w-3xl">
        <span className="text-gray-400 shrink-0">step {step}</span>
        <input
          type="range"
          min={0}
          max={steps.length - 1}
          value={idx}
          onChange={e => setStepIdx(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-gray-500 shrink-0">
          {idx + 1}/{steps.length}
        </span>
      </div>
      <div className="relative w-full max-w-3xl">
        <svg
          viewBox={`0 0 ${PW} ${PH}`}
          className="w-full h-auto bg-gray-950 rounded-lg"
          onMouseLeave={() => setHover(null)}
        >
          {data.dataset.map((d, i) => (
            <circle
              key={`d${i}`}
              cx={scale.x(d.x)}
              cy={scale.y(d.y)}
              r={byDifficulty && hasDifficulty ? 4.5 : 4}
              fill={byDifficulty && hasDifficulty ? difficultyColor(difficulty[d.path]) : '#9ca3af'}
              fillOpacity={byDifficulty && hasDifficulty ? 0.85 : 0.35}
              stroke={dupIdx.has(i) ? '#facc15' : '#9ca3af'}
              strokeOpacity={dupIdx.has(i) ? 1 : 0.6}
              className="cursor-pointer"
              onClick={() => openImageProfile(jobID, d.path)}
              onMouseEnter={() =>
                setHover({
                  x: scale.x(d.x),
                  y: scale.y(d.y),
                  content: (
                    <div className="space-y-1">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={imgSrc(d.path)} alt="" className="w-32 h-32 object-contain bg-black rounded" />
                      <div className="max-w-[8rem] truncate">{basename(d.path)}</div>
                      {difficulty[d.path] != null && (
                        <div className="text-gray-400">difficulty {difficulty[d.path].toFixed(2)}</div>
                      )}
                      {dupIdx.has(i) && <div className="text-yellow-300">has a near-duplicate</div>}
                      <div className="text-gray-500">click for its profile</div>
                    </div>
                  ),
                })
              }
            />
          ))}
          {[...byPrompt.entries()].map(([p, pts]) => {
            const shown = pts.filter(s => s.step <= step);
            return (
              <g key={`p${p}`}>
                <polyline
                  points={shown.map(s => `${scale.x(s.x)},${scale.y(s.y)}`).join(' ')}
                  fill="none"
                  stroke={pc(p)}
                  strokeOpacity={0.5}
                  strokeWidth={1.5}
                />
                {shown.map(s => (
                  <circle
                    key={s.step}
                    cx={scale.x(s.x)}
                    cy={scale.y(s.y)}
                    r={s.step === step ? 6 : 2.5}
                    fill={pc(p)}
                    stroke={s.step === step ? '#fff' : 'none'}
                    strokeWidth={1.5}
                    onMouseEnter={() => setHover({ x: scale.x(s.x), y: scale.y(s.y), content: sampleTip(s) })}
                  />
                ))}
              </g>
            );
          })}
        </svg>
        <Tooltip hover={hover} />
      </div>
      <p className="text-xs text-gray-500 max-w-3xl">
        Grey = training images, colored = one line per sample prompt, from its first sample to the selected step. Axes
        are fitted once on the dataset and the first samples, in{' '}
        {data.space === 'clip'
          ? `CLIP space (${data.model}): positions reflect content.`
          : "the model's own VAE latent space (the dataset's cached latents): positions reflect layout and colour."}{' '}
        Samples should move toward the grey cloud and stay spread out. Lines that all converge to one spot mean
        collapse. A sample sitting on top of a single grey point is probably a copy of that image.
      </p>
      <div className="overflow-x-auto">
        <table className="text-xs text-gray-300 min-w-[36rem]">
          <thead className="text-gray-500">
            <tr>
              <th className="text-left font-normal pr-3 pb-1">prompt</th>
              <th className="text-left font-normal pr-3 pb-1">sample @ {step}</th>
              <th className="text-left font-normal pr-3 pb-1">closest training image</th>
              <th className="text-left font-normal pr-3 pb-1">similarity</th>
              <th className="text-left font-normal pb-1">drift from base</th>
            </tr>
          </thead>
          <tbody>
            {current.map(s => {
              const nearest = s.nearest_dataset_idx != null ? data.dataset[s.nearest_dataset_idx] : null;
              const sim = s.nearest_similarity;
              return (
                <tr key={s.prompt_idx} className="border-t border-gray-800">
                  <td className="pr-3 py-1">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 align-middle"
                      style={{ background: pc(s.prompt_idx) }}
                    />
                    {s.prompt_idx + 1}
                  </td>
                  <td className="pr-3 py-1">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imgSrc(s.path)} alt="" className="w-16 h-16 object-contain bg-black rounded" />
                  </td>
                  <td className="pr-3 py-1">
                    {nearest ? (
                      <div className="flex items-center gap-2">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <button
                          type="button"
                          onClick={() => openImageProfile(jobID, nearest.path)}
                          title="Open image profile"
                        >
                          <img
                            src={imgSrc(nearest.path)}
                            alt=""
                            className="w-16 h-16 object-contain bg-black rounded"
                          />
                        </button>
                        <span className="max-w-[10rem] truncate text-gray-500" title={nearest.path}>
                          {basename(nearest.path)}
                        </span>
                      </div>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td
                    className={classNames(
                      'pr-3 py-1 font-mono',
                      sim != null && sim > badSim
                        ? 'text-red-400'
                        : sim != null && sim > warnSim
                          ? 'text-yellow-300'
                          : '',
                    )}
                  >
                    {fmt(sim)}
                  </td>
                  <td className="py-1 font-mono">{fmt(s.base_drift)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

// evidence for each side of "is it the adapter or the data?"
function capacityEvidence(s: Series) {
  const captured = latest(s, 'insights/capacity/grad_captured');
  const headroom = latest(s, 'insights/capacity/headroom_4x');
  const satP90 = latest(s, 'insights/capacity/saturation_p90');
  const agreement = latest(s, 'insights/capacity/sample_agreement');
  const spread = vals(s, 'insights/file_loss_spread');
  const memorized = latest(s, 'insights/memorized_fraction');
  const capacity: string[] = [];
  const data: string[] = [];
  const fine: string[] = [];
  const signalCapture = capturedSignal(s);
  const unclear: string[] = [];
  if (captured != null && signalCapture.value != null) {
    const c = signalCapture.value;
    // low capture is normal (most of the gradient is image-specific detail you
    // don't want learned); it only points at size when more size would help a lot
    if (c < 0.35 && headroom != null && headroom > 2)
      capacity.push(
        `The adapter follows ${(c * 100).toFixed(0)}% of the (noise-corrected) gradient and a bigger one would follow much more — matters only if the samples are still missing the concept.`,
      );
    else if (c >= 0.35)
      fine.push(`The adapter can follow ${(c * 100).toFixed(0)}% of the gradient — its shape is not blocking it.`);
  }
  // the capture curve is measured on the raw (noisy) gradient: only use it when
  // the noise-corrected capture exists, otherwise it just reflects noise
  if (headroom != null && signalCapture.value != null) {
    if (headroom > 2)
      capacity.push(`4× the structure would capture ${headroom.toFixed(1)}× as much of what the loss asks for.`);
    else fine.push(`A 4× bigger adapter would capture only ${headroom.toFixed(1)}× as much.`);
  }
  if (signalCapture.noisy)
    unclear.push(
      'The capacity probe sees mostly per-batch noise, so it cannot tell yet whether the adapter is too small. Raise probe_span (e.g. 16-32 steps).',
    );
  const updateRatio = latest(s, 'insights/update_ratio');
  if (updateRatio != null && updateRatio > 5e-2)
    unclear.push(
      `Each step changes the weights by ~${(updateRatio * 100).toFixed(0)}% — far too large to refine anything. Lower the learning rate (or cap the optimizer's max LR) before judging capacity or data.`,
    );
  if (satP90 != null && satP90 > 0.9) capacity.push('Most modules use nearly all their rank.');
  const bleed = latest(s, 'insights/concept_bleed');
  if (bleed != null && bleed > 0.6)
    data.push(
      `Prompts without the trigger drift ${(bleed * 100).toFixed(0)}% as much as with it: everything the images share is being learned, not just the concept (captions, Differential Output Preservation, regularisation images).`,
    );
  const strength = latest(s, 'insights/concept_strength');
  const drift = latest(s, 'insights/concept_drift');
  if (strength != null && drift != null && strength < 0.3 && drift > 0.3)
    data.push(
      `Samples moved a lot toward the dataset look but only ${(strength * 100).toFixed(0)}% toward the concept: it is copying the dataset more than learning the concept.`,
    );
  if (agreement != null && agreement < -0.02)
    data.push('Gradients of different samples cancel each other out (conflicting images or captions).');
  if (spread.length >= 5 && median(last(spread, 5)) / baseline(spread) > 1.5)
    data.push(
      'Per-file losses keep spreading apart: some images are learned, others are not (check their captions in Live Batch → Difficulty).',
    );
  if (memorized != null && memorized > 0.15)
    data.push(`${(memorized * 100).toFixed(0)}% of images are memorised while the rest lag behind.`);
  return { capacity, data, fine, unclear };
}

function DuplicatesList({ data, jobID }: { data: EmbeddingMapData | null; jobID: string }) {
  const dups = data?.duplicates ?? [];
  if (!data || !dups.length) return null;
  return (
    <div className="mt-4">
      <div className="text-xs uppercase text-gray-500 mb-2">Near-duplicate training images ({dups.length})</div>
      <div className="flex flex-wrap gap-3">
        {dups.slice(0, 24).map(([i, j, sim]) => (
          <div key={`${i}-${j}`} className="flex items-center gap-1 bg-gray-950 rounded p-1">
            {[i, j].map(k => (
              <button
                key={k}
                type="button"
                onClick={() => openImageProfile(jobID, data.dataset[k].path)}
                title={data.dataset[k].path}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={imgSrc(data.dataset[k].path)} alt="" className="w-14 h-14 object-cover rounded" />
              </button>
            ))}
            <span className="text-[10px] font-mono text-yellow-300 px-1">{sim.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        Near-identical pairs (yellow outline on the map). Duplicates are the most common cause of memorisation: they get
        trained twice as often.
      </p>
    </div>
  );
}

function CaptureCurve({ curve, kind }: { curve: [number, number][]; kind: string | null }) {
  const W = 320,
    H = 140,
    P = 28;
  const xs = curve.map((_, i) => P + (i / Math.max(1, curve.length - 1)) * (W - 2 * P));
  const y = (v: number) => H - P + 6 - v * (H - P - 12);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-sm h-auto">
      {[0, 0.5, 1].map(v => (
        <g key={v}>
          <line x1={P} x2={W - P} y1={y(v)} y2={y(v)} stroke="#374151" strokeDasharray="2 3" />
          <text x={P - 4} y={y(v) + 3} textAnchor="end" className="fill-gray-500 text-[9px]">
            {v * 100}%
          </text>
        </g>
      ))}
      <polyline
        points={curve.map(([, v], i) => `${xs[i]},${y(v)}`).join(' ')}
        fill="none"
        stroke="#60a5fa"
        strokeWidth={2}
      />
      {curve.map(([m, v], i) => (
        <g key={m}>
          <circle cx={xs[i]} cy={y(v)} r={3} fill={i === 0 ? '#fbbf24' : '#60a5fa'} />
          <text x={xs[i]} y={H - 6} textAnchor="middle" className="fill-gray-400 text-[9px]">
            {m}×
          </text>
          <text x={xs[i]} y={y(v) - 6} textAnchor="middle" className="fill-gray-300 text-[9px]">
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <text x={W / 2} y={10} textAnchor="middle" className="fill-gray-500 text-[9px]">
        {kind === 'lokr' ? 'sum of k Kronecker products' : 'rank multiples'} → share of gradient
      </text>
    </svg>
  );
}

function Spectrum({ sv }: { sv: number[] }) {
  if (sv.length < 2) return <div className="text-xs text-gray-500">No spectrum yet.</div>;
  const W = 320,
    H = 120,
    P = 8;
  const lo = Math.log10(Math.max(1e-4, Math.min(...sv.filter(v => v > 0))));
  const y = (v: number) => P + ((0 - Math.log10(Math.max(v, 1e-4))) / Math.max(1e-6, 0 - lo)) * (H - 2 * P);
  const bw = (W - 2 * P) / sv.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-sm h-auto bg-gray-950 rounded">
      {sv.map((v, i) => (
        <rect key={i} x={P + i * bw} y={y(v)} width={Math.max(1, bw - 1)} height={H - P - y(v)} fill="#60a5fa" />
      ))}
    </svg>
  );
}

function CapacitySection({ capacity, series, jobID }: { capacity: Capacity | null; series: Series; jobID: string }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'saturation' | 'captured'>('captured');
  if (!capacity || !Object.keys(capacity.modules).length) {
    return (
      <div className="text-sm text-gray-500 py-6 text-center">
        Capacity analysis appears after the first {`~100`} steps (LoRA / LoKr jobs only).
      </div>
    );
  }
  const entries = Object.entries(capacity.modules);
  const ev = capacityEvidence(series);
  const sats = entries.map(([, m]) => m.saturation);
  const bins = new Array(10).fill(0);
  for (const v of sats) bins[Math.min(9, Math.max(0, Math.floor(v * 10)))]++;
  const maxBin = Math.max(...bins, 1);
  const sorted = [...entries].sort((a, b) =>
    sortBy === 'captured' ? (a[1].captured ?? 2) - (b[1].captured ?? 2) : b[1].saturation - a[1].saturation,
  );
  const sel = selected ? capacity.modules[selected] : null;
  const verdict =
    ev.capacity.length > ev.data.length
      ? {
          text: 'Leaning: the adapter is too small or the wrong shape',
          cls: 'border-yellow-800 bg-yellow-950/40 text-yellow-100',
        }
      : ev.data.length > ev.capacity.length
        ? { text: 'Leaning: the data / captions', cls: 'border-orange-800 bg-orange-950/40 text-orange-100' }
        : ev.capacity.length
          ? { text: 'Mixed evidence', cls: 'border-gray-700 bg-gray-950 text-gray-200' }
          : { text: 'No sign of a capacity limit yet', cls: 'border-green-900 bg-green-950/30 text-green-200' };

  return (
    <div className="space-y-4">
      <div className={classNames('rounded-lg border px-4 py-3 text-sm', verdict.cls)}>
        <div className="font-medium mb-2">{verdict.text}</div>
        {ev.unclear.length > 0 && (
          <div className="text-xs mb-3 rounded border border-yellow-800 bg-yellow-950/40 text-yellow-100 px-3 py-2 space-y-1">
            {ev.unclear.map((t, i) => (
              <div key={i}>• {t}</div>
            ))}
          </div>
        )}
        <div className="grid md:grid-cols-3 gap-4 text-xs">
          <div>
            <div className="uppercase text-[10px] opacity-70 mb-1">Points to adapter size</div>
            {ev.capacity.length ? (
              ev.capacity.map((t, i) => <div key={i}>• {t}</div>)
            ) : (
              <div className="opacity-60">—</div>
            )}
          </div>
          <div>
            <div className="uppercase text-[10px] opacity-70 mb-1">Points to the data</div>
            {ev.data.length ? ev.data.map((t, i) => <div key={i}>• {t}</div>) : <div className="opacity-60">—</div>}
          </div>
          <div>
            <div className="uppercase text-[10px] opacity-70 mb-1">Looks fine</div>
            {ev.fine.length ? ev.fine.map((t, i) => <div key={i}>• {t}</div>) : <div className="opacity-60">—</div>}
          </div>
        </div>
        {capacity.kind === 'lokr' && ev.capacity.length > 0 && (
          <div className="text-xs mt-3 opacity-80">
            With LoKr, more capacity means a smaller <code>lokr_factor</code>, but go in small steps: on a 3072-wide
            layer, -1 (48×64) is ~6k parameters, 32 ~10k, 16 ~37k (6×), 8 ~147k (23×). Try 24 or 16 first and only go
            further if this probe still says the adapter can&apos;t follow the gradient.
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <div className="text-xs uppercase text-gray-500 mb-1">
            What a bigger adapter would capture (probe at step {capacity.probe?.step ?? '—'})
          </div>
          {capacity.probe ? (
            <CaptureCurve curve={capacity.probe.capture_curve} kind={capacity.kind} />
          ) : (
            <div className="text-xs text-gray-500">The first gradient probe runs after ~250 steps.</div>
          )}
          <p className="text-[11px] text-gray-500 max-w-sm">
            Share of the gradient each size could express at the probed layers. Yellow = the current size. A steep rise
            means the data asks for more than the adapter can hold.
          </p>
        </div>
        <div>
          <div className="text-xs uppercase text-gray-500 mb-1">
            Rank saturation across {entries.length} modules (step {capacity.spectrum_step ?? '—'})
          </div>
          <svg viewBox="0 0 320 120" className="w-full max-w-sm h-auto">
            {bins.map((b, i) => (
              <g key={i}>
                <rect
                  x={10 + i * 30}
                  y={100 - (b / maxBin) * 85}
                  width={26}
                  height={(b / maxBin) * 85}
                  fill={i >= 9 ? '#f59e0b' : '#60a5fa'}
                />
                <text x={23 + i * 30} y={113} textAnchor="middle" className="fill-gray-500 text-[9px]">
                  {i * 10}
                </text>
              </g>
            ))}
          </svg>
          <p className="text-[11px] text-gray-500 max-w-sm">
            Effective rank of each module&apos;s update as % of the most it can reach. A pile-up at the right (orange) =
            no room left.
          </p>
        </div>
      </div>

      {(() => {
        const files = Object.entries(capacity.files ?? {}).filter(([, f]) => f.captured != null) as [
          string,
          { captured: number; n: number; probes: number },
        ][];
        if (files.length < 4) return null;
        const med = median(files.map(([, f]) => f.captured));
        const worst = files
          .filter(([, f]) => f.captured < med * 0.6)
          .sort((a, b) => a[1].captured - b[1].captured)
          .slice(0, 16);
        if (!worst.length) return null;
        return (
          <div>
            <div className="text-xs uppercase text-gray-500 mb-2">
              Images the adapter can follow least (median {(med * 100).toFixed(0)}%)
            </div>
            <div className="flex flex-wrap gap-2">
              {worst.map(([path, f]) => (
                <button
                  key={path}
                  type="button"
                  onClick={() => openImageProfile(jobID, path)}
                  className="w-20 text-left text-[10px] text-gray-400"
                  title={path}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imgSrc(path)} alt="" className="w-20 h-20 object-cover rounded bg-black" />
                  <span className="font-mono text-yellow-300">{(f.captured * 100).toFixed(0)}%</span>
                </button>
              ))}
            </div>
            <p className="text-[11px] text-gray-500 mt-1 max-w-3xl">
              Their gradient points where the adapter can&apos;t go while the rest of the dataset is fine: they need
              something the adapter can&apos;t express, or they contradict the other images. Check them first.
            </p>
          </div>
        );
      })()}
      <div className="grid lg:grid-cols-[1fr_auto] gap-4">
        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="text-xs text-gray-300 w-full">
            <thead className="text-gray-500 sticky top-0 bg-gray-900">
              <tr>
                <th className="text-left font-normal pr-3 pb-1">module</th>
                <th className="text-left font-normal pr-3 pb-1 cursor-pointer" onClick={() => setSortBy('saturation')}>
                  saturation{sortBy === 'saturation' ? ' ▼' : ''}
                </th>
                {capacity.kind === 'lokr' && <th className="text-left font-normal pr-3 pb-1">w1 / w2</th>}
                <th className="text-left font-normal pr-3 pb-1 cursor-pointer" onClick={() => setSortBy('captured')}>
                  captured{sortBy === 'captured' ? ' ▲' : ''}
                </th>
                <th className="text-left font-normal pr-3 pb-1">grad rank</th>
                <th className="text-left font-normal pb-1">agreement</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(([name, m]) => (
                <tr
                  key={name}
                  onClick={() => setSelected(name)}
                  className={classNames(
                    'border-t border-gray-800 cursor-pointer hover:bg-gray-800/50',
                    selected === name && 'bg-gray-800',
                  )}
                >
                  <td className="pr-3 py-0.5 max-w-xs truncate" title={name}>
                    {name.replace(/^lora_(unet|transformer)_/, '')}
                  </td>
                  <td className={classNames('pr-3 font-mono', m.saturation > 0.9 && 'text-orange-300')}>
                    {(m.saturation * 100).toFixed(0)}%
                  </td>
                  {capacity.kind === 'lokr' && (
                    <td className="pr-3 font-mono text-gray-400">
                      {m.w1_saturation != null
                        ? `${(m.w1_saturation * 100).toFixed(0)} / ${((m.w2_saturation ?? 0) * 100).toFixed(0)}%`
                        : '—'}
                    </td>
                  )}
                  <td
                    className={classNames(
                      'pr-3 font-mono',
                      m.captured != null && m.captured < 0.35 && 'text-yellow-300',
                    )}
                  >
                    {m.captured != null ? `${(m.captured * 100).toFixed(0)}%` : '—'}
                  </td>
                  <td className="pr-3 font-mono">{m.grad_erank != null ? m.grad_erank.toFixed(0) : '—'}</td>
                  <td className="font-mono">{m.agreement != null ? m.agreement.toFixed(3) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="w-80">
          <div className="text-xs uppercase text-gray-500 mb-1 truncate" title={selected ?? ''}>
            {sel ? `Spectrum: ${selected}` : 'Click a module for its spectrum'}
          </div>
          {sel && (
            <>
              <Spectrum sv={sel.sv} />
              <div className="text-[11px] text-gray-400 mt-1 space-y-0.5">
                <div>
                  effective rank {sel.erank.toFixed(1)} of {sel.rank_cap} · stable rank {sel.stable_rank.toFixed(1)}
                </div>
                {sel.capture_curve && (
                  <div>
                    capture at 1/2/4/8×:{' '}
                    {sel.capture_curve
                      .slice(0, 4)
                      .map(([, v]) => `${(v * 100).toFixed(0)}%`)
                      .join(' / ')}
                  </div>
                )}
                <div className="text-gray-500">
                  Singular values of the update (log scale, top {sel.sv.length}). Flat = many directions in use; a fast
                  drop = a few dominate.
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

interface Suggestion {
  step: number;
  file: string | null;
  reasons: string[];
  limit: number | null;
}

// The last good point by the metrics: before the first sign of trouble, the
// sample step with the most concept for the least bleed (or simply the last
// one when there is no concept data), mapped to the nearest saved checkpoint.
function suggestCheckpoint(
  s: Series,
  events: InsightEvent[],
  saves: { step: number; path: string }[],
): Suggestion | null {
  if (!saves.length) return null;
  const firstAbove = (key: string, test: (v: number) => boolean) =>
    (s[key] ?? []).find(p => test(p.value))?.step ?? null;
  const trouble: [number | null, string][] = [
    [events.length ? Math.min(...events.map(e => e.step)) : null, 'the first warning'],
    [firstAbove('insights/update_ratio', v => v > 0.05), 'steps grew over 5% of the weights'],
    [firstAbove('insights/concept_bleed', v => v > 0.6), 'concept bleed passed 60%'],
  ];
  const div = s['insights/sample_diversity'] ?? [];
  if (div.length > 1)
    trouble.push([div.find(p => p.value < 0.5 * div[0].value)?.step ?? null, 'sample variety halved']);
  const hits = trouble.filter(([st]) => st != null) as [number, string][];
  const limitHit = hits.length ? hits.reduce((a, b) => (b[0] < a[0] ? b : a)) : null;
  const limit = limitHit ? limitHit[0] : null;

  const reasons: string[] = [];
  const strength = (s['insights/concept_strength'] ?? []).filter(p => limit == null || p.step <= limit);
  let target: number;
  if (strength.length) {
    const bleed = s['insights/concept_bleed'] ?? [];
    const bleedAt = (step: number) => bleed.filter(p => p.step <= step).slice(-1)[0]?.value ?? 0;
    const best = strength.reduce((a, b) =>
      b.value - Math.max(0, bleedAt(b.step) - 0.3) > a.value - Math.max(0, bleedAt(a.step) - 0.3) ? b : a,
    );
    target = best.step;
    reasons.push(
      `most concept for the least bleed (strength ${(best.value * 100).toFixed(0)}%, bleed ${(bleedAt(best.step) * 100).toFixed(0)}%)`,
    );
  } else {
    target = limit ?? Math.max(...saves.map(x => x.step));
    reasons.push(limit != null ? 'the last point before trouble started' : 'no sign of trouble yet: the latest save');
  }
  if (limitHit) reasons.push(`trouble from step ${limitHit[0]}: ${limitHit[1]}`);
  const before = saves.filter(x => x.step <= target);
  const pick = before.length ? before[before.length - 1] : saves[0];
  return { step: pick.step, file: pick.path, reasons, limit };
}

// Per-image training report for MediaTagger: download it, or push it straight
// into a running MediaTagger (its MCP server) where the flags become a filter.
function TrainingReportBar({ jobID }: { jobID: string }) {
  const [status, setStatus] = useState<string | null>(null);
  // the direct send needs MediaTagger running here with MEDIATAGGER_MCP_URL set
  const [bridge, setBridge] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/mediatagger')
      .then(res => setBridge(!!res.data?.enabled))
      .catch(() => setBridge(false));
  }, []);
  const send = () => {
    setStatus('Sending…');
    apiClient
      .post('/api/mediatagger', { action: 'import', jobID })
      .then(res =>
        setStatus(
          `MediaTagger: ${res.data?.matched ?? 0} images matched, ${res.data?.flagged ?? 0} flagged` +
            (res.data?.unmatched ? `, ${res.data.unmatched} not in its open dataset` : ''),
        ),
      )
      .catch(err => setStatus(err?.response?.data?.error ?? 'MediaTagger is not reachable'));
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
      <span title="Also written by the extension to insights/training_report.json in the job folder, which MediaTagger (Dataset → Import training report…) can read directly.">
        Per-image training report (hard / memorised / duplicate / weak caption):
      </span>
      <a
        href={`/api/jobs/${jobID}/dataset_report`}
        className="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-200"
      >
        Download
      </a>
      {bridge && (
        <button
          type="button"
          onClick={send}
          className="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-200"
          title="Send to the running MediaTagger (MCP on, with this dataset open). Flags are kept apart from tags and never written to caption files."
        >
          Send to MediaTagger
        </button>
      )}
      {status && <span>{status}</span>}
    </div>
  );
}

function SuggestedCheckpoint({ jobID, series, events }: { jobID: string; series: Series; events: InsightEvent[] }) {
  const [saves, setSaves] = useState<{ step: number; path: string }[]>([]);
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${jobID}/files`)
        .then(res => {
          const out: { step: number; path: string }[] = [];
          for (const f of res.data?.files ?? []) {
            const m = /_(\d{9})\.safetensors$/.exec(f.path);
            if (m) out.push({ step: Number(m[1]), path: f.path });
          }
          setSaves(out.sort((a, b) => a.step - b.step));
        })
        .catch(() => {}),
    60000,
    [jobID],
  );
  const sug = suggestCheckpoint(series, events, saves);
  if (!sug) return null;
  return (
    <div className="rounded-lg border border-blue-900 bg-blue-950/30 px-4 py-3 text-sm text-blue-100">
      <div>
        Suggested checkpoint: <span className="font-mono">{basename(sug.file ?? '')}</span>{' '}
        <span className="opacity-70">(step {sug.step})</span>
      </div>
      <div className="text-xs opacity-80 mt-1">
        {sug.reasons.join(' · ')}. A suggestion from the metrics: check its samples before using it.
      </div>
    </div>
  );
}

export default function JobInsights({ job }: Props) {
  const series = useInsightMetrics(job.id);
  const files = useInsightFiles(job.id);

  return (
    <div className="space-y-6 pb-8">
      <HealthSection
        series={series}
        space={files?.embeddingMap?.space}
        resumes={files?.trajectory?.resumes ?? []}
        memorized={files?.memorized ?? null}
        jobID={job.id}
        captionProbe={files?.captionProbe ?? null}
        events={files?.events?.events ?? []}
      />
      <SuggestedCheckpoint jobID={job.id} series={series} events={files?.events?.events ?? []} />
      <TrainingReportBar jobID={job.id} />
      <section className="bg-gray-900 rounded-lg p-4 space-y-2">
        <h2 className="text-lg font-semibold uppercase text-gray-500">Capacity or data?</h2>
        <CapacitySection capacity={files?.capacity ?? null} series={series} jobID={job.id} />
      </section>
      <section className="bg-gray-900 rounded-lg p-4 space-y-2">
        <h2 className="text-lg font-semibold uppercase text-gray-500">Weight trajectory</h2>
        <TrajectoryPlot trajectory={files?.trajectory ?? null} />
      </section>
      <section className="bg-gray-900 rounded-lg p-4 space-y-2">
        <h2 className="text-lg font-semibold uppercase text-gray-500">Samples vs. dataset</h2>
        <EmbeddingMapPlot data={files?.embeddingMap ?? null} jobID={job.id} />
        <DuplicatesList data={files?.embeddingMap ?? null} jobID={job.id} />
      </section>
    </div>
  );
}
