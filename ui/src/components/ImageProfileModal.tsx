'use client';

import { useEffect, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { LuLoader, LuX } from 'react-icons/lu';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import LossSparkline from './LossSparkline';

// One panel for everything known about a training image, opened from any view
// (per-file table, live batch, insights map / memorised list, nearest images).
// Data: /api/jobs/[jobID]/file_profile, which merges live_learning_monitor's
// per-file losses with training_insights' map, capacity and caption probes.

interface ProfileTarget {
  jobID: string;
  path: string;
}

interface Profile {
  file_path: string;
  caption: string | null;
  times_trained: number;
  difficulty: number | null;
  bands: (number | null)[];
  band_counts: number[];
  buckets: string[];
  history: { step: number; loss: number | null; rel_loss: number | null; timestep: number | null }[];
  nearest_samples: { path: string; step: number; prompt_idx: number; similarity: number }[];
  duplicates: { path: string; similarity: number }[];
  capacity: { captured: number | null; n: number; probes: number; step?: number } | null;
  caption_probe: { n: number; gain?: number; specificity?: number; trigger_gain?: number } | null;
  memorized: { rel_loss: number; vs_median: number } | null;
}

const profileState = createGlobalState<ProfileTarget | null>(null);

export const openImageProfile = (jobID: string, path: string) => profileState.set({ jobID, path });

const BAND_LABELS = ['0-20', '20-40', '40-60', '60-80', '80-100'];

function basename(p: string) {
  return p.replace(/^.*[\\/]/, '');
}

function imgSrc(p: string, thumb = false) {
  return `/api/img/${encodeURIComponent(p)}${thumb ? '?thumb=1' : ''}`;
}

function pct(v: number | null | undefined, digits = 0) {
  return v == null ? '—' : `${(v * 100).toFixed(digits)}%`;
}

// what the shape of the per-noise-level difficulty suggests
function bandHint(bands: (number | null)[]): string | null {
  const v = bands.map(b => b ?? NaN);
  const low = [v[0], v[1]].filter(Number.isFinite);
  const high = [v[3], v[4]].filter(Number.isFinite);
  if (!low.length || !high.length) return null;
  const lo = low.reduce((a, b) => a + b, 0) / low.length;
  const hi = high.reduce((a, b) => a + b, 0) / high.length;
  if (hi > 1.25 && hi > lo + 0.15)
    return 'Hard mainly at high noise, where composition is decided and the caption matters most: check that the caption describes this image (and what makes it different).';
  if (lo > 1.25 && lo > hi + 0.15)
    return 'Hard mainly at low noise (fine detail): usually image quality, compression, noise or resolution rather than the caption.';
  if (hi < 0.75 && lo < 0.75) return 'Easy at every noise level: likely memorised (duplicates or too many repeats).';
  return null;
}

function Bands({ bands, counts }: { bands: (number | null)[]; counts: number[] }) {
  const max = Math.max(1.6, ...bands.map(b => b ?? 0));
  const W = 260,
    H = 90,
    P = 14;
  const bw = (W - 2 * P) / bands.length;
  const y = (v: number) => H - P - (v / max) * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-xs h-auto">
      <line x1={P} x2={W - P} y1={y(1)} y2={y(1)} stroke="#9ca3af" strokeDasharray="3 3" />
      <text x={W - P} y={y(1) - 3} textAnchor="end" className="fill-gray-500 text-[8px]">
        typical
      </text>
      {bands.map((b, i) => (
        <g key={i}>
          {b != null && (
            <rect
              x={P + i * bw + 3}
              y={y(b)}
              width={bw - 6}
              height={H - P - y(b)}
              fill={b > 1.3 ? '#fb923c' : b < 0.6 ? '#38bdf8' : '#60a5fa'}
            >
              <title>
                {b.toFixed(2)} ({counts[i]} samples)
              </title>
            </rect>
          )}
          <text x={P + i * bw + bw / 2} y={H - 3} textAnchor="middle" className="fill-gray-500 text-[8px]">
            {BAND_LABELS[i]}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'warn' | 'good' | 'info' }) {
  return (
    <div className="bg-gray-950 rounded px-2.5 py-1.5">
      <div className="text-[10px] uppercase text-gray-500">{label}</div>
      <div
        className={classNames(
          'font-mono text-sm',
          tone === 'warn'
            ? 'text-orange-300'
            : tone === 'good'
              ? 'text-green-300'
              : tone === 'info'
                ? 'text-sky-300'
                : 'text-gray-100',
        )}
      >
        {value}
      </div>
    </div>
  );
}

export default function ImageProfileModal() {
  const [target, setTarget] = profileState.use();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!target) return;
    setProfile(null);
    setError(false);
    apiClient
      .get(`/api/jobs/${target.jobID}/file_profile`, { params: { file: target.path } })
      .then(res => setProfile(res.data))
      .catch(() => setError(true));
  }, [target]);

  const close = () => setTarget(null);
  const [mtStatus, setMtStatus] = useState<string | null>(null);
  // only with the optional MediaTagger bridge (MEDIATAGGER_MCP_URL) configured
  const [bridge, setBridge] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/mediatagger')
      .then(res => setBridge(!!res.data?.enabled))
      .catch(() => setBridge(false));
  }, []);
  useEffect(() => setMtStatus(null), [target]);
  const showInMediaTagger = () => {
    if (!target) return;
    setMtStatus('Sending…');
    apiClient
      .post('/api/mediatagger', { action: 'show', path: target.path })
      .then(res =>
        setMtStatus(res.data?.shown ? 'Shown in MediaTagger' : (res.data?.note ?? 'MediaTagger did not show it')),
      )
      .catch(err => setMtStatus(err?.response?.data?.error ?? 'MediaTagger is not reachable'));
  };
  const p = profile;
  const hint = p ? bandHint(p.bands) : null;
  const cp = p?.caption_probe;

  return (
    <Dialog open={!!target} onClose={close} className="relative z-50">
      <DialogBackdrop className="fixed inset-0 bg-black/70" />
      <div className="fixed inset-0 overflow-y-auto p-2 sm:p-6">
        <DialogPanel className="mx-auto max-w-5xl bg-gray-900 border border-gray-800 rounded-xl shadow-xl text-gray-200">
          <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800">
            <div className="text-sm font-medium truncate flex-1" title={target?.path}>
              {target ? basename(target.path) : ''}
            </div>
            {mtStatus && <span className="text-xs text-gray-400 truncate max-w-xs">{mtStatus}</span>}
            {bridge && (
              <button
                onClick={showInMediaTagger}
                className="px-2 py-1 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700"
                title="Select this image in the MediaTagger window (needs MediaTagger running with MCP on)"
              >
                Show in MediaTagger
              </button>
            )}
            <button onClick={close} className="p-1 rounded hover:bg-gray-800" aria-label="Close">
              <LuX className="w-4 h-4" />
            </button>
          </div>
          {!p && !error && (
            <div className="flex items-center gap-2 text-sm text-gray-500 p-8">
              <LuLoader className="w-4 h-4 animate-spin" /> Loading...
            </div>
          )}
          {error && <div className="text-sm text-red-400 p-8">Could not load this image&apos;s profile.</div>}
          {p && target && (
            <div className="p-4 grid md:grid-cols-[minmax(0,320px)_1fr] gap-5">
              <div className="space-y-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imgSrc(p.file_path)}
                  alt=""
                  className="w-full max-h-[360px] object-contain bg-black rounded"
                />
                <div className="text-xs text-gray-300 whitespace-pre-wrap break-words bg-gray-950 rounded p-2 max-h-40 overflow-y-auto">
                  {p.caption ?? <span className="italic text-gray-500">no caption file found</span>}
                </div>
                <div className="text-[11px] text-gray-500 break-all">{p.file_path}</div>
              </div>

              <div className="space-y-4 min-w-0">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <Stat
                    label="Difficulty"
                    value={p.difficulty == null ? '—' : p.difficulty.toFixed(2)}
                    tone={
                      p.difficulty == null
                        ? undefined
                        : p.difficulty > 1.3
                          ? 'warn'
                          : p.difficulty < 0.6
                            ? 'info'
                            : undefined
                    }
                  />
                  <Stat label="Times trained" value={p.times_trained.toLocaleString()} />
                  <Stat
                    label="Adapter can follow"
                    value={pct(p.capacity?.captured)}
                    tone={p.capacity?.captured != null && p.capacity.captured < 0.35 ? 'warn' : undefined}
                  />
                  <Stat label="Resolution" value={p.buckets.length ? p.buckets.join(', ') : '—'} />
                </div>
                {p.memorized && (
                  <div className="text-xs rounded bg-sky-950/50 border border-sky-900 text-sky-200 px-3 py-2">
                    Memorised: its noise-adjusted loss is {(p.memorized.vs_median * 100).toFixed(0)}% of the dataset
                    median.
                  </div>
                )}

                <div>
                  <div className="text-xs uppercase text-gray-500 mb-1">Difficulty by noise level</div>
                  {p.bands.some(b => b != null) ? (
                    <Bands bands={p.bands} counts={p.band_counts} />
                  ) : (
                    <div className="text-xs text-gray-500">Not enough samples with timesteps yet.</div>
                  )}
                  {hint && <div className="text-xs text-gray-300 mt-1">{hint}</div>}
                </div>

                <div>
                  <div className="text-xs uppercase text-gray-500 mb-1">Noise-adjusted loss over steps</div>
                  <LossSparkline
                    points={p.history.filter(h => h.rel_loss != null).map(h => ({ step: h.step, loss: h.rel_loss }))}
                    width={480}
                    height={80}
                  />
                </div>

                <div>
                  <div className="text-xs uppercase text-gray-500 mb-1">Caption probe ({cp?.n ?? 0} probes)</div>
                  {cp && cp.n > 0 ? (
                    <div className="grid grid-cols-3 gap-2">
                      <Stat
                        label="Caption helps"
                        value={pct(cp.gain, 1)}
                        tone={cp.gain != null && cp.gain < 0.01 ? 'warn' : 'good'}
                      />
                      <Stat
                        label="Fits this image"
                        value={pct(cp.specificity, 1)}
                        tone={cp.specificity != null && cp.specificity < 0.005 ? 'warn' : 'good'}
                      />
                      <Stat label="Trigger word" value={pct(cp.trigger_gain, 1)} />
                    </div>
                  ) : (
                    <div className="text-xs text-gray-500">Not probed yet (one batch every 25 steps).</div>
                  )}
                  <div className="text-[11px] text-gray-500 mt-1">
                    How much lower the loss is with the real caption than with a blank one / another image&apos;s
                    caption / no trigger word. Near 0 = the caption doesn&apos;t help the model with this image.
                  </div>
                </div>

                {p.duplicates.length > 0 && (
                  <div>
                    <div className="text-xs uppercase text-gray-500 mb-1">Near-duplicates in the dataset</div>
                    <div className="flex flex-wrap gap-2">
                      {p.duplicates.map(d => (
                        <button
                          key={d.path}
                          onClick={() => openImageProfile(target.jobID, d.path)}
                          className="w-20 text-left text-[10px] text-gray-400"
                          title={d.path}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={imgSrc(d.path, true)} alt="" className="w-20 h-20 object-cover rounded bg-black" />
                          {d.similarity.toFixed(3)}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {p.nearest_samples.length > 0 && (
                  <div>
                    <div className="text-xs uppercase text-gray-500 mb-1">
                      Samples that came out closest to this image
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {p.nearest_samples.map(s => (
                        <div key={s.path} className="w-20 text-[10px] text-gray-400" title={s.path}>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={imgSrc(s.path, true)} alt="" className="w-20 h-20 object-cover rounded bg-black" />
                          step {s.step} · {s.similarity.toFixed(2)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogPanel>
      </div>
    </Dialog>
  );
}
