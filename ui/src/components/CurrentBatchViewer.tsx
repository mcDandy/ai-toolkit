'use client';

import { useState, useEffect } from 'react';
import { Job } from '@prisma/client';
import useCurrentBatch, { CurrentBatchItem } from '@/hooks/useCurrentBatch';
import { useStepHistory, useStepHistoryRange, StepHistoryItem } from '@/hooks/useStepHistory';
import { apiClient } from '@/utils/api';
import { LuHistory, LuImageOff, LuLoader, LuRadio } from 'react-icons/lu';
import LossSparkline, { LossSparklinePoint } from './LossSparkline';
import { openImageProfile } from './ImageProfileModal';
import usePollLoop from '@/hooks/usePollLoop';

const NOISE_BANDS = ['low noise (detail)', 'low-mid noise', 'mid noise', 'mid-high noise', 'high noise (composition)'];

function noiseBand(t: number) {
  return NOISE_BANDS[Math.min(4, Math.max(0, Math.floor((t / 1000) * 5)))];
}

interface Props {
  job: Job;
}

type SeriesState = { status: 'loading' } | { status: 'error' } | { status: 'success'; points: LossSparklinePoint[] };

function basename(p: string) {
  return p.replace(/^.*[\\/]/, '');
}

function fmtMetric(v: number | string): string {
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2);
  return a < 10 ? v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : v.toFixed(2);
}

// most interesting-first ordering for the little chips above a past step
function metricEntries(metrics: Record<string, number | string>): [string, string][] {
  const rank = (k: string) =>
    k === 'learning_rate' ? 0 : k.startsWith('loss') ? 1 : k.startsWith('insights/') ? 2 : 3;
  return Object.entries(metrics)
    .sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]))
    .slice(0, 16)
    .map(([k, v]) => [k, fmtMetric(v)] as [string, string]);
}

// Shared thumbnail + name block for both the live and the history row.
function ItemThumb({ jobID, path, memorized }: { jobID: string; path: string; memorized: boolean }) {
  // Not ?thumb=1 on purpose: the pre-generated thumbnail is a sharp() resize
  // with `fit: 'cover'`, which center-crops to a square before it's ever
  // saved to disk — object-contain below can't recover image data that's
  // already gone from the file. Only a handful of images are on screen here
  // at once (the current batch), so loading the originals is cheap.
  const src = `/api/img/${encodeURIComponent(path)}`;
  return (
    <div className="relative w-40 h-40 shrink-0 bg-gray-950 rounded-lg border border-gray-800 flex items-center justify-center overflow-hidden">
      <button
        type="button"
        onClick={() => openImageProfile(jobID, path)}
        title="Open image profile"
        className="w-full h-full flex items-center justify-center"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={basename(path)} className="max-w-full max-h-full object-contain" />
      </button>
      {memorized && (
        <span className="absolute bottom-1.5 left-1.5 text-[10px] px-1.5 py-0.5 rounded bg-sky-700/80 text-white">
          memorised
        </span>
      )}
    </div>
  );
}

function TrendPanel({ seriesState }: { seriesState: SeriesState | undefined }) {
  return (
    <div className="flex-1 min-w-0 flex items-center">
      {!seriesState || seriesState.status === 'loading' ? (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <LuLoader className="w-4 h-4 animate-spin" /> Loading trend...
        </div>
      ) : seriesState.status === 'error' ? (
        <div className="text-xs text-red-400">Failed to load trend.</div>
      ) : (
        <LossSparkline points={seriesState.points} width={480} height={100} />
      )}
    </div>
  );
}

function BatchRow({
  item,
  seriesState,
  jobID,
  memorized,
}: {
  item: CurrentBatchItem;
  seriesState: SeriesState | undefined;
  jobID: string;
  memorized: boolean;
}) {
  const trainedSizeDiffers = item.train_width !== item.width || item.train_height !== item.height;

  return (
    <div className="flex items-stretch gap-4 py-3 border-b border-gray-800/60 last:border-b-0">
      <div className="relative">
        <ItemThumb jobID={jobID} path={item.path} memorized={memorized} />
        {item.is_reg && (
          <span className="absolute top-1.5 left-1.5 text-[10px] px-1.5 py-0.5 rounded bg-blue-600/80 text-white">
            reg
          </span>
        )}
      </div>

      <div className="min-w-0 w-56 shrink-0 flex flex-col justify-center">
        <div className="text-xs text-gray-200 truncate" title={item.path}>
          {basename(item.path)}
        </div>
        <div className="text-[11px] text-gray-500 line-clamp-2" title={item.caption ?? ''}>
          {item.caption || <span className="italic">no caption</span>}
        </div>
        <div className="text-[10px] text-gray-600 mt-1 space-y-0.5">
          <div>
            source {item.width}×{item.height}
          </div>
          <div className={trainedSizeDiffers ? 'text-blue-400' : undefined}>
            training at {item.train_width}×{item.train_height}
          </div>
          {item.timestep != null && (
            <div title="Timestep this image was just trained at (0-1000, higher = noisier)">
              t {Math.round(item.timestep)} · {noiseBand(item.timestep)}
            </div>
          )}
          {item.rel_loss != null && (
            <div
              title="Noise-adjusted loss of this step (1.0 = typical at that noise level) vs this image's usual value"
              className={
                item.difficulty != null && item.rel_loss > item.difficulty * 1.5 ? 'text-orange-300' : undefined
              }
            >
              adj. loss {item.rel_loss.toFixed(2)}
              {item.difficulty != null && ` (usual ${item.difficulty.toFixed(2)})`}
            </div>
          )}
        </div>
      </div>

      <TrendPanel seriesState={seriesState} />
    </div>
  );
}

// One file at one past step, from file_losses. Captions and source dimensions
// aren't stored per step (they'd go stale under a live caption edit), so this
// shows what the database does know about the step.
function HistoryRow({
  item,
  seriesState,
  jobID,
  memorized,
}: {
  item: StepHistoryItem;
  seriesState: SeriesState | undefined;
  jobID: string;
  memorized: boolean;
}) {
  return (
    <div className="flex items-stretch gap-4 py-3 border-b border-gray-800/60 last:border-b-0">
      <ItemThumb jobID={jobID} path={item.path} memorized={memorized} />

      <div className="min-w-0 w-56 shrink-0 flex flex-col justify-center">
        <div className="text-xs text-gray-200 truncate" title={item.path}>
          {basename(item.path)}
        </div>
        <div className="text-[10px] text-gray-600 mt-1 space-y-0.5">
          {item.train_width != null && item.train_height != null && (
            <div>
              trained at {item.train_width}×{item.train_height}
            </div>
          )}
          {item.timestep != null && (
            <div title="Timestep this image was trained at (0-1000, higher = noisier)">
              t {Math.round(item.timestep)} · {noiseBand(item.timestep)}
            </div>
          )}
          {item.loss != null && <div title="Raw loss of this step">loss {item.loss.toPrecision(4)}</div>}
          {item.rel_loss != null && (
            <div
              title="Noise-adjusted loss of this step (1.0 = typical at that noise level) vs this image's usual value up to then"
              className={
                item.difficulty != null && item.rel_loss > item.difficulty * 1.5 ? 'text-orange-300' : undefined
              }
            >
              adj. loss {item.rel_loss.toFixed(2)}
              {item.difficulty != null && ` (usual ${item.difficulty.toFixed(2)})`}
            </div>
          )}
        </div>
      </div>

      <TrendPanel seriesState={seriesState} />
    </div>
  );
}

export default function CurrentBatchViewer({ job }: Props) {
  // current_batch.json is just a snapshot written by whatever step last ran —
  // it isn't cleared when training stops, so it goes stale rather than empty.
  // job.status is the authoritative "is this actually running right now"
  // signal (already on hand from aitk_db.db, no extra fetch), and unlike any
  // cleanup the trainer itself might try to do on exit, it stays correct
  // even if the process was killed or crashed instead of stopping cleanly.
  const isTraining = job.status === 'running';

  const { status, loadStatus } = useCurrentBatch(job.id, isTraining ? 1500 : null);

  // step history scrubber: null = following the live step
  const range = useStepHistoryRange(job.id, isTraining ? 5000 : null);
  const [historyStep, setHistoryStep] = useState<number | null>(null);
  const { data: history, loading: historyLoading } = useStepHistory(job.id, historyStep);
  const viewingHistory = historyStep !== null;
  const hasHistory = range.max_step != null;

  const [now, setNow] = useState(Date.now());
  const [series, setSeries] = useState<Record<string, SeriesState>>({});
  // memorised images, from training_insights
  const [memorized, setMemorized] = useState<Set<string>>(new Set());
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${job.id}/insights`, { params: { parts: 'memorized' } })
        .then(res => setMemorized(new Set((res.data?.memorized?.files ?? []).map((f: any) => f.path))))
        .catch(() => {}),
    isTraining ? 30000 : null,
    [job.id],
  );

  // tick so the "updated Xs ago" label stays live between polls
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // fetch/refresh each visible file's loss trend whenever the visible set or
  // the step advances (live or scrubbed). Keeps prior points on screen while
  // refetching (no flicker) — only shows the loading spinner the first time a
  // path is seen.
  const visiblePaths = (viewingHistory ? (history?.items ?? []).map(i => i.path) : status.items.map(i => i.path)).join(
    '|',
  );
  const visibleStep = viewingHistory ? history?.step : status.step;
  useEffect(() => {
    const paths = visiblePaths ? visiblePaths.split('|') : [];
    if (paths.length === 0) return;
    let cancelled = false;

    paths.forEach(p => {
      setSeries(prev => (prev[p] ? prev : { ...prev, [p]: { status: 'loading' } }));
      apiClient
        // noise-adjusted when the run has it (the API falls back to raw)
        .get(`/api/jobs/${job.id}/file_loss`, { params: { file: p, adjusted: 1 } })
        .then(res => {
          if (cancelled) return;
          const points = (res.data?.points ?? []) as LossSparklinePoint[];
          setSeries(prev => ({ ...prev, [p]: { status: 'success', points } }));
        })
        .catch(err => {
          if (cancelled) return;
          console.error('Error fetching file loss series:', err);
          setSeries(prev => ({ ...prev, [p]: { status: 'error' } }));
        });
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, visiblePaths, visibleStep]);

  const staleSecs = status.updated_at ? Math.max(0, Math.round((now - status.updated_at * 1000) / 1000)) : null;
  const isStale = staleSecs !== null && staleSecs > 30;

  const sliderMax = range.max_step ?? 0;
  const sliderMin = range.min_step ?? 0;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center gap-3 shrink-0">
        <div className="flex items-center gap-2 shrink-0">
          <div
            className={`h-2 w-2 rounded-full ${isTraining && !viewingHistory && !isStale ? 'bg-green-400' : 'bg-gray-500'}`}
          />
          <h2 className="text-gray-100 text-sm font-medium">{viewingHistory ? 'Step history' : 'Current batch'}</h2>
          {(viewingHistory || (isTraining && status.step != null)) && (
            <span className="text-xs text-gray-400">
              step {(viewingHistory ? historyStep : status.step)!.toLocaleString()}
              {viewingHistory && <span className="text-gray-500"> · history</span>}
            </span>
          )}
        </div>

        {hasHistory && (
          <div className="flex items-center gap-2 flex-1 min-w-0 max-w-xl">
            <LuHistory className="w-3.5 h-3.5 text-gray-500 shrink-0" />
            <input
              type="range"
              min={sliderMin}
              max={sliderMax}
              step={1}
              value={historyStep ?? sliderMax}
              onChange={e => {
                const v = parseInt(e.target.value, 10);
                setHistoryStep(v >= sliderMax ? null : v);
              }}
              className="flex-1 accent-blue-500 min-w-20"
              title={`Scrub through recorded steps ${sliderMin.toLocaleString()}–${sliderMax.toLocaleString()} (far right = live)`}
            />
            <button
              type="button"
              onClick={() => setHistoryStep(null)}
              className={`text-[11px] px-2 py-0.5 rounded border shrink-0 flex items-center gap-1 ${
                viewingHistory
                  ? 'border-gray-600 text-gray-300 hover:bg-gray-700'
                  : 'border-green-700/60 text-green-400 cursor-default'
              }`}
              title="Follow the live step"
            >
              <LuRadio className="w-3 h-3" /> Live
            </button>
          </div>
        )}

        <span className="text-xs text-gray-500 shrink-0 ml-auto">
          {isTraining && !viewingHistory && loadStatus === 'loading' && 'Loading...'}
          {isTraining && !viewingHistory && staleSecs !== null && `updated ${staleSecs}s ago`}
        </span>
      </div>

      <div className="px-4 flex-1 min-h-0 overflow-y-auto">
        {viewingHistory ? (
          historyLoading && !history ? (
            <div className="flex items-center justify-center py-16 text-gray-500 gap-2 text-sm">
              <LuLoader className="w-4 h-4 animate-spin" /> Loading step…
            </div>
          ) : !history || history.items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-500">
              <LuImageOff className="w-8 h-8 mb-3" />
              <div className="text-sm">Nothing recorded at this step</div>
              <div className="text-xs opacity-75 mt-1">Steps are recorded once the job starts training.</div>
            </div>
          ) : (
            <div className="flex flex-col">
              {metricEntries(history.metrics).length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-3 pb-1 border-b border-gray-800/60">
                  {metricEntries(history.metrics).map(([k, v]) => (
                    <span
                      key={k}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-300"
                      title={`${k} (as logged at or before this step)`}
                    >
                      <span className="text-gray-500">{k}</span> {v}
                    </span>
                  ))}
                </div>
              )}
              {history.items.map((item, i) => (
                <HistoryRow
                  key={`${item.path}-${i}`}
                  item={item}
                  seriesState={series[item.path]}
                  jobID={job.id}
                  memorized={memorized.has(item.path)}
                />
              ))}
            </div>
          )
        ) : !isTraining ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            <LuImageOff className="w-8 h-8 mb-3" />
            <div className="text-sm">Not currently training</div>
            <div className="text-xs opacity-75 mt-1">This job isn&apos;t running right now.</div>
            {hasHistory && (
              <button
                type="button"
                onClick={() => setHistoryStep(range.max_step)}
                className="mt-4 text-xs px-3 py-1.5 rounded border border-gray-600 text-gray-300 hover:bg-gray-800"
              >
                Browse step {range.max_step?.toLocaleString()}
              </button>
            )}
          </div>
        ) : status.items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            <LuImageOff className="w-8 h-8 mb-3" />
            <div className="text-sm">No batch data yet</div>
            <div className="text-xs opacity-75 mt-1">This fills in once the job starts training.</div>
          </div>
        ) : (
          <div className="flex flex-col">
            {status.items.map((item, i) => (
              <BatchRow
                key={`${item.path}-${i}`}
                item={item}
                seriesState={series[item.path]}
                jobID={job.id}
                memorized={memorized.has(item.path)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
