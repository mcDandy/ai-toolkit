'use client';

import { useState, useEffect } from 'react';
import { Job } from '@prisma/client';
import useCurrentBatch, { CurrentBatchItem } from '@/hooks/useCurrentBatch';
import { apiClient } from '@/utils/api';
import { LuImageOff, LuLoader } from 'react-icons/lu';
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
  // Not ?thumb=1 on purpose: the pre-generated thumbnail is a sharp() resize
  // with `fit: 'cover'`, which center-crops to a square before it's ever
  // saved to disk ÔÇö object-contain below can't recover image data that's
  // already gone from the file. Only a handful of images are on screen here
  // at once (the current batch), so loading the originals is cheap.
  const src = `/api/img/${encodeURIComponent(item.path)}`;
  const trainedSizeDiffers = item.train_width !== item.width || item.train_height !== item.height;

  return (
    <div className="flex items-stretch gap-4 py-3 border-b border-gray-800/60 last:border-b-0">
      {/* padded thumbnail ÔÇö object-contain so the full frame is always visible, never cropped */}
      <div className="relative w-40 h-40 shrink-0 bg-gray-950 rounded-lg border border-gray-800 flex items-center justify-center overflow-hidden">
        <button
          type="button"
          onClick={() => openImageProfile(jobID, item.path)}
          title="Open image profile"
          className="w-full h-full flex items-center justify-center"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={basename(item.path)} className="max-w-full max-h-full object-contain" />
        </button>
        {memorized && (
          <span className="absolute bottom-1.5 left-1.5 text-[10px] px-1.5 py-0.5 rounded bg-sky-700/80 text-white">
            memorised
          </span>
        )}
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
            source {item.width}├Ś{item.height}
          </div>
          <div className={trainedSizeDiffers ? 'text-blue-400' : undefined}>
            training at {item.train_width}├Ś{item.train_height}
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
    </div>
  );
}

export default function CurrentBatchViewer({ job }: Props) {
  // current_batch.json is just a snapshot written by whatever step last ran ÔÇö
  // it isn't cleared when training stops, so it goes stale rather than empty.
  // job.status is the authoritative "is this actually running right now"
  // signal (already on hand from aitk_db.db, no extra fetch), and unlike any
  // cleanup the trainer itself might try to do on exit, it stays correct
  // even if the process was killed or crashed instead of stopping cleanly.
  const isTraining = job.status === 'running';

  const { status, loadStatus } = useCurrentBatch(job.id, isTraining ? 1500 : null);
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

  // fetch/refresh each visible file's loss trend whenever the training step
  // advances. Keeps prior points on screen while refetching (no flicker) ÔÇö
  // only shows the loading spinner the first time a path is seen.
  useEffect(() => {
    if (!isTraining) return;
    const paths = Array.from(new Set(status.items.map(i => i.path)));
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
  }, [job.id, status.step, isTraining]);

  const staleSecs = status.updated_at ? Math.max(0, Math.round((now - status.updated_at * 1000) / 1000)) : null;
  const isStale = staleSecs !== null && staleSecs > 30;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${isTraining && !isStale ? 'bg-green-400' : 'bg-gray-500'}`} />
          <h2 className="text-gray-100 text-sm font-medium">Current batch</h2>
          {isTraining && status.step != null && (
            <span className="text-xs text-gray-400">step {status.step.toLocaleString()}</span>
          )}
        </div>
        <span className="text-xs text-gray-500">
          {isTraining && loadStatus === 'loading' && 'Loading...'}
          {isTraining && staleSecs !== null && `updated ${staleSecs}s ago`}
        </span>
      </div>

      <div className="px-4 flex-1 min-h-0 overflow-y-auto">
        {!isTraining ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            <LuImageOff className="w-8 h-8 mb-3" />
            <div className="text-sm">Not currently training</div>
            <div className="text-xs opacity-75 mt-1">This job isn&apos;t running right now.</div>
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
