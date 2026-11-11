'use client';

import { useMemo, useState, useCallback, Fragment, type CSSProperties } from 'react';
import { Job } from '@prisma/client';
import useJobFileLoss, { FileLossRow } from '@/hooks/useJobFileLoss';
import { apiClient } from '@/utils/api';
import { FaDownload, FaTrashAlt } from 'react-icons/fa';
import { LuLoader, LuChevronDown, LuChevronRight } from 'react-icons/lu';
import LossSparkline, { LossSparklinePoint } from './LossSparkline';
import { openConfirm } from './ConfirmModal';
import { openImageProfile } from './ImageProfileModal';

// difficulty per noise band as 5 tiny bars (low noise left): tall on the right =
// hard where composition/caption matter, tall on the left = detail/quality
function BandBars({ bands }: { bands: (number | null)[] | null }) {
  if (!bands || bands.every(b => b == null)) return <span className="text-gray-600">—</span>;
  const max = Math.max(1.6, ...bands.map(b => b ?? 0));
  return (
    <svg viewBox="0 0 50 16" className="w-12 h-4" aria-hidden>
      <line x1={0} x2={50} y1={16 - (1 / max) * 16} y2={16 - (1 / max) * 16} stroke="#6b7280" strokeWidth={0.5} />
      {bands.map((b, i) =>
        b == null ? null : (
          <rect
            key={i}
            x={i * 10 + 1}
            y={16 - (b / max) * 16}
            width={8}
            height={(b / max) * 16}
            fill={b > 1.3 ? '#fb923c' : b < 0.6 ? '#38bdf8' : '#60a5fa'}
          />
        ),
      )}
    </svg>
  );
}

interface Props {
  job: Job;
}

type SeriesState = { status: 'loading' } | { status: 'error' } | { status: 'success'; points: LossSparklinePoint[] };

type SortKey = 'avg_loss' | 'count' | 'last_loss' | 'last_step' | 'file_path' | 'score' | 'difficulty';

function basename(p: string) {
  return p.replace(/^.*[\\/]/, '');
}

function fmt(v: number | null) {
  if (v === null || !Number.isFinite(v)) return 'ÔÇö';
  return v.toPrecision(4);
}

function fmtStep(v: number | null) {
  if (v === null || !Number.isFinite(v)) return 'ÔÇö';
  return v.toLocaleString();
}

function fmtScore(v: number | null) {
  if (v === null || !Number.isFinite(v)) return 'ÔÇö';
  const pct = v * 100;
  return `${pct > 0 ? '+' : ''}${pct.toFixed(0)}%`;
}

// Secondary hint shown under the score ÔÇö not sorted/colored on its own.
const CATEGORY_LABEL: Record<string, string> = {
  learning: 'learning',
  worsening: 'worsening',
  stuck: 'stuck',
  exhausted: 'exhausted',
};

// Red (not learning / possibly a bad caption) -> yellow -> green (improving),
// clamped to a +/-50% loss-reduction range so a single outlier doesn't wash
// out the rest of the gradient.
function scoreRowStyle(v: number | null): CSSProperties | undefined {
  if (v === null || !Number.isFinite(v)) return undefined;
  const clamped = Math.max(-0.5, Math.min(0.5, v));
  const hue = 60 * (clamped / 0.5 + 1);
  return { backgroundColor: `hsla(${hue}, 65%, 40%, 0.22)` };
}

export default function FileLossTable({ job }: Props) {
  // noise-adjusted by default: raw per-file loss mostly reflects which timestep
  // each file happened to be trained at. Falls back to raw on older data.
  const [adjustedPref, setAdjustedPref] = useState(true);
  const { rows, status, refresh, hasAdjusted } = useJobFileLoss(job.id, 10000, adjustedPref);
  const adjusted = adjustedPref && hasAdjusted;
  // default: worst learning score first (score ascending) ÔÇö surfaces files
  // that likely have a bad caption or are otherwise not being learned.
  const [sortKey, setSortKey] = useState<SortKey>('score');
  const [sortDesc, setSortDesc] = useState(false);
  const [expandedPath, setExpandedPath] = useState<string | null>(null);
  const [series, setSeries] = useState<Record<string, SeriesState>>({});
  const [deletedPaths, setDeletedPaths] = useState<Set<string>>(new Set());
  const [deletingPath, setDeletingPath] = useState<string | null>(null);

  const handleDelete = useCallback((filePath: string) => {
    openConfirm({
      title: 'Delete training file',
      message: `Delete "${basename(filePath)}" and its caption file from disk? This cannot be undone.`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeletingPath(filePath);
        try {
          await apiClient.post('/api/img/delete', { imgPath: filePath });
          setDeletedPaths(prev => new Set(prev).add(filePath));
          setExpandedPath(prev => (prev === filePath ? null : prev));
        } catch (err) {
          console.error('Error deleting file:', err);
        } finally {
          setDeletingPath(null);
        }
      },
    });
  }, []);

  const toggleExpand = useCallback(
    (filePath: string) => {
      if (expandedPath === filePath) {
        setExpandedPath(null);
        return;
      }
      setExpandedPath(filePath);
      const cacheKey = `${adjusted ? 'adj' : 'raw'}:${filePath}`;
      if (series[cacheKey]) return;
      setSeries(prev => ({ ...prev, [cacheKey]: { status: 'loading' } }));
      apiClient
        .get(`/api/jobs/${job.id}/file_loss`, { params: { file: filePath, ...(adjusted ? { adjusted: 1 } : {}) } })
        .then(res => {
          const points = (res.data?.points ?? []) as LossSparklinePoint[];
          setSeries(prev => ({ ...prev, [cacheKey]: { status: 'success', points } }));
        })
        .catch(err => {
          console.error('Error fetching file loss series:', err);
          setSeries(prev => ({ ...prev, [cacheKey]: { status: 'error' } }));
        });
    },
    [expandedPath, series, job.id, adjusted],
  );

  const visibleRows = useMemo(() => rows.filter(r => !deletedPaths.has(r.file_path)), [rows, deletedPaths]);

  const sorted = useMemo(() => {
    const copy = [...visibleRows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      // nulls (e.g. score with not enough history yet) always sort last,
      // regardless of direction ÔÇö they're "unknown", not "worst".
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      if (typeof av === 'string' || typeof bv === 'string') {
        return sortDesc ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      }
      return sortDesc ? (bv as number) - (av as number) : (av as number) - (bv as number);
    });
    return copy;
  }, [visibleRows, sortKey, sortDesc]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDesc(d => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const Header = ({ label, sortKeyName, title }: { label: string; sortKeyName: SortKey; title?: string }) => (
    <th
      className="px-3 py-2 text-left cursor-pointer select-none hover:text-gray-200 whitespace-nowrap"
      onClick={() => toggleSort(sortKeyName)}
      title={title}
    >
      {label}
      {sortKey === sortKeyName && <span className="ml-1">{sortDesc ? 'Ôľ╝' : 'Ôľ▓'}</span>}
    </th>
  );

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Per-file loss</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {(status === 'success' || status === 'refreshing') && `${visibleRows.length.toLocaleString()} files`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {hasAdjusted && (
            <label
              className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer select-none"
              title="Divide each loss by the run's typical loss at the same timestep (noise level). Raw diffusion loss depends mostly on the random timestep, so this makes scores and trends far less noisy. 1.0 = typical."
            >
              <input
                type="checkbox"
                checked={adjustedPref}
                onChange={e => setAdjustedPref(e.target.checked)}
                className="accent-blue-500"
              />
              Noise-adjusted
            </label>
          )}
          <button
            type="button"
            onClick={() => refresh()}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
          <a
            href={`/api/jobs/${job.id}/file_loss?format=csv${adjusted ? '&adjusted=1' : ''}`}
            download={`${job.name}-file-loss.csv`}
            className="px-3 py-1 rounded-md text-xs bg-blue-600/80 hover:bg-blue-600 text-white border border-blue-500/50 flex items-center gap-1.5"
          >
            <FaDownload className="text-[10px]" />
            Save CSV
          </a>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-auto">
        {visibleRows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            {status === 'loading' ? (
              <LuLoader className="w-6 h-6 animate-spin mb-3" />
            ) : (
              <div className="text-sm">No per-file loss data yet.</div>
            )}
            {status !== 'loading' && (
              <div className="text-xs opacity-75 mt-1 max-w-sm text-center">
                This fills in once training starts, if the <code>live_learning_monitor</code> extension is installed.
              </div>
            )}
          </div>
        ) : (
          <table className="w-full text-xs text-gray-300 border-collapse">
            <thead className="sticky top-0 bg-gray-900 text-gray-400 border-b border-gray-800">
              <tr>
                <th className="w-6" />
                <Header label="File" sortKeyName="file_path" />
                <Header
                  label="Score"
                  sortKeyName="score"
                  title="Predicted loss change from a best-fit line through this file's loss history (first step to last), as a percent of its average loss. Green = learning well, red = flat or getting worse (check the caption). The small label underneath is a secondary hint (learning / worsening / stuck / exhausted)."
                />
                <Header
                  label="Difficulty"
                  sortKeyName="difficulty"
                  title="Average noise-adjusted loss: 1.0 = as hard as a typical sample at the same timestep. Well above 1 = hard for the model (often a caption problem or an outlier image). Well below 1 = easy, or memorised."
                />
                <th
                  className="px-3 py-2 text-left whitespace-nowrap"
                  title="Difficulty by noise level, low noise (detail) on the left, high noise (composition) on the right. Line = typical. Hard only on the right usually means the caption; only on the left, image quality/resolution."
                >
                  Noise profile
                </th>
                <Header label={adjusted ? 'Avg (adj.)' : 'Avg loss'} sortKeyName="avg_loss" />
                <th className="px-3 py-2 text-left whitespace-nowrap">Min</th>
                <th className="px-3 py-2 text-left whitespace-nowrap">Max</th>
                <Header label={adjusted ? 'Last (adj.)' : 'Last loss'} sortKeyName="last_loss" />
                <Header label="Last step" sortKeyName="last_step" />
                <Header label="Times trained" sortKeyName="count" />
                <th className="w-8" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((row: FileLossRow) => {
                const isExpanded = expandedPath === row.file_path;
                const rowSeries = series[`${adjusted ? 'adj' : 'raw'}:${row.file_path}`];
                return (
                  <Fragment key={row.file_path}>
                    <tr
                      className="border-b border-gray-800/60 hover:brightness-125 cursor-pointer transition-[filter]"
                      style={scoreRowStyle(row.score)}
                      onClick={() => toggleExpand(row.file_path)}
                    >
                      <td className="pl-3 text-gray-500">
                        {isExpanded ? (
                          <LuChevronDown className="w-3.5 h-3.5" />
                        ) : (
                          <LuChevronRight className="w-3.5 h-3.5" />
                        )}
                      </td>
                      <td className="px-3 py-1.5 max-w-xs">
                        <div className="flex items-center gap-2 min-w-0">
                          <button
                            type="button"
                            title="Open image profile"
                            onClick={e => {
                              e.stopPropagation();
                              openImageProfile(job.id, row.file_path);
                            }}
                            className="shrink-0 rounded hover:ring-2 hover:ring-blue-500"
                          >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={`/api/img/${encodeURIComponent(row.file_path)}?thumb=1`}
                              alt=""
                              className="w-8 h-8 rounded object-cover border border-gray-800 bg-gray-950"
                            />
                          </button>
                          <span className="truncate" title={row.file_path}>
                            {basename(row.file_path)}
                          </span>
                        </div>
                      </td>
                      <td className="px-3 py-1.5">
                        <div className="font-medium">{fmtScore(row.score)}</div>
                        {row.category && (
                          <div className="text-[10px] text-gray-500 font-normal">{CATEGORY_LABEL[row.category]}</div>
                        )}
                      </td>
                      <td
                        className={
                          'px-3 py-1.5 font-mono ' +
                          (row.difficulty == null
                            ? ''
                            : row.difficulty > 1.3
                              ? 'text-orange-300'
                              : row.difficulty < 0.6
                                ? 'text-sky-300'
                                : '')
                        }
                      >
                        {row.difficulty == null ? '—' : row.difficulty.toFixed(2)}
                      </td>
                      <td className="px-3 py-1.5">
                        <BandBars bands={row.bands} />
                      </td>
                      <td className="px-3 py-1.5">{fmt(row.avg_loss)}</td>
                      <td className="px-3 py-1.5">{fmt(row.min_loss)}</td>
                      <td className="px-3 py-1.5">{fmt(row.max_loss)}</td>
                      <td className="px-3 py-1.5">{fmt(row.last_loss)}</td>
                      <td className="px-3 py-1.5">{fmtStep(row.last_step)}</td>
                      <td className="px-3 py-1.5">{row.count.toLocaleString()}</td>
                      <td className="px-2 py-1.5">
                        <button
                          type="button"
                          title="Delete file and caption"
                          disabled={deletingPath === row.file_path}
                          onClick={e => {
                            e.stopPropagation();
                            handleDelete(row.file_path);
                          }}
                          className="p-1.5 rounded-md text-gray-500 hover:text-red-400 hover:bg-red-950/40 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {deletingPath === row.file_path ? (
                            <LuLoader className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <FaTrashAlt className="w-3 h-3" />
                          )}
                        </button>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr className="border-b border-gray-800/60 bg-gray-950/60">
                        <td colSpan={12} className="px-4 py-3">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <div className="text-[11px] text-gray-500 mb-1" title={row.file_path}>
                                {adjusted ? 'Noise-adjusted loss' : 'Loss'} over steps for {basename(row.file_path)}
                              </div>
                              {!rowSeries || rowSeries.status === 'loading' ? (
                                <div className="flex items-center gap-2 text-xs text-gray-500 py-4">
                                  <LuLoader className="w-4 h-4 animate-spin" /> Loading trend...
                                </div>
                              ) : rowSeries.status === 'error' ? (
                                <div className="text-xs text-red-400 py-4">Failed to load trend.</div>
                              ) : (
                                <LossSparkline points={rowSeries.points} />
                              )}
                            </div>
                            <a
                              href={`/api/jobs/${job.id}/file_loss?format=csv&file=${encodeURIComponent(row.file_path)}${adjusted ? '&adjusted=1' : ''}`}
                              download={`${basename(row.file_path)}-loss.csv`}
                              onClick={e => e.stopPropagation()}
                              className="shrink-0 px-2.5 py-1 rounded-md text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 flex items-center gap-1.5"
                            >
                              <FaDownload className="text-[10px]" />
                              CSV
                            </a>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
