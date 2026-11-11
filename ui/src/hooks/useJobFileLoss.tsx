'use client';

import { useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export interface FileLossRow {
  file_path: string;
  count: number;
  avg_loss: number | null;
  min_loss: number | null;
  max_loss: number | null;
  last_loss: number | null;
  last_step: number | null;
  // predicted loss change (best-fit trend line, first logged step to last)
  // as a fraction of this file's average loss; null until it has enough
  // history. Positive = improving.
  score: number | null;
  // secondary, informational label derived from the same fit ÔÇö not used for
  // sort/color, just a hint at why the score looks the way it does.
  category: 'learning' | 'worsening' | 'stuck' | 'exhausted' | null;
  // average noise-adjusted loss: 1.0 = as hard as a typical sample at the same
  // timestep, >1 harder, well below 1 = possibly memorised. null on old data.
  difficulty: number | null;
  // difficulty in 5 noise bands, low noise (detail) first; null on old data
  bands: (number | null)[] | null;
}

// adjusted: score/avg/min/max/last use the noise-adjusted loss instead of the raw loss
export default function useJobFileLoss(jobID: string, reloadInterval: number | null = null, adjusted = false) {
  const [rows, setRows] = useState<FileLossRow[]>([]);
  // whether this job has noise-adjusted data at all (needs timesteps, newer runs)
  const [hasAdjusted, setHasAdjusted] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'refreshing' | 'success' | 'error'>('idle');

  const refresh = useCallback(async () => {
    if (!jobID) return;
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    try {
      const res = await apiClient.get(`/api/jobs/${jobID}/file_loss`, { params: adjusted ? { adjusted: 1 } : {} });
      setRows((res.data?.rows ?? []) as FileLossRow[]);
      setHasAdjusted(!!res.data?.has_adjusted);
      setStatus('success');
    } catch (err) {
      console.error('Error fetching per-file loss:', err);
      setStatus('error');
    }
  }, [jobID, adjusted]);

  usePollLoop(refresh, reloadInterval, [jobID, adjusted]);

  return { rows, status, refresh, hasAdjusted };
}
