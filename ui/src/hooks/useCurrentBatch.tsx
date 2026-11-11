'use client';

import { useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export interface CurrentBatchItem {
  path: string;
  caption: string | null;
  width: number;
  height: number;
  // resolution this file is actually being cropped/resized to for training
  // (post-bucketing) ÔÇö can differ from the source width/height above.
  train_width: number;
  train_height: number;
  is_reg: boolean;
  // filled in after the step ran (live_learning_monitor): the timestep it was
  // trained at (0..1000, higher = noisier), its raw and noise-adjusted loss,
  // and its usual noise-adjusted loss so far ("difficulty")
  timestep?: number | null;
  loss?: number | null;
  rel_loss?: number | null;
  difficulty?: number | null;
}

export interface CurrentBatchStatus {
  step: number | null;
  updated_at: number | null;
  // true once the per-item results of the step are in
  trained?: boolean;
  items: CurrentBatchItem[];
}

export default function useCurrentBatch(jobID: string, reloadInterval: number | null = 1500) {
  const [status, setStatus] = useState<CurrentBatchStatus>({ step: null, updated_at: null, items: [] });
  const [loadStatus, setLoadStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refresh = useCallback(async () => {
    if (!jobID) return;
    try {
      const res = await apiClient.get(`/api/jobs/${jobID}/current_batch`);
      setStatus(res.data as CurrentBatchStatus);
      setLoadStatus('success');
    } catch (err) {
      console.error('Error fetching current batch:', err);
      setLoadStatus('error');
    }
  }, [jobID]);

  usePollLoop(refresh, reloadInterval, [jobID]);

  return { status, loadStatus, refresh };
}
