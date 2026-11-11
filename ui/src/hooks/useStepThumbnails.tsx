'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export interface StepThumbnail {
  step: number;
  file_path: string;
}

export default function useStepThumbnails(jobID: string, reloadInterval: number | null = 3000) {
  const [items, setItems] = useState<StepThumbnail[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const didInitialLoadRef = useRef(false);
  const lastStepRef = useRef<number | null>(null);
  // wall_time of lastStepRef's step; a resume from an earlier checkpoint drops
  // and re-logs steps, which changes it (or removes the step)
  const lastWallRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);

  const refresh = useCallback(async () => {
    if (!jobID || inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(prev => (prev === 'success' ? 'success' : 'loading'));
    try {
      const params: Record<string, any> = { limit: 200000 };
      if (didInitialLoadRef.current && lastStepRef.current != null) {
        params.since_step = lastStepRef.current;
      }
      let res = await apiClient.get(`/api/jobs/${jobID}/step_thumbnails`, { params });
      let replace = !didInitialLoadRef.current;
      if (params.since_step != null) {
        const sinceWall = res.data?.since_wall_time ?? null;
        if (lastWallRef.current == null) {
          // the steps row wasn't flushed yet when we last saw this step
          lastWallRef.current = sinceWall;
        } else if (sinceWall !== lastWallRef.current) {
          // stale tail after a resume (or a deleted range): reload everything
          res = await apiClient.get(`/api/jobs/${jobID}/step_thumbnails`, { params: { limit: 200000 } });
          replace = true;
        }
      }
      const newItems = (res.data?.items ?? []) as StepThumbnail[];

      if (newItems.length > 0) {
        setItems(prev => (replace ? newItems : [...prev, ...newItems]));
        lastStepRef.current = newItems[newItems.length - 1].step;
        lastWallRef.current = res.data?.last_wall_time ?? null;
      } else if (replace) {
        setItems([]);
        lastStepRef.current = null;
        lastWallRef.current = null;
      }

      didInitialLoadRef.current = true;
      setStatus('success');
    } catch (err) {
      console.error('Error fetching step thumbnails:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID]);

  // reset when job changes
  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepRef.current = null;
    setItems([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(refresh, reloadInterval, [jobID]);

  return { items, status, refresh };
}
