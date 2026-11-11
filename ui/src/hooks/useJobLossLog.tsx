'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

// Columnar series: steps[i] pairs with values[i]. Null / non-finite values are
// dropped at ingest, so every value here is a finite number.
export interface LossSeries {
  steps: number[];
  values: number[];
}

type SeriesMap = Record<string, LossSeries>;

interface LossResponse {
  keys?: string[];
  since_step?: number;
  // resume detection (see the loss route): wall_time of since_step now, and of
  // the newest step in this response
  since_wall_time?: number | null;
  last_step?: number | null;
  last_wall_time?: number | null;
  series?: Record<string, { steps: number[]; values: (number | null)[] }>;
}

export default function useJobLossLog(jobID: string, reloadInterval: null | number = null) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);

  // Highest step seen across every key. Polls ask for step >= this (not >) so a
  // key logged at that step after the previous poll is still picked up; the
  // per-key merge below drops the one re-sent point.
  const lastStepRef = useRef<number | null>(null);
  // wall_time of lastStepRef's step. A resume from an earlier checkpoint drops
  // and re-logs steps, which changes it (or removes the step): then reload.
  const lastWallRef = useRef<number | null>(null);

  const lossKeys = useMemo(() => {
    const base = keys ?? [];
    // if keys table is empty early on, fall back to just "loss"
    if (base.length === 0) return ['loss'];
    return [...base].sort();
  }, [keys]);

  const refreshLoss = useCallback(async () => {
    if (!jobID) return;

    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
    setStatus(loadStatus);

    try {
      // One request returns every key. Incremental after the initial load.
      const params: Record<string, number> = {};
      if (reloadInterval && didInitialLoadRef.current && lastStepRef.current != null) {
        params.since_step = lastStepRef.current;
      }
      let res = await apiClient.get(`/api/jobs/${jobID}/loss`, { params });
      let data = res.data as LossResponse;
      let reload = false;
      if (params.since_step != null && lastWallRef.current != null && data.since_wall_time !== lastWallRef.current) {
        // our tail is stale (job resumed from an earlier checkpoint): start over
        res = await apiClient.get(`/api/jobs/${jobID}/loss`);
        data = res.data as LossResponse;
        reload = true;
      }

      const newKeys = data.keys ?? [];
      setKeys(newKeys);
      const wantedLossKeys = (newKeys.length ? [...newKeys] : ['loss']).sort();
      const incoming = data.series ?? {};
      const initial = !didInitialLoadRef.current || reload;

      setSeries(prev => {
        const next: SeriesMap = initial ? {} : { ...prev };
        let maxStep = lastStepRef.current ?? null;

        for (const k of wantedLossKeys) {
          const inc = incoming[k];
          const existing = initial ? undefined : next[k];
          const prevLast = existing && existing.steps.length ? existing.steps[existing.steps.length - 1] : null;

          const steps: number[] = existing ? existing.steps.slice() : [];
          const values: number[] = existing ? existing.values.slice() : [];
          let appended = false;
          if (inc) {
            const n = Math.min(inc.steps.length, inc.values.length);
            for (let i = 0; i < n; i++) {
              const v = inc.values[i];
              if (v === null || !Number.isFinite(v)) continue;
              const s = inc.steps[i];
              if (prevLast != null && s <= prevLast) continue;
              steps.push(s);
              values.push(v as number);
              appended = true;
            }
          }
          if (initial || appended) next[k] = { steps, values };
          else next[k] = existing ?? { steps: [], values: [] };

          const arr = next[k].steps;
          if (arr.length) {
            const last = arr[arr.length - 1];
            if (maxStep == null || last > maxStep) maxStep = last;
          }
        }

        // remove stale keys that no longer exist (rare, but keeps UI clean)
        for (const existingKey of Object.keys(next)) {
          if (!wantedLossKeys.includes(existingKey)) delete next[existingKey];
        }

        lastStepRef.current = maxStep;
        // only trust the wall time if it belongs to the step we will poll from
        lastWallRef.current =
          data.last_step != null && data.last_step === maxStep ? (data.last_wall_time ?? null) : null;
        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching loss logs:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval]);

  // Delete every logged step in [minStep, maxStep] from the on-disk log, then
  // drop those points locally so the chart updates without a full reload.
  const deleteRange = useCallback(
    async (minStep: number, maxStep: number) => {
      if (!jobID) return;
      await apiClient.delete(`/api/jobs/${jobID}/loss`, { data: { min_step: minStep, max_step: maxStep } });
      setSeries(prev => {
        const next: SeriesMap = {};
        let last: number | null = null;
        for (const k of Object.keys(prev)) {
          const steps: number[] = [];
          const values: number[] = [];
          const src = prev[k];
          for (let i = 0; i < src.steps.length; i++) {
            const s = src.steps[i];
            if (s < minStep || s > maxStep) {
              steps.push(s);
              values.push(src.values[i]);
            }
          }
          next[k] = { steps, values };
          if (steps.length && (last == null || steps[steps.length - 1] > last)) last = steps[steps.length - 1];
        }
        // Re-anchor incremental polling to the new tail so a deleted tail
        // isn't treated as already-fetched.
        lastStepRef.current = last;
        lastWallRef.current = null;
        return next;
      });
    },
    [jobID],
  );

  // reset when job changes. Declared before the poll loop so the reset runs
  // before the first fetch when jobID changes.
  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepRef.current = null;
    lastWallRef.current = null;
    setSeries({});
    setKeys([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(refreshLoss, reloadInterval, [jobID]);

  return { series, keys, lossKeys, status, refreshLoss, deleteRange, setSeries };
}
