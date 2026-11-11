'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Job } from '@prisma/client';
import useStepThumbnails, { StepThumbnail } from '@/hooks/useStepThumbnails';
import { LuImageOff } from 'react-icons/lu';

interface Props {
  job: Job;
  // step currently under the mouse on the loss graph, or null when the
  // mouse isn't over the chart.
  hoveredStep: number | null;
}

const ITEM_WIDTH = 72; // px, includes gap ÔÇö thumbnail box is sized to match
const THUMB_SIZE = 64;
const RENDER_BUFFER = 6; // extra items rendered past each edge of the viewport
const RETURN_TO_END_DELAY_MS = 2200;

function basename(p: string) {
  return p.replace(/^.*[\\/]/, '');
}

// items are sorted ascending by step (the API guarantees this) ÔÇö binary
// search for the closest one to `target` instead of a linear scan.
function findNearestIndex(items: StepThumbnail[], target: number): number {
  if (items.length === 0) return -1;
  let lo = 0;
  let hi = items.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].step < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(items[lo - 1].step - target) <= Math.abs(items[lo].step - target)) {
    return lo - 1;
  }
  return lo;
}

export default function StepThumbnailStrip({ job, hoveredStep }: Props) {
  const { items, status } = useStepThumbnails(job.id, 3000);

  const containerRef = useRef<HTMLDivElement | null>(null);
  // The scrollable div only exists once items.length > 0 (see the ternary
  // below), so a plain useRef + mount-only effect would miss it: on first
  // render there's nothing to attach to yet, and that effect never runs
  // again once the fetch resolves and the div actually appears. A callback
  // ref fires exactly when the node attaches/detaches, whenever that is.
  const [containerEl, setContainerEl] = useState<HTMLDivElement | null>(null);
  const setContainerRef = useCallback((node: HTMLDivElement | null) => {
    containerRef.current = node;
    setContainerEl(node);
  }, []);

  const [containerWidth, setContainerWidth] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);
  const returnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasDoneInitialScrollRef = useRef(false);

  useEffect(() => {
    if (!containerEl) return;
    const ro = new ResizeObserver(() => setContainerWidth(containerEl.clientWidth));
    ro.observe(containerEl);
    setContainerWidth(containerEl.clientWidth);
    return () => ro.disconnect();
  }, [containerEl]);

  const totalWidth = items.length * ITEM_WIDTH;
  const maxScroll = Math.max(0, totalWidth - containerWidth);

  const clearReturnTimer = useCallback(() => {
    if (returnTimerRef.current) {
      clearTimeout(returnTimerRef.current);
      returnTimerRef.current = null;
    }
  }, []);

  // Reads scrollWidth/clientWidth live from the DOM rather than closing over
  // `maxScroll`, so this callback's identity never changes ÔÇö otherwise, new
  // steps arriving during active training would keep changing `maxScroll`
  // (and therefore this function) and endlessly reset the return-to-end
  // timer below, which depends on it.
  const scrollToEnd = useCallback((smooth: boolean) => {
    const el = containerRef.current;
    if (!el) return;
    const target = Math.max(0, el.scrollWidth - el.clientWidth);
    el.scrollTo({ left: target, behavior: smooth ? 'smooth' : 'auto' });
  }, []);

  // jump to the tail once thumbnails first arrive, so the strip starts
  // "live" instead of pinned to step 0.
  useEffect(() => {
    if (hasDoneInitialScrollRef.current) return;
    if (items.length === 0 || containerWidth === 0) return;
    hasDoneInitialScrollRef.current = true;
    scrollToEnd(false);
  }, [items.length, containerWidth, scrollToEnd]);

  const nearestIdx = useMemo(() => {
    if (hoveredStep == null) return -1;
    return findNearestIndex(items, hoveredStep);
  }, [items, hoveredStep]);

  // hover tracking: center the hovered step instantly (tied directly to
  // mouse position ÔÇö a smooth-scroll here would lag behind the cursor).
  // Clamped to [0, maxScroll], so a step near the end just shows the tail
  // fully rather than centering with blank space past it.
  useEffect(() => {
    if (nearestIdx < 0) return;
    clearReturnTimer();
    const el = containerRef.current;
    if (!el) return;
    const idealCenter = nearestIdx * ITEM_WIDTH + ITEM_WIDTH / 2 - containerWidth / 2;
    const target = Math.max(0, Math.min(maxScroll, idealCenter));
    el.scrollLeft = target;
    setScrollLeft(target);
  }, [nearestIdx, containerWidth, maxScroll, clearReturnTimer]);

  // a couple seconds after the mouse leaves the graph, glide back to the
  // most recent step.
  useEffect(() => {
    if (hoveredStep != null) return;
    clearReturnTimer();
    returnTimerRef.current = setTimeout(() => {
      scrollToEnd(true);
    }, RETURN_TO_END_DELAY_MS);
    return clearReturnTimer;
  }, [hoveredStep, clearReturnTimer, scrollToEnd]);

  const onScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setScrollLeft(el.scrollLeft);
  }, []);

  const { startIdx, endIdx } = useMemo(() => {
    if (containerWidth === 0 || items.length === 0) return { startIdx: 0, endIdx: -1 };
    const start = Math.max(0, Math.floor(scrollLeft / ITEM_WIDTH) - RENDER_BUFFER);
    const end = Math.min(items.length - 1, Math.ceil((scrollLeft + containerWidth) / ITEM_WIDTH) + RENDER_BUFFER);
    return { startIdx: start, endIdx: end };
  }, [scrollLeft, containerWidth, items.length]);

  const visible: { item: StepThumbnail; idx: number }[] = [];
  for (let i = startIdx; i <= endIdx; i++) {
    visible.push({ item: items[i], idx: i });
  }

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 shrink-0">
      <div className="bg-gray-800 px-4 py-2 flex items-center justify-between">
        <h2 className="text-gray-100 text-sm font-medium">Trained images</h2>
        <span className="text-xs text-gray-500">
          {hoveredStep != null && nearestIdx >= 0
            ? `step ${items[nearestIdx].step.toLocaleString()}`
            : status === 'loading' && items.length === 0
              ? 'Loading...'
              : items.length > 0
                ? `${items.length.toLocaleString()} steps`
                : ''}
        </span>
      </div>

      {items.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-gray-500">
          <LuImageOff className="w-5 h-5 mb-1.5" />
          <div className="text-xs">
            {status === 'loading' ? 'Loading...' : 'No trained-image data yet (needs the live_learning_monitor extension installed).'}
          </div>
        </div>
      ) : (
        <div ref={setContainerRef} onScroll={onScroll} className="overflow-x-auto overflow-y-hidden" style={{ height: THUMB_SIZE + 16 }}>
          <div className="relative" style={{ width: totalWidth, height: THUMB_SIZE + 16 }}>
            {visible.map(({ item, idx }) => {
              // Compare by step, not by exact index: a step with multiple
              // files trained at it spans several consecutive slots, and
              // the whole cluster should highlight together, not just
              // whichever single slot the nearest-index search landed on.
              const isHovered = nearestIdx >= 0 && item.step === items[nearestIdx].step;
              return (
                <div
                  key={`${item.step}-${item.file_path}`}
                  className="absolute top-2"
                  style={{ left: idx * ITEM_WIDTH, width: THUMB_SIZE }}
                  title={`step ${item.step} ÔÇö ${basename(item.file_path)}`}
                >
                  <div
                    className={
                      'w-16 h-16 rounded-md overflow-hidden bg-gray-950 border transition-colors ' +
                      (isHovered ? 'border-orange-500 ring-2 ring-orange-500' : 'border-gray-800')
                    }
                  >
                    {/* No loading="lazy" here on purpose: this strip is already
                        windowed (only nearby items are ever mounted, see
                        startIdx/endIdx above), and native lazy-loading combined
                        with that virtualization plus the instant/programmatic
                        scrollLeft writes elsewhere in this file was silently
                        never firing the image load in some browsers. */}
                    {/* Not ?thumb=1: that pre-generated file is center-cropped
                        to a square by sharp()'s `fit: 'cover'` resize before
                        it's ever saved, so no CSS here can show more of the
                        frame than that crop kept. object-contain below
                        letterboxes the original instead of cropping it. */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`/api/img/${encodeURIComponent(item.file_path)}`}
                      alt={`step ${item.step}`}
                      decoding="async"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
