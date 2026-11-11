'use client';

export interface LossSparklinePoint {
  step: number;
  loss: number | null;
}

interface Props {
  points: LossSparklinePoint[];
  width?: number;
  height?: number;
}

// Light causal EMA purely to make the shape readable ÔÇö per-sample diffusion
// loss is noisy enough step-to-step that a raw polyline is just static.
function smooth(values: number[], alpha = 0.15): number[] {
  const out: number[] = [];
  let s = values[0] ?? 0;
  for (let i = 0; i < values.length; i++) {
    s = i === 0 ? values[i] : alpha * values[i] + (1 - alpha) * s;
    out.push(s);
  }
  return out;
}

// Ordinary least squares over (step, loss) ÔÇö same closed-form fit the
// per-file "score" in the table below is computed from, so the line drawn
// here matches the number shown there.
function linearFit(steps: number[], values: number[]): { slope: number; intercept: number } | null {
  const n = steps.length;
  if (n < 2) return null;
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += steps[i];
    sumY += values[i];
    sumXY += steps[i] * values[i];
    sumXX += steps[i] * steps[i];
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

export default function LossSparkline({ points, width = 560, height = 90 }: Props) {
  const clean = points.filter(p => p.loss !== null && Number.isFinite(p.loss)) as { step: number; loss: number }[];

  if (clean.length < 2) {
    return (
      <div className="text-xs text-gray-500 italic py-4 text-center" style={{ width }}>
        Not enough points yet to draw a trend.
      </div>
    );
  }

  const rawValues = clean.map(p => p.loss);
  const smoothed = smooth(rawValues);
  const fit = linearFit(
    clean.map(p => p.step),
    rawValues,
  );

  const minStep = clean[0].step;
  const maxStep = clean[clean.length - 1].step;
  const stepRange = Math.max(1, maxStep - minStep);

  const fittedStart = fit ? fit.intercept + fit.slope * minStep : null;
  const fittedEnd = fit ? fit.intercept + fit.slope * maxStep : null;

  // include the fitted endpoints in the y-domain so a steep trend line never
  // gets clipped against the raw min/max.
  const domainValues = fittedStart !== null && fittedEnd !== null ? [...rawValues, fittedStart, fittedEnd] : rawValues;
  const minLoss = Math.min(...domainValues);
  const maxLoss = Math.max(...domainValues);
  const lossRange = maxLoss - minLoss || 1;

  const pad = 6;
  const toX = (step: number) => pad + ((step - minStep) / stepRange) * (width - pad * 2);
  const toY = (loss: number) => height - pad - ((loss - minLoss) / lossRange) * (height - pad * 2);

  const rawPath = clean.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.step).toFixed(1)} ${toY(p.loss).toFixed(1)}`).join(' ');
  const smoothPath = clean
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.step).toFixed(1)} ${toY(smoothed[i]).toFixed(1)}`)
    .join(' ');
  const fitPath =
    fittedStart !== null && fittedEnd !== null
      ? `M ${toX(minStep).toFixed(1)} ${toY(fittedStart).toFixed(1)} L ${toX(maxStep).toFixed(1)} ${toY(fittedEnd).toFixed(1)}`
      : null;

  return (
    <div>
      <svg width={width} height={height} className="block">
        <path d={rawPath} fill="none" stroke="rgba(148,163,184,0.35)" strokeWidth={1} />
        <path d={smoothPath} fill="none" stroke="rgba(57,135,229,1)" strokeWidth={2} />
        {fitPath && (
          <path d={fitPath} fill="none" stroke="rgba(230,179,38,0.9)" strokeWidth={1.5} strokeDasharray="5 4" />
        )}
      </svg>
      <div className="flex items-center justify-between text-[10px] text-gray-500 mt-0.5" style={{ width }}>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-0.5 bg-blue-400" /> smoothed
          </span>
          {fitPath && (
            <span className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-0.5" style={{ background: 'rgba(230,179,38,0.9)' }} /> best fit
            </span>
          )}
        </span>
        <span>
          min {minLoss.toPrecision(3)} ┬Ě max {maxLoss.toPrecision(3)}
        </span>
      </div>
      <div className="flex justify-between text-[10px] text-gray-600 mt-0.5" style={{ width }}>
        <span>step {minStep.toLocaleString()}</span>
        <span>step {maxStep.toLocaleString()}</span>
      </div>
    </div>
  );
}
