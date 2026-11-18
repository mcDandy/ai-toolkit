import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close(err => (err ? reject(err) : resolve()));
  });
}

interface FileLossSummaryRow {
  file_path: string;
  count: number;
  avg_loss: number | null;
  min_loss: number | null;
  max_loss: number | null;
  last_loss: number | null;
  last_step: number | null;
  sum_x: number | null;
  sum_y: number | null;
  sum_xy: number | null;
  sum_xx: number | null;
  min_step: number | null;
  max_step: number | null;
  avg_rel_loss: number | null;
}

// A file needs at least this many logged steps before its regression line
// (and therefore its score/category) is trusted rather than left null —
// below this, a "trend" is mostly noise.
const MIN_POINTS_FOR_FIT = 4;

// A score smaller than this (as a fraction of the file's average loss)
// counts as "flat" rather than improving/worsening, for category purposes
// only — the numeric score itself is never clamped to zero.
const FLAT_THRESHOLD = 0.05;

interface Fit {
  // predicted loss change over the observed step range (best-fit line, first
  // step to last), as a fraction of this file's *average* loss rather than
  // its fitted starting point. Dividing by the average instead of a single
  // (noisy) endpoint keeps the score stable for files that are already
  // learned and sitting near zero loss, where a tiny absolute wobble used to
  // read as a huge swing.
  score: number | null;
  slope: number | null;
  intercept: number | null;
}

// Ordinary least squares over (step, loss), via the closed-form sums SQL
// already computed per file. Same shape of result as before (a "did it get
// better" percentage) but derived from every point on the line instead of a
// crude first-half-vs-second-half average, so it's far less noisy.
function computeFit(r: FileLossSummaryRow): Fit {
  const { count: n, sum_x, sum_y, sum_xy, sum_xx, min_step, max_step, avg_loss } = r;
  if (
    n < MIN_POINTS_FOR_FIT ||
    sum_x === null ||
    sum_y === null ||
    sum_xy === null ||
    sum_xx === null ||
    min_step === null ||
    max_step === null ||
    avg_loss === null ||
    avg_loss <= 0
  ) {
    return { score: null, slope: null, intercept: null };
  }

  const denom = n * sum_xx - sum_x * sum_x;
  if (denom === 0) return { score: null, slope: null, intercept: null };

  const slope = (n * sum_xy - sum_x * sum_y) / denom;
  const intercept = (sum_y - slope * sum_x) / n;

  const fittedStart = intercept + slope * min_step;
  const fittedEnd = intercept + slope * max_step;
  const score = (fittedStart - fittedEnd) / avg_loss;

  return { score, slope, intercept };
}

type Category = 'learning' | 'worsening' | 'stuck' | 'exhausted';

// Secondary, informational-only label. Slope says whether it's moving;
// comparing this file's average loss to the job-wide average says whether
// "not moving" means it already learned it (exhausted) or never started
// (stuck) — slope alone can't tell those apart.
function computeCategory(score: number | null, avgLoss: number | null, jobAvgLoss: number | null): Category | null {
  if (score === null || avgLoss === null || jobAvgLoss === null) return null;
  if (score > FLAT_THRESHOLD) return 'learning';
  if (score < -FLAT_THRESHOLD) return 'worsening';
  return avgLoss > jobAvgLoss ? 'stuck' : 'exhausted';
}

interface FileLossPointRow {
  step: number;
  loss: number | null;
}

function csvCell(value: string | number): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

const SUMMARY_CSV_HEADER = 'file_path,count,avg_loss,min_loss,max_loss,last_loss,last_step,score,category,difficulty';
const SERIES_CSV_HEADER = 'step,loss';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  const url = new URL(request.url);
  const format = url.searchParams.get('format');
  const filePath = url.searchParams.get('file');
  // adjusted=1: use the noise-adjusted loss (loss / the run's typical loss at
  // that timestep) instead of the raw loss, which mostly reflects which
  // timestep a file happened to be trained at
  const wantAdjusted = url.searchParams.get('adjusted') === '1';

  const emptyResponse = () => {
    if (format === 'csv') {
      const header = filePath ? SERIES_CSV_HEADER : SUMMARY_CSV_HEADER;
      const suffix = filePath ? `-${path.basename(filePath)}` : '';
      return new NextResponse(header + '\n', {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': `attachment; filename="${job.name}${suffix}-file-loss.csv"`,
        },
      });
    }
    return NextResponse.json(filePath ? { file_path: filePath, points: [] } : { rows: [] });
  };

  // view=weighting: the adaptive_weighting extension's current per-file
  // weights, kept in the weighting_* tables inside loss_log.db. Null when
  // the feature has never run for this job.
  if (url.searchParams.get('view') === 'weighting') {
    interface WStateRow {
      file_path: string;
      category: string | null;
      multiplier: number | null;
      stuck: number | null;
      easy_streak: number | null;
      boost_streak: number | null;
      last_mean: number | null;
    }
    try {
      await fs.promises.access(logPath);
      const wdb = openDb(logPath);
      try {
        const hasTable = await all<{ name: string }>(
          wdb,
          `SELECT name FROM sqlite_master WHERE type='table' AND name='weighting_state'`,
        );
        if (hasTable.length === 0) return NextResponse.json(null);
        const rows = await all<WStateRow>(
          wdb,
          `SELECT file_path, category, multiplier, stuck, easy_streak, boost_streak, last_mean
           FROM weighting_state`,
        );
        if (rows.length === 0) return NextResponse.json(null);
        const hist = await all<{ file_path: string; step: number; category: string; mult: number }>(
          wdb,
          `SELECT file_path, step, category, mult FROM weighting_history ORDER BY step`,
        );
        const meta = await all<{ key: string; value: string }>(wdb, `SELECT key, value FROM weighting_meta`);
        const files: Record<
          string,
          {
            category: string;
            multiplier: number;
            stuck: number;
            easy_streak: number;
            boost_streak: number;
            last_mean: number | null;
            history: [number, string, number][];
          }
        > = {};
        for (const r of rows) {
          files[r.file_path] = {
            category: r.category ?? 'warming',
            multiplier: r.multiplier ?? 1,
            stuck: r.stuck ?? 0,
            easy_streak: r.easy_streak ?? 0,
            boost_streak: r.boost_streak ?? 0,
            last_mean: r.last_mean,
            history: [],
          };
        }
        for (const h of hist) {
          files[h.file_path]?.history.push([h.step, h.category, h.mult]);
        }
        const metaMap = Object.fromEntries(meta.map(m => [m.key, m.value]));
        return NextResponse.json({
          updated_step: Number(metaMap.updated_step ?? 0),
          window: metaMap.window ? Number(metaMap.window) : null,
          files,
        });
      } finally {
        await closeDb(wdb).catch(() => undefined);
      }
    } catch {
      return NextResponse.json(null);
    }
  }

  try {
    await fs.promises.access(logPath);
  } catch {
    return emptyResponse();
  }

  const db = openDb(logPath);
  try {
    const tableExists = await all<{ name: string }>(
      db,
      `SELECT name FROM sqlite_master WHERE type='table' AND name='file_losses'`,
    );
    if (tableExists.length === 0) {
      return emptyResponse();
    }
    // older databases have no rel_loss column, or only rows logged before it
    const columns = await all<{ name: string }>(db, `PRAGMA table_info(file_losses)`);
    const hasRel =
      columns.some(c => c.name === 'rel_loss') &&
      (await all(db, `SELECT 1 FROM file_losses WHERE rel_loss IS NOT NULL LIMIT 1`)).length > 0;
    const adjusted = wantAdjusted && hasRel;
    const hasBucket = columns.some(c => c.name === 'width');

    // view=buckets: noise-adjusted difficulty per training resolution. A bucket
    // that is consistently harder than the rest points at resolution/cropping,
    // not captions.
    if (url.searchParams.get('view') === 'buckets') {
      if (!hasBucket || !hasRel) return NextResponse.json({ buckets: [] });
      const buckets = await all<{ width: number; height: number; samples: number; files: number; difficulty: number }>(
        db,
        `SELECT width, height, COUNT(*) AS samples, COUNT(DISTINCT file_path) AS files, AVG(rel_loss) AS difficulty
         FROM file_losses WHERE rel_loss IS NOT NULL AND width IS NOT NULL
         GROUP BY width, height ORDER BY samples DESC`,
      );
      return NextResponse.json({ buckets });
    }
    // view=noise: run-level noise insight, aggregated on the fly from the
    // per-step rows already stored — which timestep values have actually been
    // trained (coverage histogram over 20 bins) and how the noise-adjusted
    // loss evolved in each of the 5 noise bands (step x band heatmap).
    if (url.searchParams.get('view') === 'noise') {
      const range = await all<{ lo: number | null; hi: number | null }>(
        db,
        `SELECT MIN(step) AS lo, MAX(step) AS hi FROM file_losses`,
      );
      const lo = range[0]?.lo ?? 0;
      const hi = range[0]?.hi ?? 0;
      const coverage = await all<{ bin: number; n: number }>(
        db,
        `SELECT MIN(19, MAX(0, CAST(timestep * 20 / 1000 AS INTEGER))) AS bin, COUNT(*) AS n
         FROM file_losses WHERE timestep IS NOT NULL
         GROUP BY bin ORDER BY bin`,
      );
      const HEATMAP_BINS = 40;
      const heatmap = hasRel
        ? await all<{ sb: number; band: number; v: number; n: number }>(
            db,
            // MIN() keeps the newest step inside the last bin (step == hi
            // would otherwise bin to `bins` itself)
            `SELECT MIN(${HEATMAP_BINS - 1}, CAST((step - ?) * ? / MAX(1, ? - ?) AS INTEGER)) AS sb,
                    MIN(4, MAX(0, CAST(timestep * 5 / 1000 AS INTEGER))) AS band,
                    AVG(rel_loss) AS v, COUNT(*) AS n
             FROM file_losses WHERE rel_loss IS NOT NULL AND timestep IS NOT NULL
             GROUP BY sb, band`,
            [lo, HEATMAP_BINS, hi, lo],
          )
        : [];
      return NextResponse.json({
        coverage,
        heatmap,
        min_step: range[0]?.lo ?? null,
        max_step: range[0]?.hi ?? null,
        heatmap_bins: HEATMAP_BINS,
      });
    }
    const valueCol = adjusted ? 'rel_loss' : 'loss';

    // Single-file mode: raw per-step series, for a loss-over-time trend view.
    if (filePath) {
      const points = await all<FileLossPointRow>(
        db,
        `SELECT step, ${valueCol} AS loss FROM file_losses WHERE file_path = ? AND ${valueCol} IS NOT NULL ORDER BY step ASC`,
        [filePath],
      );

      if (format === 'csv') {
        const lines = points.map(p => [p.step, p.loss].map(v => csvCell(v ?? '')).join(','));
        const csv = [SERIES_CSV_HEADER, ...lines].join('\n') + '\n';
        return new NextResponse(csv, {
          headers: {
            'Content-Type': 'text/csv',
            'Content-Disposition': `attachment; filename="${job.name}-${path.basename(filePath)}-loss.csv"`,
          },
        });
      }

      return NextResponse.json({ file_path: filePath, adjusted, points });
    }

    // Summary mode: aggregate stats per file, computed from the raw per-step
    // rows. sum_x/sum_y/sum_xy/sum_xx are the plain sums a closed-form OLS
    // regression needs — cheaper to compute here as aggregates than to pull
    // every row back and fit client-side.
    const rows = await all<FileLossSummaryRow>(
      db,
      `
      WITH scored AS (
        SELECT file_path, step, ${valueCol} AS loss, ${hasRel ? 'rel_loss' : 'NULL'} AS rel,
               ROW_NUMBER() OVER (PARTITION BY file_path ORDER BY step DESC) AS rn
        FROM file_losses
        WHERE ${valueCol} IS NOT NULL
      )
      SELECT
        file_path,
        COUNT(*) AS count,
        AVG(loss) AS avg_loss,
        MIN(loss) AS min_loss,
        MAX(loss) AS max_loss,
        MAX(CASE WHEN rn = 1 THEN loss END) AS last_loss,
        MAX(CASE WHEN rn = 1 THEN step END) AS last_step,
        SUM(step) AS sum_x,
        SUM(loss) AS sum_y,
        SUM(step * loss) AS sum_xy,
        SUM(step * step) AS sum_xx,
        MIN(step) AS min_step,
        MAX(step) AS max_step,
        AVG(rel) AS avg_rel_loss
      FROM scored
      GROUP BY file_path
      `,
    );

    // job-wide average loss, used only to tell "flat and already low"
    // (exhausted) apart from "flat and still high" (stuck) — a fit's slope
    // alone can't distinguish those two.
    const withAvg = rows.filter(r => r.avg_loss !== null);
    const jobAvgLoss =
      withAvg.length > 0 ? withAvg.reduce((s, r) => s + (r.avg_loss as number), 0) / withAvg.length : null;

    // per file: noise-adjusted difficulty in each of 5 noise bands (low noise first)
    const bandsByFile: Record<string, (number | null)[]> = {};
    if (hasRel) {
      const bandRows = await all<{ file_path: string; b: number; v: number }>(
        db,
        `SELECT file_path, MIN(4, MAX(0, CAST(timestep * 5 / 1000 AS INTEGER))) AS b, AVG(rel_loss) AS v
         FROM file_losses WHERE rel_loss IS NOT NULL AND timestep IS NOT NULL
         GROUP BY file_path, b`,
      );
      for (const r of bandRows) {
        (bandsByFile[r.file_path] ??= [null, null, null, null, null])[r.b] = r.v;
      }
    }

    const shaped = rows.map(r => {
      const fit = computeFit(r);
      return {
        file_path: r.file_path,
        count: r.count,
        avg_loss: r.avg_loss,
        min_loss: r.min_loss,
        max_loss: r.max_loss,
        last_loss: r.last_loss,
        last_step: r.last_step,
        score: fit.score,
        category: computeCategory(fit.score, r.avg_loss, jobAvgLoss),
        // 1.0 = as hard as a typical sample at the same noise level
        difficulty: r.avg_rel_loss,
        bands: bandsByFile[r.file_path] ?? null,
      };
    });

    if (format === 'csv') {
      const lines = shaped.map(r =>
        [
          r.file_path,
          r.count,
          r.avg_loss,
          r.min_loss,
          r.max_loss,
          r.last_loss,
          r.last_step,
          r.score,
          r.category,
          r.difficulty,
        ]
          .map(v => csvCell(v ?? ''))
          .join(','),
      );
      const csv = [SUMMARY_CSV_HEADER, ...lines].join('\n') + '\n';
      return new NextResponse(csv, {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': `attachment; filename="${job.name}-file-loss.csv"`,
        },
      });
    }

    return NextResponse.json({ rows: shaped, adjusted, has_adjusted: hasRel });
  } finally {
    await closeDb(db);
  }
}
