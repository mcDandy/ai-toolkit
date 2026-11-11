import path from 'path';
import fs from 'fs';
import sqlite3 from 'sqlite3';

// Per-image findings of a training run, for dataset tools (MediaTagger's
// "Import training report"): noise-adjusted difficulty from live_learning_monitor
// plus training_insights' memorised list, duplicates, caption probe and capacity
// probe, condensed into flags and a short note per file.
//
// Format: {source, job, step, generated_at, files: [{path, difficulty, bands,
// times_trained, flags: [...], notes}]}. Flags: hard, hard-composition,
// hard-detail, weak-caption, memorised, duplicate, adapter-limited.

export interface ReportFile {
  path: string;
  difficulty: number | null;
  bands: (number | null)[];
  times_trained: number;
  flags: string[];
  notes: string;
}

export interface TrainingReport {
  source: 'ai-toolkit';
  job: string;
  step: number;
  generated_at: string;
  files: ReportFile[];
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) =>
    db.all(sql, params, (err, rows) => (err ? reject(err) : resolve(rows as T[]))),
  );
}

async function readJson(file: string) {
  try {
    return JSON.parse(await fs.promises.readFile(file, 'utf-8'));
  } catch {
    return null;
  }
}

function median(xs: number[]) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

export async function buildTrainingReport(jobFolder: string, jobName: string): Promise<TrainingReport> {
  const files = new Map<string, ReportFile>();
  const get = (p: string) => {
    let f = files.get(p);
    if (!f)
      files.set(
        p,
        (f = {
          path: p,
          difficulty: null,
          bands: [null, null, null, null, null],
          times_trained: 0,
          flags: [],
          notes: '',
        }),
      );
    return f;
  };
  const bandCounts = new Map<string, number[]>();
  const notes = new Map<string, string[]>();
  const note = (p: string, t: string) => (notes.get(p) ?? notes.set(p, []).get(p)!).push(t);
  let step = 0;

  const logPath = path.join(jobFolder, 'loss_log.db');
  if (fs.existsSync(logPath)) {
    const db = new sqlite3.Database(logPath, sqlite3.OPEN_READONLY);
    try {
      const cols = (await all<{ name: string }>(db, `PRAGMA table_info(file_losses)`)).map(c => c.name);
      if (cols.length) {
        const hasRel = cols.includes('rel_loss');
        for (const r of await all<{ file_path: string; n: number; d: number | null; last: number }>(
          db,
          `SELECT file_path, COUNT(*) AS n, ${hasRel ? 'AVG(rel_loss)' : 'NULL'} AS d, MAX(step) AS last
           FROM file_losses GROUP BY file_path`,
        )) {
          const f = get(r.file_path);
          f.times_trained = r.n;
          f.difficulty = r.d;
          step = Math.max(step, r.last);
        }
        if (hasRel && cols.includes('timestep')) {
          for (const r of await all<{ file_path: string; b: number; v: number; n: number }>(
            db,
            `SELECT file_path, MIN(4, MAX(0, CAST(timestep * 5 / 1000 AS INTEGER))) AS b, AVG(rel_loss) AS v, COUNT(*) AS n
             FROM file_losses WHERE rel_loss IS NOT NULL AND timestep IS NOT NULL GROUP BY file_path, b`,
          )) {
            get(r.file_path).bands[r.b] = r.v;
            (bandCounts.get(r.file_path) ?? bandCounts.set(r.file_path, [0, 0, 0, 0, 0]).get(r.file_path)!)[r.b] = r.n;
          }
        }
      }
    } finally {
      await new Promise<void>(res => db.close(() => res()));
    }
  }

  // A single sample's loss is very noisy: only flag with enough of them (an image
  // seen 3 times in a run can't support "hard at composition" from one sample).
  const MIN_SAMPLES = 3;
  const MIN_PER_SIDE = 2;
  for (const f of files.values()) {
    if (f.difficulty != null && f.difficulty > 1.3 && f.times_trained >= MIN_SAMPLES) {
      f.flags.push('hard');
      note(f.path, `${f.difficulty.toFixed(2)}x as hard as a typical image at the same noise level.`);
    }
    const n = bandCounts.get(f.path) ?? [0, 0, 0, 0, 0];
    const lo = mean([f.bands[0], f.bands[1]].filter((v): v is number => v != null));
    const hi = mean([f.bands[3], f.bands[4]].filter((v): v is number => v != null));
    if (Number.isFinite(lo) && Number.isFinite(hi) && n[0] + n[1] >= MIN_PER_SIDE && n[3] + n[4] >= MIN_PER_SIDE) {
      if (hi > 1.25 && hi > lo + 0.15) {
        f.flags.push('hard-composition');
        note(f.path, 'Hard mainly at high noise (composition): check that the caption describes this image.');
      } else if (lo > 1.25 && lo > hi + 0.15) {
        f.flags.push('hard-detail');
        note(f.path, 'Hard mainly at low noise (fine detail): image quality, compression or resolution.');
      }
    }
  }

  const insights = path.join(jobFolder, 'insights');
  const [memorized, map, captions, capacity] = await Promise.all([
    readJson(path.join(insights, 'memorized.json')),
    readJson(path.join(insights, 'embedding_map.json')),
    readJson(path.join(insights, 'caption_probe.json')),
    readJson(path.join(insights, 'capacity.json')),
  ]);
  for (const m of memorized?.files ?? []) {
    get(m.path).flags.push('memorised');
    note(m.path, `Memorised: its loss is ${(m.vs_median * 100).toFixed(0)}% of the dataset median.`);
  }
  for (const [i, j, sim] of map?.duplicates ?? []) {
    const a = map.dataset[i]?.path,
      b = map.dataset[j]?.path;
    if (!a || !b) continue;
    for (const [p, other] of [
      [a, b],
      [b, a],
    ]) {
      if (!get(p).flags.includes('duplicate')) get(p).flags.push('duplicate');
      note(p, `Near-duplicate of ${path.basename(other)} (${sim.toFixed(3)}).`);
    }
  }
  for (const [p, c] of Object.entries<any>(captions?.files ?? {})) {
    if (c.n >= 2 && c.specificity != null && c.specificity < 0.005) {
      get(p).flags.push('weak-caption');
      note(p, `Its caption fits it barely better than another image's caption (${(c.specificity * 100).toFixed(1)}%).`);
    }
  }
  const capFiles = Object.entries<any>(capacity?.files ?? {}).filter(([, c]) => c.captured != null);
  const capMedian = median(capFiles.map(([, c]) => c.captured));
  for (const [p, c] of capFiles) {
    if (capFiles.length >= 4 && c.captured < 0.6 * capMedian) {
      get(p).flags.push('adapter-limited');
      note(
        p,
        `The adapter can follow ${(c.captured * 100).toFixed(0)}% of its gradient (median ${(capMedian * 100).toFixed(0)}%).`,
      );
    }
  }

  for (const f of files.values()) f.notes = (notes.get(f.path) ?? []).join(' ');
  return {
    source: 'ai-toolkit',
    job: jobName,
    step,
    generated_at: new Date().toISOString(),
    files: [...files.values()].sort((a, b) => (b.difficulty ?? 0) - (a.difficulty ?? 0)),
  };
}
