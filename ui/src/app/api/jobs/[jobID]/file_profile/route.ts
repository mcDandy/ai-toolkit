import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

// Everything both extensions know about one training image, for the image
// profile panel: per-step losses (raw, noise-adjusted, timestep, bucket) from
// live_learning_monitor's file_losses table, and from training_insights its
// place on the dataset map, the samples closest to it, near-duplicates, the
// capacity probe's per-image capture, the caption probe's numbers and whether
// it is in the memorised list.

const BANDS = 5;

function openDb(filename: string) {
  const db = new sqlite3.Database(filename, sqlite3.OPEN_READONLY);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => (err ? reject(err) : resolve(rows as T[])));
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => db.close(err => (err ? reject(err) : resolve())));
}

async function readJson(file: string) {
  try {
    return JSON.parse(await fs.promises.readFile(file, 'utf-8'));
  } catch {
    return null;
  }
}

interface Row {
  step: number;
  loss: number | null;
  timestep: number | null;
  rel_loss: number | null;
  width: number | null;
  height: number | null;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const filePath = new URL(request.url).searchParams.get('file');
  if (!filePath) return NextResponse.json({ error: 'file is required' }, { status: 400 });

  const jobFolder = path.join(await getTrainingFolder(), job.name);
  const insightsFolder = path.join(jobFolder, 'insights');
  const logPath = path.join(jobFolder, 'loss_log.db');

  // per-step history
  let rows: Row[] = [];
  try {
    await fs.promises.access(logPath);
    const db = openDb(logPath);
    try {
      const cols = (await all<{ name: string }>(db, `PRAGMA table_info(file_losses)`)).map(c => c.name);
      if (cols.length) {
        const pick = (c: string) => (cols.includes(c) ? c : `NULL AS ${c}`);
        rows = await all<Row>(
          db,
          `SELECT step, loss, ${pick('timestep')}, ${pick('rel_loss')}, ${pick('width')}, ${pick('height')}
           FROM file_losses WHERE file_path = ? ORDER BY step ASC`,
          [filePath],
        );
      }
    } finally {
      await closeDb(db);
    }
  } catch {
    // no log yet
  }

  const [embeddingMap, capacity, captionProbe, memorized] = await Promise.all([
    readJson(path.join(insightsFolder, 'embedding_map.json')),
    readJson(path.join(insightsFolder, 'capacity.json')),
    readJson(path.join(insightsFolder, 'caption_probe.json')),
    readJson(path.join(insightsFolder, 'memorized.json')),
  ]);

  const datasetIdx: number = embeddingMap?.dataset?.findIndex((d: any) => d.path === filePath) ?? -1;
  const known = rows.length > 0 || datasetIdx >= 0;

  // caption sidecar: only for an image this job really trained on
  let caption: string | null = null;
  if (known) {
    const stem = filePath.slice(0, filePath.length - path.extname(filePath).length);
    for (const ext of ['.txt', '.caption']) {
      try {
        caption = (await fs.promises.readFile(stem + ext, 'utf-8')).trim();
        break;
      } catch {
        // try the next one
      }
    }
  }

  // difficulty per noise band: hard only at high noise usually means the
  // caption/composition, hard only at low noise detail/quality/resolution
  const bandSum = new Array(BANDS).fill(0);
  const bandN = new Array(BANDS).fill(0);
  for (const r of rows) {
    if (r.rel_loss == null || r.timestep == null) continue;
    const b = Math.min(BANDS - 1, Math.max(0, Math.floor((r.timestep / 1000) * BANDS)));
    bandSum[b] += r.rel_loss;
    bandN[b] += 1;
  }
  const rel = rows.filter(r => r.rel_loss != null).map(r => r.rel_loss as number);
  const buckets = [...new Set(rows.filter(r => r.width && r.height).map(r => `${r.width}x${r.height}`))];

  const nearestSamples =
    datasetIdx >= 0
      ? (embeddingMap.samples ?? [])
          .filter((s: any) => s.nearest_dataset_idx === datasetIdx)
          .sort((a: any, b: any) => b.nearest_similarity - a.nearest_similarity)
          .slice(0, 12)
          .map((s: any) => ({ path: s.path, step: s.step, prompt_idx: s.prompt_idx, similarity: s.nearest_similarity }))
      : [];
  const duplicates =
    datasetIdx >= 0
      ? (embeddingMap.duplicates ?? [])
          .filter((d: number[]) => d[0] === datasetIdx || d[1] === datasetIdx)
          .map((d: number[]) => ({
            path: embeddingMap.dataset[d[0] === datasetIdx ? d[1] : d[0]].path,
            similarity: d[2],
          }))
      : [];

  return NextResponse.json({
    file_path: filePath,
    caption,
    times_trained: rows.length,
    difficulty: rel.length ? rel.reduce((a, b) => a + b, 0) / rel.length : null,
    bands: bandN.map((n, i) => (n ? bandSum[i] / n : null)),
    band_counts: bandN,
    buckets,
    history: rows.map(r => ({ step: r.step, loss: r.loss, rel_loss: r.rel_loss, timestep: r.timestep })),
    map_point: datasetIdx >= 0 ? embeddingMap.dataset[datasetIdx] : null,
    nearest_samples: nearestSamples,
    duplicates,
    capacity: capacity?.files?.[filePath] ?? null,
    caption_probe: captionProbe?.files?.[filePath] ?? null,
    memorized: (memorized?.files ?? []).find((f: any) => f.path === filePath) ?? null,
  });
}
