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

interface StepThumbnailRow {
  step: number;
  file_path: string;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  try {
    await fs.promises.access(logPath);
  } catch {
    return NextResponse.json({ items: [] });
  }

  const url = new URL(request.url);
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 200000), 500000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;

  const db = openDb(logPath);

  try {
    const tableExists = await all<{ name: string }>(
      db,
      `SELECT name FROM sqlite_master WHERE type='table' AND name='file_losses'`,
    );
    if (tableExists.length === 0) {
      return NextResponse.json({ items: [] });
    }

    // Every file trained at each step, not just one representative — a batch
    // can (and often does) train several files at the same step, and the
    // strip shows them as consecutive slots for that step rather than
    // collapsing to a single thumbnail.
    const rows = await all<StepThumbnailRow>(
      db,
      `
      SELECT step, file_path
      FROM file_losses
      WHERE (? IS NULL OR step > ?)
      ORDER BY step ASC, file_path ASC
      LIMIT ?
      `,
      [sinceStep, sinceStep, limit],
    );

    // resume detection for incremental polling, see the loss route: the
    // client holds the wall_time of its last step and reloads if it changed
    const wallTimeOf = async (step: number | null) => {
      if (step == null) return null;
      const r = await all<{ wall_time: number }>(db, `SELECT wall_time FROM steps WHERE step = ?`, [step]);
      return r.length ? r[0].wall_time : null;
    };
    return NextResponse.json({
      items: rows,
      since_wall_time: await wallTimeOf(sinceStep),
      last_wall_time: await wallTimeOf(rows.length ? rows[rows.length - 1].step : null),
    });
  } finally {
    await closeDb(db);
  }
}
