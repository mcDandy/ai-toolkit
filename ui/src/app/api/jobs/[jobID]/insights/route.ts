import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

// Serves the JSON files written by extensions/training_insights: the PCA'd
// weight trajectory and the CLIP sample/dataset map. Either may be null when
// the run hasn't produced it yet (or the extension isn't installed).
async function readJson(file: string) {
  try {
    return JSON.parse(await fs.promises.readFile(file, 'utf-8'));
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const insightsFolder = path.join(trainingFolder, job.name, 'insights');

  // ?parts=memorized: just the memorised list (Live Batch polls it), without
  // the much larger map/trajectory files
  if (new URL(request.url).searchParams.get('parts') === 'memorized') {
    return NextResponse.json({ memorized: await readJson(path.join(insightsFolder, 'memorized.json')) });
  }

  const [trajectory, embeddingMap, memorized, capacity, captionProbe, events] = await Promise.all([
    readJson(path.join(insightsFolder, 'trajectory.json')),
    readJson(path.join(insightsFolder, 'embedding_map.json')),
    readJson(path.join(insightsFolder, 'memorized.json')),
    readJson(path.join(insightsFolder, 'capacity.json')),
    readJson(path.join(insightsFolder, 'caption_probe.json')),
    readJson(path.join(insightsFolder, 'events.json')),
  ]);

  return NextResponse.json({ trajectory, embeddingMap, memorized, capacity, captionProbe, events });
}
