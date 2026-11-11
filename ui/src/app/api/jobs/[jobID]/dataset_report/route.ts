import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { buildTrainingReport } from '@/server/trainingReport';

export const runtime = 'nodejs';

// GET: the job's per-image training report, as a download (e.g. for MediaTagger's
// Dataset → Import training report). The training_insights extension writes it
// into the job folder itself (insights/training_report.json); jobs from before
// that are built here (server/trainingReport.ts, same format).
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const jobFolder = path.join(await getTrainingFolder(), job.name);
  let body: string;
  try {
    body = await fs.promises.readFile(path.join(jobFolder, 'insights', 'training_report.json'), 'utf-8');
  } catch {
    body = JSON.stringify(await buildTrainingReport(jobFolder, job.name), null, 1);
  }
  return new NextResponse(body, {
    headers: {
      'Content-Type': 'application/json',
      'Content-Disposition': `attachment; filename="${job.name}-training-report.json"`,
    },
  });
}
