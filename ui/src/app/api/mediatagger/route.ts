import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { buildTrainingReport } from '@/server/trainingReport';

export const runtime = 'nodejs';

// Optional bridge to MediaTagger (a separate dataset tagging app) through its
// MCP server, e.g. MEDIATAGGER_MCP_URL=http://localhost:5150/mcp. Off unless that
// variable is set: most setups don't run MediaTagger next to AI Toolkit, and it
// reads the same data from the job folder anyway (insights/training_report.json).
//   GET                             {enabled}
//   POST {action: 'show', path}     select that image in the MediaTagger window
//   POST {action: 'import', jobID}  send the job's training report (flags per image)

const MCP_URL = process.env.MEDIATAGGER_MCP_URL || '';

export async function GET() {
  return NextResponse.json({ enabled: !!MCP_URL });
}

// MediaTagger's server is stateless Streamable HTTP: every JSON-RPC request is
// a POST on its own, answered as JSON or as one SSE "message" event.
async function rpc(method: string, params: any): Promise<any> {
  const res = await fetch(MCP_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' },
    body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw new Error(`MediaTagger answered ${res.status}`);
  const text = await res.text();
  const payload = (res.headers.get('content-type') ?? '').includes('text/event-stream')
    ? text
        .split('\n')
        .filter(l => l.startsWith('data:'))
        .map(l => l.slice(5).trim())
        .pop()
    : text;
  const msg = JSON.parse(payload ?? '{}');
  if (msg.error) throw new Error(msg.error.message ?? 'MCP error');
  return msg.result;
}

async function callTool(name: string, args: Record<string, unknown>) {
  await rpc('initialize', {
    protocolVersion: '2025-06-18',
    capabilities: {},
    clientInfo: { name: 'ai-toolkit', version: '1.0' },
  });
  const result = await rpc('tools/call', { name, arguments: args });
  const text = result?.content?.find((c: any) => c.type === 'text')?.text;
  if (result?.isError) throw new Error(text || 'MediaTagger tool failed');
  try {
    return text ? JSON.parse(text) : result;
  } catch {
    return { text };
  }
}

export async function POST(request: NextRequest) {
  if (!MCP_URL) {
    return NextResponse.json(
      { error: 'MediaTagger bridge is off (set MEDIATAGGER_MCP_URL to enable it).' },
      { status: 404 },
    );
  }
  let body: { action?: string; path?: string; jobID?: string } = {};
  try {
    body = await request.json();
  } catch {
    // validated below
  }
  try {
    if (body.action === 'show' && body.path) {
      return NextResponse.json(await callTool('show_file', { path: body.path }));
    }
    if (body.action === 'import' && body.jobID) {
      const job = await prisma.job.findUnique({ where: { id: body.jobID } });
      if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
      const report = await buildTrainingReport(path.join(await getTrainingFolder(), job.name), job.name);
      return NextResponse.json(await callTool('import_training_report', { report_json: JSON.stringify(report) }));
    }
    return NextResponse.json({ error: "Use {action: 'show', path} or {action: 'import', jobID}" }, { status: 400 });
  } catch (e: any) {
    const offline = /fetch failed|ECONNREFUSED|timeout/i.test(String(e?.message ?? e) + String(e?.cause ?? ''));
    return NextResponse.json(
      {
        error: offline
          ? `MediaTagger is not reachable at ${MCP_URL}. Start MediaTagger and turn on MCP in its toolbar.`
          : (e?.message ?? String(e)),
      },
      { status: 502 },
    );
  }
}
