'use client';

import { Job } from '@prisma/client';
import CurrentBatchViewer from './CurrentBatchViewer';
import FileLossTable from './FileLossTable';

interface Props {
  job: Job;
}

export default function LiveBatchPage({ job }: Props) {
  return (
    <div className="flex flex-col gap-4 h-full min-h-0">
      <div className="flex-1 min-h-0">
        <CurrentBatchViewer job={job} />
      </div>
      <div className="flex-1 min-h-0">
        <FileLossTable job={job} />
      </div>
    </div>
  );
}
