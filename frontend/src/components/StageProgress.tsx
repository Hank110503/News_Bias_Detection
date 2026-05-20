export const STAGES = [
  { key: "boot", label: "Prepare runtime" },
  { key: "stage1", label: "Generate observations" },
  { key: "stage2", label: "Generate cues and consistency" },
  { key: "done", label: "Assemble final result" },
];

export type StageProgressItem = {
  key: string;
  label: string;
  detail: string;
  elapsedMs?: number;
};

type StageProgressProps = {
  currentStage: string;
  items: StageProgressItem[];
};

function formatDuration(elapsedMs?: number) {
  if (!elapsedMs) {
    return null;
  }

  if (elapsedMs < 1000) {
    return `${elapsedMs} ms`;
  }

  return `${(elapsedMs / 1000).toFixed(1)} s`;
}

export function StageProgress({ currentStage, items }: StageProgressProps) {
  return (
    <div className="panel p-5">
      <p className="panel-title mb-4">Run Progress</p>
      <div className="space-y-4">
        {items.map((stage, index) => {
          const isDone =
            currentStage === "done" ||
            STAGES.findIndex((item) => item.key === currentStage) > index;
          const isCurrent = stage.key === currentStage;
          const elapsed = formatDuration(stage.elapsedMs);

          return (
            <div key={stage.key} className="rounded-2xl bg-slate-50/90 p-4">
              <div className="flex items-start gap-3">
              <div
                className={`mt-1 flex h-6 w-6 items-center justify-center rounded-full border text-xs font-semibold ${
                  isDone
                    ? "border-slate-900 bg-slate-900 text-white"
                    : isCurrent
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-slate-300 bg-white text-slate-500"
                }`}
              >
                {isDone ? "✓" : index + 1}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-800">
                  <span>{stage.label}</span>
                  {elapsed ? (
                    <span className="rounded-full bg-white px-2 py-0.5 text-xs text-slate-500">
                      {elapsed}
                    </span>
                  ) : null}
                </div>
                <div className="mt-1 text-sm text-slate-500">
                  {stage.detail || "Waiting"}
                </div>
              </div>
            </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
