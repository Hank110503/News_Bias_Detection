import type { Stage1Result } from "../types/analysis";

type ObservationPanelProps = {
  observations: Stage1Result;
  sourceLabel?: string;
};

const observationItems: Array<{
  key: keyof Stage1Result;
  label: string;
}> = [
  { key: "visual_observation", label: "Visual observation" },
  { key: "textual_observation", label: "Textual observation" },
  { key: "joint_mechanism", label: "Joint mechanism" },
];

export function ObservationPanel({
  observations,
  sourceLabel,
}: ObservationPanelProps) {
  return (
    <div className="panel p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="panel-title">Intermediate Reasoning</p>
          <h2 className="mt-2 text-xl font-semibold text-slate-900">
            Stage 1 Observations
          </h2>
        </div>
        {sourceLabel ? (
          <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500">
            {sourceLabel}
          </div>
        ) : null}
      </div>
      <div className="space-y-4">
        {observationItems.map((item, index) => (
          <div
            key={item.key}
            className="flex gap-4 rounded-2xl border border-slate-200/80 bg-slate-50 p-4"
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-sm font-semibold text-white">
              {index + 1}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold text-slate-900">
                {item.label}
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                {observations[item.key]}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
