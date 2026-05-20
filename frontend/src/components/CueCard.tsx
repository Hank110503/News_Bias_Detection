import { CUE_GROUP_COLORS, CUE_META } from "../config/cues";
import type { CueCode, CueItem } from "../types/analysis";

type CueCardProps = {
  cueCode: CueCode;
  item: CueItem;
};

export function CueCard({ cueCode, item }: CueCardProps) {
  const meta = CUE_META[cueCode];
  const strength = Math.max(0, Math.min(2, item.score));

  return (
    <div
      className={`min-w-0 rounded-3xl border bg-gradient-to-br p-5 transition ${
        item.present
          ? "border-slate-900/15 shadow-panel ring-1 ring-slate-900/10"
          : "border-slate-200"
      } ${CUE_GROUP_COLORS[meta.group]}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {meta.group}
          </p>
          <h3 className="mt-2 text-base font-semibold text-slate-900">
            {cueCode}
          </h3>
          <p className="mt-1 text-sm text-slate-600">{meta.label}</p>
        </div>
        <div
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            item.present
              ? "bg-slate-900 text-white"
              : "bg-slate-100 text-slate-500"
          }`}
        >
          {item.present ? "Present" : "Absent"}
        </div>
      </div>
      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between text-xs font-medium text-slate-500">
          <span>Strength</span>
          <span>{item.score}</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[0, 1, 2].map((level) => (
            <div
              key={level}
              className={`h-2.5 rounded-full ${
                level < strength
                  ? "bg-slate-900"
                  : strength === 0 && level === 0
                    ? "bg-slate-200"
                    : "bg-white/80"
              }`}
            />
          ))}
        </div>
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-700">{item.reason}</p>
    </div>
  );
}
