import { CUE_GROUPS, CUE_META, CUE_ORDER } from "../config/cues";
import type { CueCode, FullAnalysisResult } from "../types/analysis";
import {
  getActiveCueCount,
  getAverageCueScore,
  getCueCountByGroup,
} from "../utils/analysis";
import { CueCard } from "./CueCard";
import { ObservationPanel } from "./ObservationPanel";
import { RelationshipBadge } from "./RelationshipBadge";

type ResultOverviewProps = {
  result: FullAnalysisResult;
  observationSourceLabel?: string;
};

export function ResultOverview({
  result,
  observationSourceLabel,
}: ResultOverviewProps) {
  const activeCueCount = getActiveCueCount(result);
  const meanScore = getAverageCueScore(result);
  const cueGroups = getCueCountByGroup(result);
  const relationship = result.final_result.consistency_dimension.D1_Relationship_Type;

  return (
    <div className="min-w-0 space-y-6">
      <div className="panel p-5">
        <p className="panel-title mb-4">Summary</p>
        <div className="grid gap-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="min-w-0 rounded-3xl bg-slate-100 p-5">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                Completion
              </div>
              <div className="mt-3 break-words text-xl font-semibold text-slate-900">
                Analysis complete
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                This result already includes observations, cues, and consistency.
              </p>
            </div>
            <div className="min-w-0 rounded-3xl bg-slate-100 p-5">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                Mean Score
              </div>
              <div className="mt-3 text-4xl font-semibold text-slate-900">
                {meanScore.toFixed(2)}
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                Average cue intensity across all 9 cues.
              </p>
            </div>
            <div className="min-w-0 rounded-3xl bg-slate-100 p-5">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                Active Cues
              </div>
              <div className="mt-3 text-4xl font-semibold text-slate-900">
                {activeCueCount}
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                Number of cues marked as active for this sample.
              </p>
            </div>
          </div>

          <div className="min-w-0 rounded-3xl bg-slate-950 p-5 text-white">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Consistency
            </div>
            <div className="mt-3">
              <RelationshipBadge value={relationship} />
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-200">
              {result.final_result.consistency_dimension.reason}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
        {cueGroups.map((group) => (
          <div key={group.key} className="panel min-w-0 p-5">
            <p className="panel-title">{group.label}</p>
            <div className="mt-3 flex items-end justify-between gap-3">
              <div className="text-3xl font-semibold text-slate-900">
                {group.activeCount}
              </div>
              <div className="text-sm text-slate-500">active cues</div>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              {group.description}
            </p>
          </div>
        ))}
      </div>

      {CUE_GROUPS.map((group) => {
        const groupCodes = CUE_ORDER.filter(
          (cueCode) => CUE_META[cueCode].group === group.key,
        );
        return (
          <div key={group.key} className="panel min-w-0 p-5">
            <div className="mb-5 flex items-end justify-between gap-3">
              <div>
                <p className="panel-title">{group.label}</p>
                <h2 className="mt-2 text-xl font-semibold text-slate-900">
                  {group.description}
                </h2>
              </div>
              <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500">
                {groupCodes.filter((cueCode) => result.final_result.cues[cueCode]).length} /{" "}
                {groupCodes.length}
              </div>
            </div>
            <div className="grid gap-4 2xl:grid-cols-2">
              {groupCodes.map((cueCode) => {
                const item = result.final_result.cues[cueCode];
                if (!item) {
                  return null;
                }

                return (
                  <CueCard
                    key={cueCode}
                    cueCode={cueCode as CueCode}
                    item={item}
                  />
                );
              })}
            </div>
          </div>
        );
      })}

      <ObservationPanel
        observations={result.observations}
        sourceLabel={observationSourceLabel}
      />

      <div className="panel p-5">
        <p className="panel-title mb-4">Raw JSON</p>
        <pre className="overflow-x-auto rounded-3xl bg-slate-950 p-4 font-mono text-xs leading-6 text-slate-100">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  );
}
