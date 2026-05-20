import { CUE_GROUPS, CUE_META, CUE_ORDER } from "../config/cues";
import type {
  CaseRecord,
  CueCode,
  CueMap,
  FullAnalysisResult,
} from "../types/analysis";

export function getCueEntries(cues: CueMap) {
  return CUE_ORDER.map((cueCode) => ({
    cueCode,
    item: cues[cueCode],
  })).filter((entry) => entry.item);
}

export function getActiveCueCount(result: FullAnalysisResult) {
  return getCueEntries(result.final_result.cues).filter(
    (entry) => entry.item?.present,
  ).length;
}

export function getAverageCueScore(result: FullAnalysisResult) {
  const entries = getCueEntries(result.final_result.cues);
  if (!entries.length) {
    return 0;
  }

  const total = entries.reduce((sum, entry) => sum + (entry.item?.score ?? 0), 0);
  return total / entries.length;
}

export function getCueCountByGroup(result: FullAnalysisResult) {
  return CUE_GROUPS.map((group) => {
    const codes = CUE_ORDER.filter((cueCode) => CUE_META[cueCode].group === group.key);
    const activeCount = codes.filter(
      (cueCode) => result.final_result.cues[cueCode]?.present,
    ).length;

    return {
      ...group,
      codes,
      activeCount,
    };
  });
}

export function getActiveCueCodes(result: FullAnalysisResult) {
  return getCueEntries(result.final_result.cues)
    .filter((entry) => entry.item?.present)
    .map((entry) => entry.cueCode);
}

export function formatObservationSource(caseRecord: CaseRecord) {
  if (caseRecord.observationDisplaySource === "gold_reference") {
    return "Evaluation reference observations";
  }

  if (caseRecord.observationSource === "stage1_predicted") {
    return "Predicted Stage 1 observations";
  }

  return "Local sample observations";
}

export function pickRepresentativeCue(result: FullAnalysisResult): CueCode {
  const firstActive = getActiveCueCodes(result)[0];
  return firstActive ?? "T1_Agent_Label";
}
