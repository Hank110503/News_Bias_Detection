import { MOCK_CASES } from "../data/mockCases";
import { MOCK_METRICS } from "../data/mockMetrics";
import type { CaseRecord, EvaluationDataset, EvaluationMetrics } from "../types/analysis";

let metricsPromise: Promise<EvaluationMetrics> | null = null;
let casesPromise: Promise<CaseRecord[]> | null = null;

async function fetchJson<T>(path: string) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }

  return (await response.json()) as T;
}

export async function loadEvaluationMetrics() {
  metricsPromise ??= fetchJson<EvaluationMetrics>("/demo/eval_metrics.json").catch(
    () => MOCK_METRICS,
  );
  return metricsPromise;
}

export async function loadCases() {
  casesPromise ??= fetchJson<EvaluationDataset>("/demo/eval_cases.json")
    .then((payload) => payload.cases)
    .catch(() => MOCK_CASES);
  return casesPromise;
}
