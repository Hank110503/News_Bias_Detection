import { Link } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { MetricCard } from "../components/MetricCard";
import { RelationshipBadge } from "../components/RelationshipBadge";
import { loadCases, loadEvaluationMetrics } from "../services/cases";
import type { CaseRecord, EvaluationMetrics } from "../types/analysis";
import {
  getActiveCueCount,
  pickRepresentativeCue,
} from "../utils/analysis";
import { CUE_META } from "../config/cues";

export function DashboardPage() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [cases, setCases] = useState<CaseRecord[]>([]);

  useEffect(() => {
    loadEvaluationMetrics().then(setMetrics).catch(() => setMetrics(null));
    loadCases().then(setCases).catch(() => setCases([]));
  }, []);

  const relationshipStats = useMemo(() => {
    const base = {
      Reinforcing: 0,
      Contrastive: 0,
      Supplementary: 0,
    };

    for (const item of cases) {
      const relation = item.result.final_result.consistency_dimension.D1_Relationship_Type;
      if (relation in base) {
        base[relation as keyof typeof base] += 1;
      }
    }

    const total = cases.length || 1;

    return Object.entries(base).map(([key, count]) => ({
      key,
      count,
      ratio: count / total,
    }));
  }, [cases]);

  const featuredCase = cases[0];
  const representativeCue = featuredCase
    ? pickRepresentativeCue(featuredCase.result)
    : null;

  return (
    <div className="space-y-6">
      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="panel overflow-hidden p-6">
          <p className="panel-title">Product Framing</p>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
            A single-sample demo for multimodal war news bias analysis
          </h2>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600">
            The core interaction is one image plus one passage, followed by a two-stage output:
            observations first, then 9 cues and a consistency judgment. This page summarizes the system and links into the main analysis flow.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link className="primary-action" to="/analysis">
              Start analysis
            </Link>
            <Link className="secondary-action" to="/cases">
              Browse cases
            </Link>
          </div>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {[
              "Enter image and article text",
              "Stage 1 generates observations",
              "Stage 2 produces cues and consistency",
            ].map((item, index) => (
              <div key={item} className="rounded-3xl bg-slate-50 p-4">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                  Step {index + 1}
                </div>
                <div className="mt-2 text-sm font-medium text-slate-900">{item}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel p-6">
          <p className="panel-title">Evaluation Set Size</p>
          <div className="mt-4 text-5xl font-semibold tracking-tight text-slate-900">
            {metrics?.matchedRows ?? "--"}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-600">
            The local demo is already wired to real evaluation outputs and article text for case replay and credibility checks.
          </p>
          <div className="mt-6 space-y-3">
            {relationshipStats.map((item) => (
              <div key={item.key}>
                <div className="mb-2 flex items-center justify-between text-sm text-slate-600">
                  <span>{item.key}</span>
                  <span>{item.count}</span>
                </div>
                <div className="h-2 rounded-full bg-slate-100">
                  <div
                    className="h-2 rounded-full bg-slate-900"
                    style={{ width: `${Math.max(6, item.ratio * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-5">
        <MetricCard
          label="JSON Parse Rate"
          value={metrics ? metrics.jsonParseRate.toFixed(2) : "--"}
          hint="Share of outputs that can be parsed as valid structured JSON."
        />
        <MetricCard
          label="Cue Presence F1"
          value={metrics ? metrics.cuePresentMacroF1.toFixed(4) : "--"}
          hint="Macro F1 for cue presence prediction."
        />
        <MetricCard
          label="Cue Score Accuracy"
          value={metrics ? metrics.cueScoreAccuracy.toFixed(4) : "--"}
          hint="Accuracy for the three-level cue strength score."
        />
        <MetricCard
          label="Cue Score MAE"
          value={metrics ? metrics.cueScoreMae.toFixed(4) : "--"}
          hint="Lower is better."
        />
        <MetricCard
          label="Relation Macro F1"
          value={metrics ? metrics.relationMacroF1.toFixed(4) : "--"}
          hint="Macro F1 for image-text relationship classification."
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="panel p-6">
          <p className="panel-title">Output Preview</p>
          {featuredCase ? (
            <div className="mt-4 space-y-4">
              <div className="rounded-3xl bg-slate-50 p-5">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                  Featured Sample
                </div>
                <div className="mt-2 text-lg font-semibold text-slate-900">
                  {featuredCase.title}
                </div>
                <div className="mt-4">
                  <RelationshipBadge
                    value={
                      featuredCase.result.final_result.consistency_dimension
                        .D1_Relationship_Type
                    }
                  />
                </div>
                <p className="mt-4 text-sm leading-6 text-slate-600">
                  {featuredCase.result.observations.joint_mechanism}
                </p>
              </div>
              {representativeCue ? (
                <div className="rounded-3xl border border-slate-200 bg-white p-5">
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    Representative Cue
                  </div>
                  <div className="mt-2 text-lg font-semibold text-slate-900">
                    {representativeCue} · {CUE_META[representativeCue].label}
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-600">
                    {
                      featuredCase.result.final_result.cues[representativeCue]
                        ?.reason
                    }
                  </p>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="mt-4 text-sm leading-6 text-slate-600">Loading sample data.</p>
          )}
        </div>

        <div className="panel p-6">
          <p className="panel-title">Case Entry Points</p>
          <div className="mt-4 space-y-4">
            {cases.slice(0, 3).map((item) => (
              <Link
                key={item.newsId}
                to="/cases"
                className="block rounded-3xl border border-slate-200 bg-slate-50 p-4 transition hover:bg-white"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold text-slate-900">
                    #{item.newsId}
                  </span>
                  <span className="text-xs text-slate-500">
                    {getActiveCueCount(item.result)} active cues
                  </span>
                </div>
                <div className="mt-2 text-sm leading-6 text-slate-700">
                  {item.title}
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
