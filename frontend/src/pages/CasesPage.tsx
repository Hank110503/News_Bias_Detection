import { useEffect, useMemo, useState } from "react";
import { CUE_META, CUE_ORDER } from "../config/cues";
import { RelationshipBadge } from "../components/RelationshipBadge";
import { ResultOverview } from "../components/ResultOverview";
import { loadCases } from "../services/cases";
import type { CaseRecord } from "../types/analysis";
import {
  formatObservationSource,
  getActiveCueCount,
} from "../utils/analysis";
import { resolveCaseImageUrl } from "../utils/imagePreview";

export function CasesPage() {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [query, setQuery] = useState("");
  const [relationshipFilter, setRelationshipFilter] = useState("all");
  const [cueFilter, setCueFilter] = useState("all");
  const [presenceFilter, setPresenceFilter] = useState("all");
  const [selectedId, setSelectedId] = useState("");

  useEffect(() => {
    loadCases()
      .then((loadedCases) => {
        setCases(loadedCases);
        setSelectedId((current) => current || loadedCases[0]?.newsId || "");
      })
      .catch(() => setCases([]));
  }, []);

  const filteredCases = useMemo(() => {
    const normalized = query.trim().toLowerCase();

    return cases.filter((item) => {
      const relation =
        item.result.final_result.consistency_dimension.D1_Relationship_Type;
      const relationMatched =
        relationshipFilter === "all" || relation === relationshipFilter;
      const searchMatched =
        !normalized ||
        item.newsId.toLowerCase().includes(normalized) ||
        item.title.toLowerCase().includes(normalized) ||
        item.articleText.toLowerCase().includes(normalized);

      const cueItem =
        cueFilter === "all"
          ? null
          : item.result.final_result.cues[cueFilter as keyof typeof item.result.final_result.cues];

      const cueMatched =
        cueFilter === "all"
          ? true
          : presenceFilter === "all"
            ? Boolean(cueItem)
            : presenceFilter === "present"
              ? Boolean(cueItem?.present)
              : cueItem?.present === false;

      return relationMatched && searchMatched && cueMatched;
    });
  }, [cases, cueFilter, presenceFilter, query, relationshipFilter]);

  const selectedCase =
    filteredCases.find((item) => item.newsId === selectedId) ?? filteredCases[0] ?? null;
  const selectedCaseImageUrl = useMemo(
    () => resolveCaseImageUrl(selectedCase),
    [selectedCase],
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
      <section className="panel p-5">
        <div className="mb-5">
          <p className="panel-title">Case Library</p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-900">
            Replay real evaluation outputs
          </h2>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            These cases come from the local evaluation set and are used to validate the presentation layer, replay cues and consistency, and quickly load inputs into the analysis page.
          </p>
        </div>

        <div className="space-y-3">
          <input
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by news_id, title, or article text"
          />
          <select
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
            value={relationshipFilter}
            onChange={(event) => setRelationshipFilter(event.target.value)}
          >
            <option value="all">All relationship types</option>
            <option value="Reinforcing">Reinforcing</option>
            <option value="Contrastive">Contrastive</option>
            <option value="Supplementary">Supplementary</option>
          </select>
          <select
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
            value={cueFilter}
            onChange={(event) => setCueFilter(event.target.value)}
          >
            <option value="all">All cues</option>
            {CUE_ORDER.map((cueCode) => (
              <option key={cueCode} value={cueCode}>
                {cueCode} · {CUE_META[cueCode].label}
              </option>
            ))}
          </select>
          <select
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
            value={presenceFilter}
            onChange={(event) => setPresenceFilter(event.target.value)}
          >
            <option value="all">All presence states</option>
            <option value="present">present = true</option>
            <option value="absent">present = false</option>
          </select>
        </div>

        <div className="mt-4 rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-600">
          {filteredCases.length} / {cases.length} matching cases
        </div>

        <div className="mt-4 max-h-[760px] space-y-3 overflow-y-auto pr-1">
          {filteredCases.map((item) => (
            <button
              key={item.newsId}
              className={`w-full rounded-3xl border p-4 text-left transition ${
                item.newsId === selectedCase?.newsId
                  ? "border-slate-900 bg-slate-900 text-white"
                  : "border-slate-200 bg-slate-50 hover:bg-white"
              }`}
              onClick={() => setSelectedId(item.newsId)}
              type="button"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-semibold">#{item.newsId}</div>
                <div className="text-xs opacity-80">
                  {getActiveCueCount(item.result)} active cues
                </div>
              </div>
              <p className="mt-2 text-sm leading-6">{item.title}</p>
            </button>
          ))}
        </div>
      </section>

      <section className="min-w-0">
        {selectedCase ? (
          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
              <div className="panel p-5">
                <p className="panel-title mb-4">Case Context</p>
                <div className="rounded-3xl bg-[linear-gradient(135deg,#20304a,#485f84)] p-5 text-white">
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-200">
                    Image Input
                  </div>
                  {selectedCaseImageUrl ? (
                    <div className="mt-4 overflow-hidden rounded-3xl border border-white/10 bg-slate-950/25">
                      <img
                        src={selectedCaseImageUrl}
                        alt={selectedCase.title}
                        className="h-64 w-full object-contain"
                      />
                    </div>
                  ) : null}
                  <div className="mt-3 text-lg font-semibold">
                    {selectedCase.imageHint || "No image filename recorded"}
                  </div>
                  <p className="mt-4 text-sm leading-6 text-slate-100/90">
                    {selectedCaseImageUrl
                      ? "The original sample image is loaded from a repository-relative path and remains accessible as long as the repo structure stays the same."
                      : "The original sample image could not be previewed. Make sure the relative structure between data and frontend is unchanged."}
                  </p>
                  {selectedCaseImageUrl ? (
                    <div className="mt-4 inline-flex rounded-full bg-white/10 px-3 py-1 text-xs text-slate-100">
                      Local sample image connected
                    </div>
                  ) : null}
                  {selectedCase.imagePath ? (
                    <div className="mt-4 rounded-2xl bg-white/10 px-3 py-2 text-xs break-all text-slate-100">
                      {selectedCase.imagePath}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="panel p-5">
                <p className="panel-title mb-4">Article Text and Metadata</p>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">
                    {formatObservationSource(selectedCase)}
                  </span>
                  {selectedCase.sourceUrl ? (
                    <a
                      className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600 hover:bg-slate-200"
                      href={selectedCase.sourceUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Source URL
                    </a>
                  ) : null}
                </div>
                <p className="mt-4 max-h-[300px] overflow-y-auto whitespace-pre-wrap text-sm leading-7 text-slate-700">
                  {selectedCase.articleText || "The original article text could not be restored for this sample."}
                </p>
                <div className="mt-4">
                  <RelationshipBadge
                    value={
                      selectedCase.result.final_result.consistency_dimension
                        .D1_Relationship_Type
                    }
                  />
                </div>
              </div>
            </div>

            <ResultOverview
              result={selectedCase.result}
              observationSourceLabel={formatObservationSource(selectedCase)}
            />
          </div>
        ) : (
          <div className="panel p-8 text-sm text-slate-600">No matching case found.</div>
        )}
      </section>
    </div>
  );
}
