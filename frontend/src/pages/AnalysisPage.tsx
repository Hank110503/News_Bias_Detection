import { useEffect, useMemo, useRef, useState } from "react";
import { ResultOverview } from "../components/ResultOverview";
import { STAGES, StageProgress } from "../components/StageProgress";
import {
  canUseLocalInferenceBridge,
  getLocalInferenceMode,
} from "../services/localBridge";
import {
  buildRecentRecordTitle,
  createRecentRecord,
  loadAnalysisDraft,
  loadRecentAnalyses,
  runAnalysisStream,
  saveAnalysisDraft,
  saveRecentAnalysis,
} from "../services/analysis";
import { loadCases } from "../services/cases";
import type { CaseRecord, FullAnalysisResult, RecentAnalysisRecord } from "../types/analysis";
import {
  getActiveCueCount,
  getAverageCueScore,
} from "../utils/analysis";
import {
  isImagePath,
  resolveCaseImageUrl,
  resolveImageUrl,
} from "../utils/imagePreview";

type StageRuntime = Record<
  string,
  {
    detail: string;
    startedAt?: number;
    completedAt?: number;
  }
>;

type ImagePreviewState = {
  url: string;
  kind: "none" | "upload" | "sample" | "path";
  label: string;
  note: string;
  sizeLabel?: string;
};

type ImageInfo = {
  width: number;
  height: number;
};

const EMPTY_PREVIEW: ImagePreviewState = {
  url: "",
  kind: "none",
  label: "",
  note: "",
};

function createInitialStageRuntime(): StageRuntime {
  return Object.fromEntries(
    STAGES.map((stage) => [
      stage.key,
      {
        detail: "Waiting",
      },
    ]),
  );
}

function createEmptyResult() {
  return null as FullAnalysisResult | null;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function readFileAsBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error("Unable to read the local image file."));
    };
    reader.onerror = () => {
      reject(new Error("Failed to read the local image file."));
    };
    reader.readAsDataURL(file);
  });
}

function loadImageElementFromUrl(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode the selected image in the browser."));
    image.src = url;
  });
}

async function convertImageFileToPngBase64(file: File) {
  const objectUrl = URL.createObjectURL(file);

  try {
    const image = await loadImageElementFromUrl(objectUrl);
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Failed to prepare a canvas for image conversion.");
    }

    context.drawImage(image, 0, 0);
    return canvas.toDataURL("image/png");
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function prepareImagePayloadForInference(file: File) {
  const requiresBrowserTranscode =
    file.type === "image/avif" || /\.avif$/i.test(file.name);

  if (requiresBrowserTranscode) {
    return {
      imageDataBase64: await convertImageFileToPngBase64(file),
      imageName: file.name.replace(/\.[^.]+$/, "") + ".png",
    };
  }

  return {
    imageDataBase64: await readFileAsBase64(file),
    imageName: file.name,
  };
}

export function AnalysisPage() {
  const draft = useMemo(() => loadAnalysisDraft(), []);
  const localInferenceMode = useMemo(() => getLocalInferenceMode(), []);
  const localBridgeEnabled = useMemo(() => canUseLocalInferenceBridge(), []);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [articleText, setArticleText] = useState(draft.articleText);
  const [imagePath, setImagePath] = useState(draft.imagePath);
  const [imageName, setImageName] = useState("");
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<ImagePreviewState>(EMPTY_PREVIEW);
  const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedSampleId, setSelectedSampleId] = useState(draft.sampleId);
  const [loadedSampleId, setLoadedSampleId] = useState(draft.sampleId);
  const [currentStage, setCurrentStage] = useState("boot");
  const [stageRuntime, setStageRuntime] = useState<StageRuntime>(
    createInitialStageRuntime(),
  );
  const [result, setResult] = useState<FullAnalysisResult | null>(createEmptyResult);
  const [resultSourceLabel, setResultSourceLabel] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<RecentAnalysisRecord[]>(() =>
    loadRecentAnalyses(),
  );
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    loadCases()
      .then((loadedCases) => {
        setCases(loadedCases);
        setSelectedSampleId((current) => current || loadedCases[0]?.newsId || "");
      })
      .catch(() => {
        setCases([]);
      });
  }, []);

  useEffect(() => {
    saveAnalysisDraft({
      articleText,
      imagePath,
      sampleId: selectedSampleId,
    });
  }, [articleText, imagePath, selectedSampleId]);

  useEffect(() => {
    if (!running) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 250);

    return () => {
      window.clearInterval(timer);
    };
  }, [running]);

  useEffect(() => {
    if (!imagePreview.url) {
      setImageInfo(null);
      return undefined;
    }

    const image = new Image();
    image.onload = () => {
      setImageInfo({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };
    image.onerror = () => {
      setImageInfo(null);
    };
    image.src = imagePreview.url;

    return () => {
      image.onload = null;
      image.onerror = null;
    };
  }, [imagePreview.url]);

  useEffect(() => {
    return () => {
      if (imagePreview.kind === "upload" && imagePreview.url) {
        URL.revokeObjectURL(imagePreview.url);
      }
    };
  }, [imagePreview]);

  const selectedSample =
    cases.find((item) => item.newsId === selectedSampleId) ?? null;
  const loadedSample =
    cases.find((item) => item.newsId === loadedSampleId) ?? null;

  const replacePreview = (next: ImagePreviewState) => {
    setImagePreview((prev) => {
      if (prev.kind === "upload" && prev.url && prev.url !== next.url) {
        URL.revokeObjectURL(prev.url);
      }
      return next;
    });
  };

  const clearSampleBinding = () => {
    setLoadedSampleId("");
  };

  const applySelectedFile = (file: File) => {
    clearSampleBinding();
    setSelectedImageFile(file);
    const objectUrl = URL.createObjectURL(file);
    replacePreview({
      url: objectUrl,
      kind: "upload",
      label: file.name,
      note: "Local image selected. The browser will pass the file contents to local inference directly instead of relying on a filesystem path.",
      sizeLabel: formatBytes(file.size),
    });
    setImageName(file.name);
    setImagePath("");
    resetRunState();
  };

  const previewImagePath = (pathValue: string) => {
    if (!pathValue || !pathValue.startsWith("/") || !isImagePath(pathValue)) {
      return;
    }

    clearSampleBinding();
    setSelectedImageFile(null);
    setImageName(pathValue.split("/").pop() ?? "");
    replacePreview({
      url: resolveImageUrl(pathValue),
      kind: "path",
      label: pathValue.split("/").pop() ?? pathValue,
      note: "Previewing the image from an absolute path. Direct local file access is only available in development mode.",
    });
    resetRunState();
  };

  const stageItems = useMemo(
    () =>
      STAGES.map((stage) => {
        const runtime = stageRuntime[stage.key];
        const elapsedMs = runtime?.startedAt
          ? (runtime.completedAt ?? now) - runtime.startedAt
          : undefined;

        return {
          key: stage.key,
          label: stage.label,
          detail: runtime?.detail ?? "Waiting",
          elapsedMs,
        };
      }),
    [now, stageRuntime],
  );

  const summary = useMemo(() => {
    const articleLength = articleText.trim().length;
    const imageState = imagePath || imageName ? "image provided" : "no image provided";
    if (!articleLength) {
      return localBridgeEnabled
        ? localInferenceMode === "tauri"
          ? "Enter a news passage and provide an image or image path. Desktop mode prioritizes the real local model bridge."
          : "Enter a news passage and provide an image or image path. Browser dev mode calls real local Python inference through the Vite bridge."
        : "Enter a news passage and provide an image or image path.";
    }

    return `Current article length: ${articleLength} characters, ${imageState}. ${
      localBridgeEnabled
        ? localInferenceMode === "tauri"
          ? "Running in desktop mode, so analysis prioritizes real local Python inference."
          : "Running in browser dev mode, so analysis uses the Vite local bridge for real Python inference."
        : loadedSample
          ? `The current input is bound to sample ${loadedSample.newsId}, so analysis will replay that sample result.`
          : "The current run will use the demo fallback logic."
    } `;
  }, [articleText, imageName, imagePath, loadedSample, localBridgeEnabled, localInferenceMode]);

  const stats = useMemo(() => {
    if (!result) {
      return null;
    }

    return {
      activeCueCount: getActiveCueCount(result),
      meanScore: getAverageCueScore(result),
      relationship: result.final_result.consistency_dimension.D1_Relationship_Type,
    };
  }, [result]);

  const resetRunState = () => {
    setCurrentStage("boot");
    setStageRuntime(createInitialStageRuntime());
    setResult(null);
    setResultSourceLabel("");
    setError("");
  };

  const markStageStarted = (stageKey: string, detail: string) => {
    const timestamp = Date.now();
    setStageRuntime((prev) => {
      const next: StageRuntime = { ...prev };
      const currentIndex = STAGES.findIndex((stage) => stage.key === stageKey);

      if (currentIndex > 0) {
        const previousKey = STAGES[currentIndex - 1]?.key;
        if (previousKey && next[previousKey]?.startedAt && !next[previousKey]?.completedAt) {
          next[previousKey] = {
            ...next[previousKey],
            completedAt: timestamp,
          };
        }
      }

      next[stageKey] = {
        detail,
        startedAt: next[stageKey]?.startedAt ?? timestamp,
        completedAt: undefined,
      };

      return next;
    });
  };

  const markDone = (detail: string) => {
    const timestamp = Date.now();
    setStageRuntime((prev) => ({
      ...prev,
      stage2: prev.stage2?.startedAt
        ? {
            ...prev.stage2,
            completedAt: prev.stage2.completedAt ?? timestamp,
            detail: "Stage 2 completed",
          }
        : prev.stage2,
      done: {
        detail,
        startedAt: prev.done?.startedAt ?? timestamp,
        completedAt: timestamp,
      },
    }));
  };

  const handleLoadSample = () => {
    if (!selectedSample) {
      return;
    }

    setSelectedImageFile(null);
    setArticleText(selectedSample.articleText);
    setImagePath(selectedSample.imagePath ?? selectedSample.imageHint);
    setImageName(selectedSample.imageHint);
    setLoadedSampleId(selectedSample.newsId);
    const samplePreviewUrl = resolveCaseImageUrl(selectedSample);
    replacePreview(
      samplePreviewUrl
        ? {
            url: samplePreviewUrl,
            kind: "sample",
            label: selectedSample.imageHint,
            note: "This preview comes from the sample image asset and is resolved from the repository-relative path.",
          }
        : EMPTY_PREVIEW,
    );
    resetRunState();
  };

  const handleRun = async () => {
    if (localBridgeEnabled && !selectedImageFile && !imagePath.trim()) {
      setError("Real local inference requires an image file or an accessible image path.");
      return;
    }

    resetRunState();
    setRunning(true);
    setNow(Date.now());

    try {
      const uploadedImagePayload = selectedImageFile
        ? await prepareImagePayloadForInference(selectedImageFile)
        : null;

      const input = {
        articleText,
        imagePath: selectedImageFile ? "" : imagePath,
        imageName: uploadedImagePayload?.imageName ?? imageName,
        imageDataBase64: uploadedImagePayload?.imageDataBase64,
        sampleCase: loadedSample,
      };

      for await (const event of runAnalysisStream(input)) {
        if (event.type === "status") {
          setCurrentStage(event.stage);
          markStageStarted(event.stage, event.message);
        }

        if (event.type === "partial") {
          setStageRuntime((prev) => ({
            ...prev,
            stage1: {
              ...prev.stage1,
              detail: "Stage 1 completed, observations received",
            },
          }));
          setResultSourceLabel(event.sourceLabel ?? "");
          setResult({
            observations: event.data,
            final_result: {
              cues: {},
              consistency_dimension: {
                reason: "",
                D1_Relationship_Type: "",
              },
            },
          });
        }

        if (event.type === "final") {
          setCurrentStage("done");
          markDone("Results assembled and ready to inspect");
          setResultSourceLabel(event.sourceLabel ?? "");
          setResult(event.data);
          const recentRecord = createRecentRecord(
            buildRecentRecordTitle(input, loadedSample),
            input,
            event.data,
          );
          setHistory(saveRecentAnalysis(recentRecord));
        }

        if (event.type === "error") {
          setError(event.message);
        }
      }
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Analysis failed.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <section className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        <section className="panel p-5">
          <div className="mb-5">
            <p className="panel-title">Input</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900">
              Single-image + single-text input
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              The demo centers on single-sample analysis. Sample cases are only used for quick replay and do not change the input format.
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Load sample
              </label>
              <div className="flex gap-2">
                <select
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
                  value={selectedSampleId}
                  onChange={(event) => setSelectedSampleId(event.target.value)}
                >
                  {cases.map((item) => (
                    <option key={item.newsId} value={item.newsId}>
                      {item.newsId} · {item.title}
                    </option>
                  ))}
                </select>
                <button
                  className="rounded-2xl border border-slate-300 px-4 py-3 text-sm font-medium text-slate-700 transition hover:bg-white"
                  onClick={handleLoadSample}
                  type="button"
                >
                  Load
                </button>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Image input
              </label>
              <input
                ref={fileInputRef}
                className="hidden"
                type="file"
                accept="image/*"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) {
                    return;
                  }

                  applySelectedFile(file);
                }}
              />
              <div
                className={`rounded-3xl border border-dashed px-4 py-5 transition ${
                  dragActive
                    ? "border-slate-900 bg-slate-100"
                    : "border-slate-300 bg-slate-50"
                }`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  const file = event.dataTransfer.files?.[0];
                  if (file && file.type.startsWith("image/")) {
                    applySelectedFile(file);
                  }
                }}
              >
                <div className="text-sm font-medium text-slate-800">
                  Click to choose an image, or drag one here
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Supports `png / jpg / jpeg / webp / gif`. A real preview appears immediately after selection.
                </p>
                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
                    onClick={() => fileInputRef.current?.click()}
                    type="button"
                  >
                    Choose image
                  </button>
                  <button
                    className="rounded-full border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-white"
                    onClick={() => {
                      replacePreview(EMPTY_PREVIEW);
                      setImageName("");
                      setImagePath("");
                      setSelectedImageFile(null);
                      clearSampleBinding();
                      resetRunState();
                      if (fileInputRef.current) {
                        fileInputRef.current.value = "";
                      }
                    }}
                    type="button"
                  >
                    Clear image
                  </button>
                </div>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Image path or filename
              </label>
              <div className="flex gap-2">
                <input
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
                  value={imagePath}
                  onChange={(event) => {
                    clearSampleBinding();
                    setSelectedImageFile(null);
                    setImageName("");
                    setImagePath(event.target.value);
                  }}
                  placeholder="For example: /opt/data/.../example.jpg"
                />
                <button
                  className="rounded-2xl border border-slate-300 px-4 py-3 text-sm font-medium text-slate-700 transition hover:bg-white"
                  onClick={() => previewImagePath(imagePath)}
                  type="button"
                >
                  Preview path
                </button>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Article text
              </label>
              <textarea
                className="min-h-[260px] w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none transition focus:border-slate-400"
                value={articleText}
                onChange={(event) => {
                  clearSampleBinding();
                  setArticleText(event.target.value);
                }}
                placeholder="Paste a news passage..."
              />
            </div>

            <div className="rounded-3xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">
              {summary}
            </div>

            {imagePreview.url ? (
              <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white">
                <img
                  src={imagePreview.url}
                  alt="input preview"
                  className="h-60 w-full bg-slate-100 object-contain"
                />
                <div className="border-t border-slate-200 px-4 py-3">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="rounded-full bg-slate-100 px-2 py-1">
                      {imagePreview.label}
                    </span>
                    <span className="rounded-full bg-slate-100 px-2 py-1">
                      {imagePreview.kind === "upload"
                        ? "Local upload"
                        : imagePreview.kind === "sample"
                          ? "Sample image"
                          : "Path preview"}
                    </span>
                    {imagePreview.sizeLabel ? (
                      <span className="rounded-full bg-slate-100 px-2 py-1">
                        {imagePreview.sizeLabel}
                      </span>
                    ) : null}
                    {imageInfo ? (
                      <span className="rounded-full bg-slate-100 px-2 py-1">
                        {imageInfo.width} × {imageInfo.height}
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-600">
                    {imagePreview.note}
                  </p>
                </div>
              </div>
            ) : (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
                No image preview yet. Choose a local image, drag one here, or enter an absolute path and click "Preview path".
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <button
                className="rounded-full bg-slate-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
                onClick={handleRun}
                disabled={running || !articleText.trim()}
                type="button"
              >
                {running ? "Running..." : "Run analysis"}
              </button>
              <button
                className="rounded-full border border-slate-300 px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-white"
                onClick={handleLoadSample}
                type="button"
              >
                Load sample
              </button>
              <button
                className="rounded-full border border-slate-300 px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-white"
                onClick={() => {
                  setArticleText("");
                  setImagePath("");
                  setImageName("");
                  setSelectedImageFile(null);
                  replacePreview(EMPTY_PREVIEW);
                  setImageInfo(null);
                  clearSampleBinding();
                  resetRunState();
                }}
                type="button"
              >
                Clear
              </button>
            </div>

            {error ? (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {error}
              </div>
            ) : null}
          </div>
        </section>

        <div className="min-w-0 space-y-6 xl:col-start-2">
          <div className="grid gap-6 xl:grid-cols-2">
            <div className="min-w-0">
              <div className="panel h-full p-5">
                <p className="panel-title mb-4">Run Summary</p>
                {stats ? (
                  <div className="space-y-4">
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                        Relationship
                      </div>
                      <div className="mt-2 text-lg font-semibold text-slate-900">
                        {stats.relationship || "Pending"}
                      </div>
                    </div>
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-2xl bg-slate-50 p-4">
                        <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                          Active Cues
                        </div>
                        <div className="mt-2 text-3xl font-semibold text-slate-900">
                          {stats.activeCueCount}
                        </div>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-4">
                        <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                          Mean Score
                        </div>
                        <div className="mt-2 text-3xl font-semibold text-slate-900">
                          {stats.meanScore.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm leading-6 text-slate-600">
                    After the run completes, this panel shows the relationship type, active cue count, and mean score.
                  </p>
                )}
              </div>
            </div>

            <div className="min-w-0">
              <StageProgress currentStage={currentStage} items={stageItems} />
            </div>
          </div>

          <section className="min-w-0">
            {result ? (
              <ResultOverview
                result={result}
                observationSourceLabel={resultSourceLabel || "Current input observations"}
              />
            ) : (
              <div className="panel flex min-h-[620px] items-center justify-center p-8 text-center">
                <div className="max-w-2xl">
                  <p className="panel-title mb-3">Results</p>
                  <h2 className="text-2xl font-semibold text-slate-900">
                    Waiting for a single-sample run
                  </h2>
                  <p className="mt-4 text-sm leading-6 text-slate-600">
                    After you provide one image and one passage, the page first shows Stage 1 observations and then fills in the 9 cues and the consistency result.
                    {localBridgeEnabled
                      ? localInferenceMode === "tauri"
                        ? " Desktop mode prioritizes the real local model bridge."
                        : " Browser dev mode uses the Vite local bridge to call real local inference."
                      : " In static mode, the app replays real evaluation samples and falls back to a demo-structured result for manual input."}
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <section className="panel p-5">
        <div className="mb-5 flex items-end justify-between gap-3">
          <div>
            <p className="panel-title">Recent Analyses</p>
            <h2 className="mt-2 text-xl font-semibold text-slate-900">
              Local cached history
            </h2>
          </div>
          <div className="text-sm text-slate-500">Last 6 runs</div>
        </div>
        {history.length ? (
          <div className="grid gap-4 xl:grid-cols-3">
            {history.map((item) => (
              <button
                key={item.id}
                className="rounded-3xl border border-slate-200 bg-slate-50 p-4 text-left transition hover:bg-white"
                onClick={() => {
                  setResultSourceLabel("Cached result");
                  setResult(item.result);
                }}
                type="button"
              >
                <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
                  {new Date(item.createdAt).toLocaleString()}
                </div>
                <div className="mt-2 text-base font-semibold text-slate-900">
                  {item.title}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                  <span className="rounded-full bg-white px-2 py-1">
                    {item.relationship || "Pending"}
                  </span>
                  <span className="rounded-full bg-white px-2 py-1">
                    {item.activeCueCount} active cues
                  </span>
                  {item.imageName ? (
                    <span className="rounded-full bg-white px-2 py-1">
                      {item.imageName}
                    </span>
                  ) : null}
                </div>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-sm leading-6 text-slate-600">
            No history yet. After you finish a run, the result is cached locally in the browser.
          </p>
        )}
      </section>
    </div>
  );
}
