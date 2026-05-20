import { loadCases } from "./cases";
import { canUseLocalInferenceBridge, streamLocalInference } from "./localBridge";
import { getActiveCueCount } from "../utils/analysis";
import type {
  AnalysisInput,
  AnalysisStreamEvent,
  CaseRecord,
  FullAnalysisResult,
  RecentAnalysisRecord,
  Stage1Result,
} from "../types/analysis";

const ANALYSIS_HISTORY_KEY = "war-news-bias-analysis-history";
const ANALYSIS_DRAFT_KEY = "war-news-bias-analysis-draft";

const delay = (ms: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });

const DEFAULT_OBSERVATION: Stage1Result = {
  visual_observation:
    "The current input is not yet connected to a real vision model, so the local demo first generates a readable observation summary from image availability and input source.",
  textual_observation:
    "The current input is not yet connected to real text inference, so the local demo extracts a summary from the article text and simulates a Stage 1 result.",
  joint_mechanism:
    "The current demo prioritizes the two-stage product structure. Once local Python inference is fully connected, this field will be replaced with the real joint_mechanism.",
};

function readJsonFromStorage<T>(key: string, fallback: T) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJsonToStorage(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage failures in demo mode.
  }
}

function trimText(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function buildStage1FromInput(input: AnalysisInput): Stage1Result {
  const preview = trimText(input.articleText).slice(0, 220);
  const sampleHint = input.sampleCase?.title
    ? `This input references the sample "${input.sampleCase.title}".`
    : "This input was entered manually.";

  return {
    visual_observation: input.imagePath
      ? `Image input received: ${input.imageName ?? input.imagePath}. ${sampleHint} The demo keeps the stage feedback and result layout, while the real visual observation will be produced once the Python inference bridge is used.`
      : "No resolvable image path was provided, so the visual observation currently falls back to an explanatory placeholder.",
    textual_observation: preview
      ? `Article summary: ${preview}${preview.length >= 220 ? "..." : ""}`
      : DEFAULT_OBSERVATION.textual_observation,
    joint_mechanism:
      "Image input and article text jointly affect cues and consistency in Stage 2. The demo prioritizes replaying existing sample outputs or using an approximate result template.",
  };
}

function getKeywordScore(text: string, keywords: string[]) {
  const normalized = text.toLowerCase();
  return keywords.reduce(
    (count, keyword) => count + Number(normalized.includes(keyword.toLowerCase())),
    0,
  );
}

function mergeStage1(result: FullAnalysisResult, stage1: Stage1Result): FullAnalysisResult {
  return {
    observations: stage1,
    final_result: result.final_result,
  };
}

function buildHeuristicResult(input: AnalysisInput, stage1: Stage1Result): FullAnalysisResult {
  const template: FullAnalysisResult = {
    observations: stage1,
    final_result: {
      cues: {
        V1_Salience: {
          present: Boolean(input.imagePath),
          score: input.imagePath ? 1 : 0,
          reason: input.imagePath
            ? "The current input includes an image, so the system keeps the visual subject analysis slot active, although real visual encoding is not yet connected."
            : "No image path was provided, so visual subject salience cannot be assessed.",
        },
        T1_Agent_Label: {
          present: getKeywordScore(input.articleText, [
            "refugee",
            "victim",
            "terror",
            "侵略",
            "难民",
            "无辜",
          ]) > 0,
          score: Math.min(
            2,
            getKeywordScore(input.articleText, [
              "refugee",
              "victim",
              "terror",
              "侵略",
              "难民",
              "无辜",
            ]),
          ),
          reason:
            "A heuristic judgment is made from label-like wording in the article text. This is only for demonstrating the product structure and does not represent real model output.",
        },
        T2_Causal_Attribution: {
          present: getKeywordScore(input.articleText, [
            "because",
            "caused by",
            "led to",
            "blamed",
            "导致",
            "归咎于",
            "造成",
          ]) > 0,
          score: Math.min(
            2,
            getKeywordScore(input.articleText, [
              "because",
              "caused by",
              "led to",
              "blamed",
              "导致",
              "归咎于",
              "造成",
            ]),
          ),
          reason:
            "Causal attribution wording appears in the text, so T2 is marked as potentially active. This will be replaced by the real Stage 2 model output later.",
        },
        M2_Binary_Roles: {
          present:
            getKeywordScore(input.articleText, [
              "hero",
              "enemy",
              "aggressor",
              "victim",
              "英雄",
              "敌人",
              "侵略者",
              "受害者",
            ]) > 1,
          score:
            getKeywordScore(input.articleText, [
              "hero",
              "enemy",
              "aggressor",
              "victim",
              "英雄",
              "敌人",
              "侵略者",
              "受害者",
            ]) > 1
              ? 1
              : 0,
          reason:
            "If the article clearly organizes actors into opposing roles, the demo highlights the binary-role cue.",
        },
      },
      consistency_dimension: {
        D1_Relationship_Type: input.imagePath ? "Supplementary" : "",
        reason: input.imagePath
          ? "The demo currently treats the image as supplementary evidence for the text by default, pending a more detailed relationship judgment from real inference."
          : "A complete image-text input has not been formed yet, so no relationship type is emitted.",
      },
    },
  };

  return template;
}

export function createRecentRecord(
  title: string,
  input: AnalysisInput,
  result: FullAnalysisResult,
): RecentAnalysisRecord {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    title,
    imageName: input.imageName,
    relationship: result.final_result.consistency_dimension.D1_Relationship_Type,
    activeCueCount: getActiveCueCount(result),
    result,
  };
}

export function loadAnalysisDraft() {
  return readJsonFromStorage(ANALYSIS_DRAFT_KEY, {
    articleText: "",
    imagePath: "",
    sampleId: "",
  });
}

export function saveAnalysisDraft(draft: {
  articleText: string;
  imagePath: string;
  sampleId: string;
}) {
  writeJsonToStorage(ANALYSIS_DRAFT_KEY, draft);
}

export function loadRecentAnalyses() {
  return readJsonFromStorage<RecentAnalysisRecord[]>(ANALYSIS_HISTORY_KEY, []);
}

export function saveRecentAnalysis(record: RecentAnalysisRecord) {
  const history = loadRecentAnalyses();
  const next = [record, ...history].slice(0, 6);
  writeJsonToStorage(ANALYSIS_HISTORY_KEY, next);
  return next;
}

async function resolveSampleCase(input: AnalysisInput): Promise<CaseRecord | null> {
  if (input.sampleCase) {
    return input.sampleCase;
  }

  const cases = await loadCases();
  const normalized = trimText(input.articleText);

  const exactTitle = cases.find((item) => trimText(item.title) === normalized);
  if (exactTitle) {
    return exactTitle;
  }

  return null;
}

export async function* runAnalysisStream(
  input: AnalysisInput,
): AsyncGenerator<AnalysisStreamEvent, void, void> {
  try {
    if (canUseLocalInferenceBridge()) {
      if (!input.imagePath && !input.imageDataBase64) {
        throw new Error("Real local inference requires an image file or an accessible image path.");
      }

      for await (const event of streamLocalInference(input)) {
        yield event;
      }
      return;
    }

    const sampleCase = await resolveSampleCase(input);
    const stage1 = sampleCase?.result.observations ?? buildStage1FromInput(input);
    const finalResult = sampleCase
      ? mergeStage1(sampleCase.result, stage1)
      : buildHeuristicResult(input, stage1);

    yield { type: "status", stage: "boot", message: "Validating input and preparing the local analysis task" };
    await delay(450);

    yield {
      type: "status",
      stage: "stage1",
      message: sampleCase
        ? "Replaying sample observations"
        : "Stage 1: generating observations",
    };
    await delay(700);

    yield {
      type: "partial",
      stage: "stage1",
      data: stage1,
      sourceLabel: sampleCase ? "Evaluation reference observations" : "Current input observations",
    };
    await delay(450);

    yield {
      type: "status",
      stage: "stage2",
      message: sampleCase
        ? "Replaying cues and consistency"
        : "Stage 2: assembling cues and consistency",
    };
    await delay(900);

    yield {
      type: "final",
      data: finalResult,
      sourceLabel: sampleCase ? "Evaluation reference observations" : "Current input observations",
    };
  } catch (error) {
    yield {
      type: "error",
      message: error instanceof Error ? error.message : "Analysis failed",
    };
  }
}

export function buildRecentRecordTitle(
  input: AnalysisInput,
  sampleCase: CaseRecord | null,
) {
  if (sampleCase?.title) {
    return sampleCase.title;
  }

  const preview = trimText(input.articleText);
  return preview ? preview.slice(0, 80) : "Untitled input";
}
