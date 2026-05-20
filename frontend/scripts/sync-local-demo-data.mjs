import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(frontendRoot, "..");

const reportPath = path.join(
  repoRoot,
  "outputs",
  "stage2_v3",
  "eval_predicted_obs_report.json",
);
const predictionsPath = path.join(
  repoRoot,
  "outputs",
  "stage2_v3",
  "eval_predicted_obs_predictions.jsonl",
);
const articlesRoot = path.join(
  repoRoot,
  "data",
  "processed",
  "texts",
  "articles_texts",
);
const outputDir = path.join(frontendRoot, "public", "demo");

function resolveRepoRelativeImagePath(sourceImagePath) {
  if (typeof sourceImagePath !== "string") {
    return "";
  }

  const imageName = path.basename(sourceImagePath);
  return imageName
    ? path.posix.join("data", "processed", "imgs", "war_imgs_total", imageName)
    : "";
}

function normalizeMetrics(report) {
  return {
    jsonParseRate: report.json_parse_rate,
    matchedRows: report.matched_rows,
    cuePresentMacroF1: report.cue_present_macro_f1,
    cueScoreAccuracy: report.cue_score_accuracy,
    cueScoreMae: report.cue_score_mae,
    relationMacroF1: report.relation_macro_f1,
    cueReasonNonEmptyRate: report.cue_reason_non_empty_rate,
    consistencyReasonNonEmptyRate: report.consistency_reason_non_empty_rate,
  };
}

async function readArticleText(newsId) {
  const articleId = String(newsId).padStart(6, "0");
  const articlePath = path.join(articlesRoot, `${articleId}.json`);

  try {
    const raw = await readFile(articlePath, "utf8");
    const parsed = JSON.parse(raw);
    return {
      articleText: typeof parsed.text === "string" ? parsed.text.trim() : "",
      sourceUrl: typeof parsed.url === "string" ? parsed.url : "",
    };
  } catch {
    return {
      articleText: "",
      sourceUrl: "",
    };
  }
}

async function buildCaseRecord(line) {
  const parsed = JSON.parse(line);
  const { meta = {}, cues = {}, consistency_dimension } = parsed;
  const { articleText, sourceUrl } = await readArticleText(meta.news_id);
  const imageHint =
    typeof meta.source_image_path === "string"
      ? path.basename(meta.source_image_path)
      : "";
  const imagePath = resolveRepoRelativeImagePath(meta.source_image_path);

  return {
    newsId: String(meta.news_id ?? ""),
    title: String(meta.title ?? ""),
    imageHint,
    imagePath,
    articleText,
    sourceUrl,
    sourceIndex: typeof meta.source_index === "number" ? meta.source_index : null,
    observationSource:
      typeof meta.observation_source === "string" ? meta.observation_source : "",
    observationDisplaySource: meta.gold_observations
      ? "gold_reference"
      : meta.observation_source ?? "",
    goldObservations: meta.gold_observations ?? null,
    result: {
      observations: meta.gold_observations ?? {
        visual_observation: "当前样例未提供 observations。",
        textual_observation: "当前样例未提供 observations。",
        joint_mechanism: "当前样例未提供 observations。",
      },
      final_result: {
        cues,
        consistency_dimension,
      },
    },
  };
}

async function main() {
  const reportRaw = await readFile(reportPath, "utf8");
  const report = JSON.parse(reportRaw);
  const predictionsRaw = await readFile(predictionsPath, "utf8");
  const predictionLines = predictionsRaw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const cases = [];
  for (const line of predictionLines) {
    cases.push(await buildCaseRecord(line));
  }

  await mkdir(outputDir, { recursive: true });
  await writeFile(
    path.join(outputDir, "eval_metrics.json"),
    JSON.stringify(normalizeMetrics(report)),
  );
  await writeFile(
    path.join(outputDir, "eval_cases.json"),
    JSON.stringify({
      generatedAt: new Date().toISOString(),
      totalCases: cases.length,
      cases,
    }),
  );

  console.log(`Wrote ${cases.length} demo cases to ${outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
