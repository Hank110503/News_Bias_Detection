import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(frontendRoot, "..");

const finalLabelPath = path.join(
  repoRoot,
  "data",
  "processed",
  "labels",
  "final_label.json",
);
const articlesRoot = path.join(
  repoRoot,
  "data",
  "processed",
  "texts",
  "articles_texts",
);
const outputDir = path.join(frontendRoot, "public", "demo");

const CUE_CODES = [
  "V1_Salience",
  "V2_Color_Polarity",
  "V3_Power_Angle",
  "V4_Visual_Selectivity",
  "T1_Agent_Label",
  "T2_Causal_Attribution",
  "M1_Affect_Mismatch",
  "M2_Binary_Roles",
  "M3_Symbol_Decontex",
];

function resolveRepoRelativeImagePath(sourceImagePath) {
  if (typeof sourceImagePath !== "string") {
    return "";
  }

  const imageName = path.basename(sourceImagePath);
  return imageName
    ? path.posix.join("data", "processed", "imgs", "war_imgs_total", imageName)
    : "";
}

function normalizeCueItem(item) {
  if (!item || typeof item !== "object") {
    return undefined;
  }

  return {
    present: Boolean(item.present),
    score: typeof item.score === "number" ? item.score : 0,
    reason: typeof item.reason === "string" ? item.reason : "",
  };
}

function flattenCues(groups) {
  const flattened = {};

  for (const groupName of ["visual_only", "text_only", "cross_modal"]) {
    const group = groups?.[groupName];
    if (!group || typeof group !== "object") {
      continue;
    }

    for (const cueCode of Object.keys(group)) {
      if (!CUE_CODES.includes(cueCode)) {
        continue;
      }

      const normalized = normalizeCueItem(group[cueCode]);
      if (normalized) {
        flattened[cueCode] = normalized;
      }
    }
  }

  return flattened;
}

async function readArticlesById() {
  const articleFiles = await readFileIndex();
  const articleMap = new Map();

  for (const articleFile of articleFiles) {
    const articlePath = path.join(articlesRoot, articleFile);
    const raw = await readFile(articlePath, "utf8");
    const parsed = JSON.parse(raw);
    const articleId =
      typeof parsed.article_id === "string"
        ? parsed.article_id
        : path.basename(articleFile, ".json");

    articleMap.set(articleId, {
      articleText: typeof parsed.text === "string" ? parsed.text.trim() : "",
      sourceUrl: typeof parsed.url === "string" ? parsed.url : "",
    });
  }

  return articleMap;
}

async function readFileIndex() {
  const { readdir } = await import("node:fs/promises");
  const files = await readdir(articlesRoot);
  return files.filter((file) => file.endsWith(".json")).sort();
}

function buildCaseRecord(item, articleMap) {
  const newsId = String(item.news_id ?? "");
  const articleId = newsId.padStart(6, "0");
  const article = articleMap.get(articleId) ?? {
    articleText: "",
    sourceUrl: "",
  };
  const imageHint =
    typeof item.image_path === "string" ? path.basename(item.image_path) : "";

  return {
    newsId,
    title: typeof item.title === "string" ? item.title : "",
    imageHint,
    imagePath: resolveRepoRelativeImagePath(item.image_path),
    articleText: article.articleText,
    sourceUrl: article.sourceUrl,
    sourceIndex: null,
    observationSource: "final_label",
    observationDisplaySource: "final_label",
    goldObservations: null,
    result: {
      observations: {
        visual_observation:
          typeof item.analysis_chain?.visual_observation === "string"
            ? item.analysis_chain.visual_observation
            : "",
        textual_observation:
          typeof item.analysis_chain?.textual_observation === "string"
            ? item.analysis_chain.textual_observation
            : "",
        joint_mechanism:
          typeof item.analysis_chain?.joint_mechanism === "string"
            ? item.analysis_chain.joint_mechanism
            : "",
      },
      final_result: {
        cues: flattenCues(item.cues),
        consistency_dimension: {
          reason:
            typeof item.consistency_dimension?.D1_reason === "string"
              ? item.consistency_dimension.D1_reason
              : "",
          D1_Relationship_Type:
            typeof item.consistency_dimension?.D1_Relationship_Type === "string"
              ? item.consistency_dimension.D1_Relationship_Type
              : "",
        },
      },
    },
  };
}

async function main() {
  const [finalLabelRaw, articleMap] = await Promise.all([
    readFile(finalLabelPath, "utf8"),
    readArticlesById(),
  ]);

  const finalLabels = JSON.parse(finalLabelRaw);
  const cases = finalLabels.map((item) => buildCaseRecord(item, articleMap));

  await mkdir(outputDir, { recursive: true });
  await writeFile(
    path.join(outputDir, "eval_cases.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        totalCases: cases.length,
        cases,
      },
      null,
      2,
    ),
  );

  console.log(`Wrote ${cases.length} demo cases to ${outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
