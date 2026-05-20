import { invoke, isTauri } from "@tauri-apps/api/core";
import type {
  AnalysisInput,
  AnalysisStreamEvent,
  LocalInferenceResponse,
} from "../types/analysis";

export type LocalInferenceMode = "tauri" | "vite-dev";

export function getLocalInferenceMode(): LocalInferenceMode | null {
  if (isTauri()) {
    return "tauri";
  }

  if (import.meta.env.DEV) {
    return "vite-dev";
  }

  return null;
}

export function canUseLocalInferenceBridge() {
  return getLocalInferenceMode() !== null;
}

async function* streamTauriLocalInference(
  input: AnalysisInput,
): AsyncGenerator<AnalysisStreamEvent, void, void> {
  yield {
    type: "status",
    stage: "boot",
    message: "Connecting to the Tauri local inference bridge and preparing the Python model environment",
  };

  const inferencePromise = invoke<LocalInferenceResponse>("run_local_analysis", {
    request: {
      articleText: input.articleText,
      imagePath: input.imagePath,
      imageName: input.imageName,
      imageDataBase64: input.imageDataBase64,
    },
  });

  yield {
    type: "status",
    stage: "stage1",
    message: "Running the real local model: generating observations",
  };
  yield {
    type: "status",
    stage: "stage2",
    message: "Running the real local model: generating cues and consistency",
  };

  const response = await inferencePromise;

  yield {
    type: "partial",
    stage: "stage1",
    data: response.result.observations,
    sourceLabel: "Real local model inference",
  };
  yield {
    type: "final",
    data: response.result,
    sourceLabel: "Real local model inference",
  };
}

async function readErrorResponse(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    const payload = (await response.json()) as { error?: string };
    return payload.error || "Local inference request failed.";
  }

  const text = await response.text();
  return text || "Local inference request failed.";
}

async function* streamViteDevInference(
  input: AnalysisInput,
): AsyncGenerator<AnalysisStreamEvent, void, void> {
  const response = await fetch("/api/local-analysis/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      articleText: input.articleText,
      imagePath: input.imagePath,
      imageName: input.imageName,
      imageDataBase64: input.imageDataBase64,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }

  if (!response.body) {
    throw new Error("The local inference stream is unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let lineBreakIndex = buffer.indexOf("\n");
    while (lineBreakIndex >= 0) {
      const rawLine = buffer.slice(0, lineBreakIndex).trim();
      buffer = buffer.slice(lineBreakIndex + 1);

      if (rawLine) {
        yield JSON.parse(rawLine) as AnalysisStreamEvent;
      }

      lineBreakIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    yield JSON.parse(trailing) as AnalysisStreamEvent;
  }
}

export async function* streamLocalInference(
  input: AnalysisInput,
): AsyncGenerator<AnalysisStreamEvent, void, void> {
  const mode = getLocalInferenceMode();

  if (!mode) {
    throw new Error("Local inference bridge is unavailable in the current runtime.");
  }

  if (mode === "tauri") {
    yield* streamTauriLocalInference(input);
    return;
  }

  yield* streamViteDevInference(input);
}
