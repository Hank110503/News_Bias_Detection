import { spawn } from "node:child_process";
import { webcrypto } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

if (!globalThis.crypto) {
  globalThis.crypto = webcrypto as typeof globalThis.crypto;
}

const DEFAULT_ADAPTER_PATH = "outputs/stage2_v3/final_adapter";
const DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct";
const DEFAULT_PROJECT_PYTHON = "/root/miniconda3/envs/news_bias_312/bin/python";

type LocalAnalysisRequest = {
  articleText?: string;
  imagePath?: string;
  imageName?: string;
  imageDataBase64?: string;
  pythonPath?: string;
  adapterPath?: string;
  modelName?: string;
  maxNewTokens?: number;
};

function resolvePythonPath() {
  const fromEnv = process.env.WAR_NEWS_BIAS_PYTHON?.trim();
  if (fromEnv) {
    return fromEnv;
  }

  return DEFAULT_PROJECT_PYTHON;
}

function guessImageExtension(request: LocalAnalysisRequest) {
  const dataUrlMime = request.imageDataBase64?.match(/^data:image\/([a-zA-Z0-9.+-]+);base64,/i)?.[1];
  if (dataUrlMime) {
    return dataUrlMime.toLowerCase() === "jpeg" ? "jpg" : dataUrlMime.toLowerCase();
  }

  const imageName = request.imageName ?? request.imagePath ?? "";
  const ext = path.extname(imageName).replace(/^\./, "").trim();
  return ext || "png";
}

function decodeBase64Payload(rawValue: string) {
  const encoded = rawValue.includes(",") ? rawValue.split(",").at(-1) ?? "" : rawValue;
  return Buffer.from(encoded.trim(), "base64");
}

async function readJsonBody(req: NodeJS.ReadableStream) {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as LocalAnalysisRequest;
}

function jsonErrorResponse(message: string) {
  return JSON.stringify({ error: message });
}

function eventLine(event: unknown) {
  return `${JSON.stringify(event)}\n`;
}

function localInferenceDevPlugin(repoRoot: string): Plugin {
  return {
    name: "war-news-bias-local-inference-dev",
    configureServer(server) {
      server.middlewares.use("/api/local-analysis/stream", async (req, res, next) => {
        if (req.method !== "POST") {
          next();
          return;
        }

        let tempDir = "";
        let child: ReturnType<typeof spawn> | null = null;
        let stderr = "";

        try {
          const request = await readJsonBody(req);
          const articleText = request.articleText?.trim() ?? "";
          if (!articleText) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(jsonErrorResponse("Article text is required for local inference."));
            return;
          }

          if (!request.imagePath?.trim() && !request.imageDataBase64?.trim()) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(jsonErrorResponse("An image file or image path is required for full local inference."));
            return;
          }

          tempDir = await mkdtemp(path.join(os.tmpdir(), "war-news-bias-"));
          const textPath = path.join(tempDir, "article.txt");
          await writeFile(textPath, articleText, "utf8");

          let imageArgument = "";
          if (request.imageDataBase64?.trim()) {
            const imagePath = path.join(tempDir, `image.${guessImageExtension(request)}`);
            await writeFile(imagePath, decodeBase64Payload(request.imageDataBase64));
            imageArgument = imagePath;
          } else {
            imageArgument = request.imagePath?.trim() ?? "";
          }

          const pythonPath = request.pythonPath?.trim() || resolvePythonPath();
          const adapterPath = request.adapterPath?.trim() || DEFAULT_ADAPTER_PATH;
          const modelName = request.modelName?.trim() || DEFAULT_MODEL_NAME;
          const args = [
            "infer_stream.py",
            "--stage",
            "full",
            "--adapter_path",
            adapterPath,
            "--model_name",
            modelName,
            "--text-file",
            textPath,
            "--image",
            imageArgument,
          ];

          if (request.maxNewTokens) {
            args.push("--max_new_tokens", String(request.maxNewTokens));
          }

          res.statusCode = 200;
          res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("X-Accel-Buffering", "no");

          child = spawn(pythonPath, args, {
            cwd: repoRoot,
            env: {
              ...process.env,
              PYTHONIOENCODING: "utf-8",
            },
          });

          child.stdout.on("data", (chunk) => {
            res.write(chunk);
          });

          child.stderr.on("data", (chunk) => {
            stderr += chunk.toString("utf8");
          });

          child.on("error", async (error) => {
            if (!res.writableEnded) {
              res.write(
                eventLine({
                  type: "error",
                  message: `Failed to launch local inference with ${pythonPath}: ${error.message}`,
                }),
              );
              res.end();
            }
            if (tempDir) {
              await rm(tempDir, { recursive: true, force: true });
            }
          });

          child.on("close", async (code) => {
            if (!res.writableEnded) {
              if (code !== 0) {
                const detail = stderr.trim();
                res.write(
                  eventLine({
                    type: "error",
                    message: detail
                      ? `Local inference failed with status ${code}. ${detail}`
                      : `Local inference failed with status ${code}.`,
                  }),
                );
              }
              res.end();
            }
            if (tempDir) {
              await rm(tempDir, { recursive: true, force: true });
            }
          });

          req.on("close", async () => {
            if (child && !child.killed) {
              child.kill("SIGTERM");
            }
            if (tempDir) {
              await rm(tempDir, { recursive: true, force: true });
            }
          });
        } catch (error) {
          if (tempDir) {
            await rm(tempDir, { recursive: true, force: true });
          }
          if (!res.writableEnded) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(
              jsonErrorResponse(
                error instanceof Error ? error.message : "Local inference bridge failed.",
              ),
            );
          }
        }
      });
    },
  };
}

export default defineConfig({
  define: {
    __REPO_ROOT__: JSON.stringify(path.resolve(__dirname, "..").replace(/\\/g, "/")),
  },
  plugins: [react(), localInferenceDevPlugin(path.resolve(__dirname, ".."))],
  server: {
    fs: {
      allow: [path.resolve(__dirname, "..")],
    },
    port: 1420,
    strictPort: true,
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
  preview: {
    port: 1421,
    strictPort: true,
  },
});
