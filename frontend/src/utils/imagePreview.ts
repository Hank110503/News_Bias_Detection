import { convertFileSrc, isTauri } from "@tauri-apps/api/core";

type CaseImageSource = {
  imagePath?: string;
};

export function isImagePath(value: string) {
  return /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(value);
}

function isAbsoluteFilePath(value: string) {
  return /^(?:[A-Za-z]:[\\/]|\/)/.test(value);
}

function normalizeFilePath(value: string) {
  return value.replace(/\\/g, "/");
}

export function toDevFsUrl(absolutePath?: string) {
  if (!absolutePath || !import.meta.env.DEV) {
    return "";
  }

  const normalizedPath = normalizeFilePath(absolutePath);
  return /^[A-Za-z]:\//.test(normalizedPath)
    ? `/@fs/${encodeURI(normalizedPath)}`
    : `/@fs${encodeURI(normalizedPath)}`;
}

function toRepoAbsolutePath(pathValue: string) {
  return `${__REPO_ROOT__}/${pathValue.replace(/^\/+/, "")}`;
}

export function resolveImageUrl(pathValue?: string) {
  if (!pathValue) {
    return "";
  }

  if (/^(?:https?:|data:|blob:)/.test(pathValue)) {
    return pathValue;
  }

  if (pathValue.startsWith("/demo/")) {
    return pathValue;
  }

  const absolutePath = isAbsoluteFilePath(pathValue)
    ? normalizeFilePath(pathValue)
    : toRepoAbsolutePath(pathValue);

  if (!isImagePath(absolutePath)) {
    return "";
  }

  return isTauri() ? convertFileSrc(absolutePath) : toDevFsUrl(absolutePath);
}

export function resolveCaseImageUrl(caseItem?: CaseImageSource | null) {
  return resolveImageUrl(caseItem?.imagePath);
}
