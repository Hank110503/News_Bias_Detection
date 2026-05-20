#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

const DEFAULT_ADAPTER_PATH: &str = "outputs/stage2_v3/final_adapter";
const DEFAULT_MODEL_NAME: &str = "Qwen/Qwen2.5-VL-7B-Instruct";
const DEFAULT_PROJECT_PYTHON: &str = "/root/miniconda3/envs/news_bias_312/bin/python";

#[derive(Serialize)]
struct BridgeStatus {
    status: String,
    message: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalAnalysisRequest {
    article_text: String,
    image_path: Option<String>,
    image_name: Option<String>,
    image_data_base64: Option<String>,
    python_path: Option<String>,
    adapter_path: Option<String>,
    model_name: Option<String>,
    max_new_tokens: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BridgeRuntime {
    mode: String,
    repo_root: String,
    python_path: String,
    adapter_path: String,
    model_name: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalAnalysisResponse {
    result: serde_json::Value,
    runtime: BridgeRuntime,
}

fn repo_root() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "Failed to resolve repository root from Tauri manifest dir.".to_string())
}

fn current_epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn build_temp_path(prefix: &str, extension: &str) -> PathBuf {
    let safe_extension = extension
        .trim()
        .trim_start_matches('.')
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    let extension = if safe_extension.is_empty() {
        "tmp"
    } else {
        safe_extension.as_str()
    };

    env::temp_dir().join(format!(
        "{prefix}-{}-{}.{}",
        std::process::id(),
        current_epoch_millis(),
        extension
    ))
}

fn guess_image_extension(image_name: Option<&str>, image_path: Option<&str>) -> String {
    image_name
        .and_then(|value| Path::new(value).extension())
        .or_else(|| image_path.and_then(|value| Path::new(value).extension()))
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "png".to_string())
}

fn decode_base64_payload(raw_value: &str) -> Result<Vec<u8>, String> {
    let encoded = raw_value
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(raw_value)
        .trim();
    STANDARD
        .decode(encoded)
        .map_err(|error| format!("Failed to decode uploaded image payload: {error}"))
}

fn resolve_python_path(requested: Option<&str>) -> String {
    if let Some(value) = requested.filter(|value| !value.trim().is_empty()) {
        return value.trim().to_string();
    }

    if let Ok(value) = env::var("WAR_NEWS_BIAS_PYTHON") {
        if !value.trim().is_empty() {
            return value.trim().to_string();
        }
    }

    if Path::new(DEFAULT_PROJECT_PYTHON).exists() {
        return DEFAULT_PROJECT_PYTHON.to_string();
    }

    "python".to_string()
}

fn cleanup_temp_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

#[tauri::command]
fn ping_bridge() -> BridgeStatus {
    BridgeStatus {
        status: "ok".to_string(),
        message: "Tauri bridge placeholder is ready".to_string(),
    }
}

#[tauri::command]
fn run_local_analysis(request: LocalAnalysisRequest) -> Result<LocalAnalysisResponse, String> {
    if request.article_text.trim().is_empty() {
        return Err("Article text is required for local inference.".to_string());
    }

    let repo_root = repo_root()?;
    let python_path = resolve_python_path(request.python_path.as_deref());
    let adapter_path = request
        .adapter_path
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_ADAPTER_PATH.to_string());
    let model_name = request
        .model_name
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());

    let text_path = build_temp_path("war-news-bias-input", "txt");
    fs::write(&text_path, &request.article_text)
        .map_err(|error| format!("Failed to write temporary article text file: {error}"))?;

    let mut temp_paths = vec![text_path.clone()];

    let image_argument = if let Some(encoded_image) = request.image_data_base64.as_deref() {
        let image_bytes = decode_base64_payload(encoded_image)?;
        let extension =
            guess_image_extension(request.image_name.as_deref(), request.image_path.as_deref());
        let temp_image_path = build_temp_path("war-news-bias-image", &extension);
        fs::write(&temp_image_path, image_bytes)
            .map_err(|error| format!("Failed to write temporary image file: {error}"))?;
        temp_paths.push(temp_image_path.clone());
        temp_image_path.to_string_lossy().to_string()
    } else if let Some(image_path) = request.image_path.clone().filter(|value| !value.trim().is_empty()) {
        image_path
    } else {
        cleanup_temp_paths(&temp_paths);
        return Err("An image file or image path is required for full local inference.".to_string());
    };

    let mut command = Command::new(&python_path);
    command
        .current_dir(&repo_root)
        .env("PYTHONIOENCODING", "utf-8")
        .arg("infer.py")
        .arg("--stage")
        .arg("full")
        .arg("--adapter_path")
        .arg(&adapter_path)
        .arg("--model_name")
        .arg(&model_name)
        .arg("--text-file")
        .arg(&text_path)
        .arg("--image")
        .arg(&image_argument);

    if let Some(max_new_tokens) = request.max_new_tokens {
        command.arg("--max_new_tokens").arg(max_new_tokens.to_string());
    }

    let output = command.output().map_err(|error| {
        cleanup_temp_paths(&temp_paths);
        format!("Failed to launch local inference command with {python_path}: {error}")
    })?;

    cleanup_temp_paths(&temp_paths);

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            stdout
        } else if stdout.is_empty() {
            stderr
        } else {
            format!("{stderr}\n{stdout}")
        };

        return Err(format!(
            "Local inference failed with status {:?}. {detail}",
            output.status.code()
        ));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Local inference returned non-UTF8 output: {error}"))?;
    let result = serde_json::from_str::<serde_json::Value>(stdout.trim()).map_err(|error| {
        format!(
            "Failed to parse local inference JSON output: {error}. Output preview: {}",
            stdout.chars().take(400).collect::<String>()
        )
    })?;

    Ok(LocalAnalysisResponse {
        result,
        runtime: BridgeRuntime {
            mode: "tauri_local".to_string(),
            repo_root: repo_root.to_string_lossy().to_string(),
            python_path,
            adapter_path,
            model_name,
        },
    })
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![ping_bridge, run_local_analysis])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
