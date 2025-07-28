mod explorer;
mod gguf;
mod tree;
mod ui;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;

use crate::explorer::Explorer;

#[derive(Parser)]
#[command(name = "safetensors-explorer")]
#[command(about = "Interactive explorer for SafeTensors and GGUF files")]
struct Args {
    #[arg(
        help = "SafeTensors and GGUF files, directories, or glob patterns to explore (e.g., *.safetensors, model-*.gguf)"
    )]
    paths: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "Recursively search directories for SafeTensors and GGUF files"
    )]
    recursive: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.paths.is_empty() {
        eprintln!(
            "Error: Please specify one or more SafeTensors or GGUF files or directories to explore."
        );
        eprintln!(
            "Usage: safetensors-explorer <file1.safetensors> [file2.gguf] [directory] [*.safetensors] ..."
        );
        std::process::exit(1);
    }

    let files = collect_safetensors_files(&args.paths, args.recursive)?;

    if files.is_empty() {
        eprintln!("Error: No SafeTensors or GGUF files found in the specified paths.");
        std::process::exit(1);
    }

    let mut explorer = Explorer::new(files);
    explorer.run()
}

fn collect_safetensors_files(paths: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for path in paths {
        // Try to expand as glob pattern
        let expanded_paths: Vec<PathBuf> = match glob::glob(&path.to_string_lossy()) {
            Ok(paths) => paths.filter_map(Result::ok).collect(),
            Err(_) => vec![path.clone()], // Not a valid glob, treat as literal path
        };

        // Process each expanded path
        for expanded_path in expanded_paths {
            if !expanded_path.exists() {
                eprintln!("Warning: Path does not exist: {}", expanded_path.display());
                continue;
            }

            if expanded_path.is_file() {
                let ext = expanded_path.extension().and_then(|s| s.to_str());
                if ext == Some("safetensors") || ext == Some("gguf") {
                    files.push(expanded_path.clone());
                } else {
                    eprintln!(
                        "Warning: Skipping unsupported file: {}",
                        expanded_path.display()
                    );
                }
            } else if expanded_path.is_dir() {
                // Check for SafeTensors index file first
                let index_path = expanded_path.join("model.safetensors.index.json");
                if index_path.exists() {
                    let index_files = parse_safetensors_index(&index_path)?;
                    for file in index_files {
                        let full_path = expanded_path.join(file);
                        if full_path.exists() {
                            files.push(full_path);
                        }
                    }
                } else {
                    // Fallback to directory scanning
                    let patterns = if recursive {
                        vec![
                            format!("{}/**/*.safetensors", expanded_path.display()),
                            format!("{}/**/*.gguf", expanded_path.display()),
                        ]
                    } else {
                        vec![
                            format!("{}/*.safetensors", expanded_path.display()),
                            format!("{}/*.gguf", expanded_path.display()),
                        ]
                    };

                    for pattern in patterns {
                        for entry in glob::glob(&pattern).context("Failed to read glob pattern")? {
                            match entry {
                                Ok(file_path) => files.push(file_path),
                                Err(e) => eprintln!("Warning: Error reading file: {e}"),
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort files for consistent ordering
    files.sort();
    Ok(files)
}

fn parse_safetensors_index(index_path: &PathBuf) -> Result<Vec<String>> {
    let content = fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;

    let index: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse index file: {}", index_path.display()))?;

    let mut files = Vec::new();

    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for file_name in weight_map.values() {
            if let Some(file_str) = file_name.as_str() {
                if !files.contains(&file_str.to_string()) {
                    files.push(file_str.to_string());
                }
            }
        }
    }

    files.sort();
    Ok(files)
}
