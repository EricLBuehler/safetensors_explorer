mod explorer;
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
#[command(about = "Interactive explorer for SafeTensors files")]
struct Args {
    #[arg(help = "SafeTensors files or directories to explore")]
    paths: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "Recursively search directories for SafeTensors files"
    )]
    recursive: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.paths.is_empty() {
        eprintln!("Error: Please specify one or more SafeTensors files or directories to explore.");
        eprintln!("Usage: safetensors-explorer <file1.safetensors> [directory] ...");
        std::process::exit(1);
    }

    let files = collect_safetensors_files(&args.paths, args.recursive)?;

    if files.is_empty() {
        eprintln!("Error: No SafeTensors files found in the specified paths.");
        std::process::exit(1);
    }

    let mut explorer = Explorer::new(files);
    explorer.run()
}

fn collect_safetensors_files(paths: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for path in paths {
        if !path.exists() {
            eprintln!("Warning: Path does not exist: {}", path.display());
            continue;
        }

        if path.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                files.push(path.clone());
            } else {
                eprintln!("Warning: Skipping non-SafeTensors file: {}", path.display());
            }
        } else if path.is_dir() {
            // Check for SafeTensors index file first
            let index_path = path.join("model.safetensors.index.json");
            if index_path.exists() {
                let index_files = parse_safetensors_index(&index_path)?;
                for file in index_files {
                    let full_path = path.join(file);
                    if full_path.exists() {
                        files.push(full_path);
                    }
                }
            } else {
                // Fallback to directory scanning
                let pattern = if recursive {
                    format!("{}/**/*.safetensors", path.display())
                } else {
                    format!("{}/*.safetensors", path.display())
                };

                for entry in glob::glob(&pattern).context("Failed to read glob pattern")? {
                    match entry {
                        Ok(file_path) => files.push(file_path),
                        Err(e) => eprintln!("Warning: Error reading file: {}", e),
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
