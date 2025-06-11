mod explorer;
mod tree;
mod ui;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use crate::explorer::Explorer;

#[derive(Parser)]
#[command(name = "safetensors-explorer")]
#[command(about = "Interactive explorer for SafeTensors files")]
struct Args {
    #[arg(help = "SafeTensors files to explore")]
    files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.files.is_empty() {
        eprintln!("Error: Please specify one or more SafeTensors files to explore.");
        eprintln!("Usage: safetensors-explorer <file1.safetensors> [file2.safetensors] ...");
        std::process::exit(1);
    }

    for file in &args.files {
        if !file.exists() {
            eprintln!("Error: File does not exist: {}", file.display());
            std::process::exit(1);
        }
    }

    let mut explorer = Explorer::new(args.files);
    explorer.run()
}