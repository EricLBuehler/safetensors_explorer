use anyhow::{Context, Result};
use clap::Parser;
use safetensors::SafeTensors;
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::PathBuf,
};

#[derive(Parser)]
#[command(name = "safetensors-explorer")]
#[command(about = "Interactive explorer for SafeTensors files")]
struct Args {
    #[arg(help = "SafeTensors files to explore")]
    files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    size_bytes: usize,
}

#[derive(Debug, Clone)]
enum TreeNode {
    Group {
        name: String,
        children: Vec<TreeNode>,
        expanded: bool,
        tensor_count: usize,
        total_size: usize,
    },
    Tensor {
        info: TensorInfo,
    },
}

impl TreeNode {
    fn name(&self) -> &str {
        match self {
            TreeNode::Group { name, .. } => name,
            TreeNode::Tensor { info } => &info.name,
        }
    }

    fn is_expanded(&self) -> bool {
        match self {
            TreeNode::Group { expanded, .. } => *expanded,
            TreeNode::Tensor { .. } => false,
        }
    }

    fn toggle_expand(&mut self) {
        if let TreeNode::Group { expanded, .. } = self {
            *expanded = !*expanded;
        }
    }

    fn tensor_count(&self) -> usize {
        match self {
            TreeNode::Group { tensor_count, .. } => *tensor_count,
            TreeNode::Tensor { .. } => 1,
        }
    }

    fn total_size(&self) -> usize {
        match self {
            TreeNode::Group { total_size, .. } => *total_size,
            TreeNode::Tensor { info } => info.size_bytes,
        }
    }
}

struct Explorer {
    files: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    tree: Vec<TreeNode>,
}

impl Explorer {
    fn new(files: Vec<PathBuf>) -> Self {
        Self {
            files,
            tensors: Vec::new(),
            tree: Vec::new(),
        }
    }

    fn load_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let tensors = SafeTensors::deserialize(&buffer)
            .with_context(|| format!("Failed to parse SafeTensors file: {}", file_path.display()))?;

        self.tensors.clear();
        for name in tensors.names() {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape().to_vec();
            let dtype = format!("{:?}", tensor.dtype());
            let size_bytes = tensor.data().len();

            self.tensors.push(TensorInfo {
                name: name.to_string(),
                dtype,
                shape,
                size_bytes,
            });
        }

        self.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        self.build_tree();
        Ok(())
    }

    fn build_tree(&mut self) {
        let mut root_map: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        
        for tensor in &self.tensors {
            let parts: Vec<&str> = tensor.name.split('.').collect();
            if parts.len() > 1 {
                let prefix = parts[0].to_string();
                root_map.entry(prefix).or_insert_with(Vec::new).push(tensor.clone());
            } else {
                root_map.entry("_root".to_string()).or_insert_with(Vec::new).push(tensor.clone());
            }
        }

        self.tree.clear();
        for (prefix, mut tensors) in root_map {
            if prefix == "_root" {
                for tensor in tensors {
                    self.tree.push(TreeNode::Tensor { info: tensor });
                }
            } else {
                tensors.sort_by(|a, b| a.name.cmp(&b.name));
                let tensor_count = tensors.len();
                let total_size = tensors.iter().map(|t| t.size_bytes).sum();
                
                let children = self.build_subtree(&tensors, &prefix);
                
                self.tree.push(TreeNode::Group {
                    name: prefix,
                    children,
                    expanded: true,
                    tensor_count,
                    total_size,
                });
            }
        }
        
        self.tree.sort_by(|a, b| a.name().cmp(b.name()));
    }

    fn build_subtree(&self, tensors: &[TensorInfo], prefix: &str) -> Vec<TreeNode> {
        let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        let mut direct_tensors = Vec::new();

        for tensor in tensors {
            let remaining = tensor.name.strip_prefix(&format!("{}.", prefix)).unwrap_or(&tensor.name);
            let parts: Vec<&str> = remaining.split('.').collect();
            
            if parts.len() == 1 {
                direct_tensors.push(tensor.clone());
            } else {
                let next_prefix = parts[0].to_string();
                groups.entry(next_prefix).or_insert_with(Vec::new).push(tensor.clone());
            }
        }

        let mut result = Vec::new();
        
        for tensor in direct_tensors {
            result.push(TreeNode::Tensor { info: tensor });
        }

        for (group_name, group_tensors) in groups {
            let tensor_count = group_tensors.len();
            let total_size = group_tensors.iter().map(|t| t.size_bytes).sum();
            let full_prefix = format!("{}.{}", prefix, group_name);
            let children = self.build_subtree(&group_tensors, &full_prefix);
            
            result.push(TreeNode::Group {
                name: group_name,
                children,
                expanded: false,
                tensor_count,
                total_size,
            });
        }

        result.sort_by(|a, b| a.name().cmp(b.name()));
        result
    }

    fn print_tree(&self) {
        println!("SafeTensors File Structure");
        println!("=========================");
        println!();
        
        for node in &self.tree {
            self.print_node(node, 0);
        }
    }

    fn print_node(&self, node: &TreeNode, depth: usize) {
        let indent = "  ".repeat(depth);
        
        match node {
            TreeNode::Group { name, children, tensor_count, total_size, .. } => {
                println!("{}📁 {} ({} tensors, {})", 
                    indent, 
                    name, 
                    tensor_count,
                    Self::format_size(*total_size)
                );
                
                for child in children {
                    self.print_node(child, depth + 1);
                }
            }
            TreeNode::Tensor { info } => {
                let short_name = info.name.split('.').last().unwrap_or(&info.name);
                println!("{}  {} [{}, {}, {}]", 
                    indent,
                    short_name,
                    info.dtype,
                    Self::format_shape(&info.shape),
                    Self::format_size(info.size_bytes)
                );
            }
        }
    }

    fn format_shape(shape: &[usize]) -> String {
        format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
    }

    fn format_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        if unit_idx == 0 {
            format!("{} {}", bytes, UNITS[unit_idx])
        } else {
            format!("{:.1} {}", size, UNITS[unit_idx])
        }
    }

    fn run(&mut self) -> Result<()> {
        for file_path in &self.files.clone() {
            println!("Loading file: {}", file_path.display());
            println!();
            
            self.load_file(file_path)?;
            self.print_tree();
            
            if self.files.len() > 1 {
                println!();
                println!("=====================================");
                println!();
            }
        }
        Ok(())
    }
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