use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use safetensors::SafeTensors;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read, Write},
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
    current_file_idx: usize,
    selected_idx: usize,
    scroll_offset: usize,
    flattened_tree: Vec<(TreeNode, usize)>,
}

impl Explorer {
    fn new(files: Vec<PathBuf>) -> Self {
        Self {
            files,
            tensors: Vec::new(),
            tree: Vec::new(),
            current_file_idx: 0,
            selected_idx: 0,
            scroll_offset: 0,
            flattened_tree: Vec::new(),
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
                root_map.entry(prefix).or_default().push(tensor.clone());
            } else {
                root_map.entry("_root".to_string()).or_default().push(tensor.clone());
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
        self.flatten_tree();
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
                groups.entry(next_prefix).or_default().push(tensor.clone());
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
                let short_name = info.name.split('.').next_back().unwrap_or(&info.name);
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

    fn flatten_tree(&mut self) {
        self.flattened_tree.clear();
        let tree_clone = self.tree.clone();
        for node in &tree_clone {
            self.flatten_node(node, 0);
        }
    }

    fn flatten_node(&mut self, node: &TreeNode, depth: usize) {
        self.flattened_tree.push((node.clone(), depth));
        
        if let TreeNode::Group { children, expanded, .. } = node {
            if *expanded {
                for child in children {
                    self.flatten_node(child, depth + 1);
                }
            }
        }
    }

    fn run(&mut self) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }

        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;
        
        let result = self.interactive_loop();
        
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Show)?;
        terminal::disable_raw_mode()?;
        
        result
    }

    fn interactive_loop(&mut self) -> Result<()> {
        self.load_file(&self.files[self.current_file_idx].clone())?;
        
        loop {
            self.draw_screen()?;
            
            if let Event::Key(key_event) = event::read()? {
                match key_event {
                    KeyEvent { code: KeyCode::Char('q'), .. } => break,
                    KeyEvent { code: KeyCode::Char('c'), modifiers: KeyModifiers::CONTROL, .. } => break,
                    KeyEvent { code: KeyCode::Up, .. } => self.move_selection(-1),
                    KeyEvent { code: KeyCode::Down, .. } => self.move_selection(1),
                    KeyEvent { code: KeyCode::Enter, .. } | KeyEvent { code: KeyCode::Char(' '), .. } => {
                        self.toggle_current_node();
                    },
                    KeyEvent { code: KeyCode::Left, .. } => self.previous_file(),
                    KeyEvent { code: KeyCode::Right, .. } => self.next_file(),
                    _ => {},
                }
            }
        }
        
        Ok(())
    }

    fn draw_screen(&mut self) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::MoveTo(0, 0))?;
        
        let (_, terminal_height) = terminal::size()?;
        let header_height = 3;
        let footer_height = 2;
        let available_height = (terminal_height as usize).saturating_sub(header_height + footer_height);
        
        writeln!(stdout, "SafeTensors Explorer - {} ({}/{})\r",
            self.files[self.current_file_idx].display(),
            self.current_file_idx + 1,
            self.files.len())?;
        writeln!(stdout, "Use ↑/↓ to navigate, Enter/Space to expand/collapse, ←/→ for files, q to quit\r")?;
        writeln!(stdout, "{}\r", "=".repeat(80))?;
        
        if self.selected_idx >= self.scroll_offset + available_height {
            self.scroll_offset = self.selected_idx.saturating_sub(available_height - 1);
        } else if self.selected_idx < self.scroll_offset {
            self.scroll_offset = self.selected_idx;
        }
        
        let end_idx = (self.scroll_offset + available_height).min(self.flattened_tree.len());
        
        for i in self.scroll_offset..end_idx {
            let (node, depth) = &self.flattened_tree[i];
            let is_selected = i == self.selected_idx;
            
            if is_selected {
                execute!(stdout, SetForegroundColor(Color::Black), 
                        crossterm::style::SetBackgroundColor(Color::White))?;
            }
            
            self.draw_node(node, *depth, &mut stdout)?;
            
            if is_selected {
                execute!(stdout, ResetColor)?;
            }
        }
        
        execute!(stdout, cursor::MoveTo(0, terminal_height - 1))?;
        writeln!(stdout, "Selected: {}/{} | Scroll: {}\r", 
            self.selected_idx + 1, 
            self.flattened_tree.len(),
            self.scroll_offset)?;
        
        stdout.flush()?;
        Ok(())
    }

    fn draw_node(&self, node: &TreeNode, depth: usize, stdout: &mut io::Stdout) -> Result<()> {
        let indent = "  ".repeat(depth);
        
        match node {
            TreeNode::Group { name, expanded, tensor_count, total_size, .. } => {
                let icon = if *expanded { "▼" } else { "▶" };
                writeln!(stdout, "{}{} 📁 {} ({} tensors, {})\r", 
                    indent, icon, name, tensor_count, Self::format_size(*total_size))?;
            }
            TreeNode::Tensor { info } => {
                let short_name = info.name.split('.').next_back().unwrap_or(&info.name);
                writeln!(stdout, "{}  📄 {} [{}, {}, {}]\r", 
                    indent,
                    short_name,
                    info.dtype,
                    Self::format_shape(&info.shape),
                    Self::format_size(info.size_bytes)
                )?;
            }
        }
        Ok(())
    }

    fn move_selection(&mut self, delta: i32) {
        if self.flattened_tree.is_empty() {
            return;
        }
        
        let new_idx = if delta < 0 {
            self.selected_idx.saturating_sub((-delta) as usize)
        } else {
            (self.selected_idx + delta as usize).min(self.flattened_tree.len() - 1)
        };
        
        self.selected_idx = new_idx;
    }

    fn toggle_current_node(&mut self) {
        if self.selected_idx < self.flattened_tree.len() {
            let (selected_node, _) = &self.flattened_tree[self.selected_idx];
            
            if let TreeNode::Group { name, .. } = selected_node {
                let name = name.clone();
                let mut tree_clone = self.tree.clone();
                self.toggle_node_by_name(&name, &mut tree_clone);
                self.tree = tree_clone;
                self.flatten_tree();
            }
        }
    }

    fn toggle_node_by_name(&self, target_name: &str, nodes: &mut [TreeNode]) {
        for node in nodes {
            match node {
                TreeNode::Group { name, expanded, children, .. } => {
                    if name == target_name {
                        *expanded = !*expanded;
                        return;
                    }
                    self.toggle_node_by_name(target_name, children);
                }
                TreeNode::Tensor { .. } => {}
            }
        }
    }

    fn next_file(&mut self) {
        if self.files.len() > 1 {
            self.current_file_idx = (self.current_file_idx + 1) % self.files.len();
            self.selected_idx = 0;
            self.scroll_offset = 0;
            if let Err(_) = self.load_file(&self.files[self.current_file_idx].clone()) {
                // Handle error silently for now
            }
        }
    }

    fn previous_file(&mut self) {
        if self.files.len() > 1 {
            self.current_file_idx = if self.current_file_idx > 0 {
                self.current_file_idx - 1
            } else {
                self.files.len() - 1
            };
            self.selected_idx = 0;
            self.scroll_offset = 0;
            if let Err(_) = self.load_file(&self.files[self.current_file_idx].clone()) {
                // Handle error silently for now
            }
        }
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