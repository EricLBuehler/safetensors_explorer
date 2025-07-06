use anyhow::{Context, Result};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{self, ClearType},
};
use safetensors::SafeTensors;
use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read},
    path::PathBuf,
};

use crate::gguf::GGUFFile;

use crate::tree::{MetadataInfo, TensorInfo, TreeBuilder, TreeNode, natural_sort_key};
use crate::ui::UI;

pub struct Explorer {
    files: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
    tree: Vec<TreeNode>,
    selected_idx: usize,
    scroll_offset: usize,
    flattened_tree: Vec<(TreeNode, usize)>,
    total_parameters: usize,
}

impl Explorer {
    pub fn new(files: Vec<PathBuf>) -> Self {
        Self {
            files,
            tensors: Vec::new(),
            metadata: Vec::new(),
            tree: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            flattened_tree: Vec::new(),
            total_parameters: 0,
        }
    }

    fn load_all_files(&mut self) -> Result<()> {
        self.tensors.clear();
        self.metadata.clear();

        let files = self.files.clone();
        for file_path in &files {
            let extension = file_path.extension().and_then(|s| s.to_str());

            match extension {
                Some("safetensors") => {
                    self.load_safetensors_file(file_path)?;
                }
                Some("gguf") => {
                    self.load_gguf_file(file_path)?;
                }
                _ => {
                    eprintln!("Warning: Unsupported file format: {}", file_path.display());
                }
            }
        }

        // Deduplicate tensors by name
        let mut seen_names = HashSet::new();
        self.tensors
            .retain(|tensor| seen_names.insert(tensor.name.clone()));

        self.tensors
            .sort_by(|a, b| natural_sort_key(&a.name).cmp(&natural_sort_key(&b.name)));
        self.total_parameters = self.tensors.iter().map(|t| t.num_elements).sum::<usize>();
        self.build_tree();
        Ok(())
    }

    fn load_safetensors_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let tensors = SafeTensors::deserialize(&buffer).with_context(|| {
            format!("Failed to parse SafeTensors file: {}", file_path.display())
        })?;

        for name in tensors.names() {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape().to_vec();
            let num_elements = shape.iter().product::<usize>();
            let dtype = format!("{:?}", tensor.dtype());
            let size_bytes = tensor.data().len();

            self.tensors.push(TensorInfo {
                name: name.to_string(),
                dtype,
                shape,
                size_bytes,
                num_elements,
            });
        }

        Ok(())
    }

    fn load_gguf_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let gguf = GGUFFile::read(&buffer)
            .with_context(|| format!("Failed to parse GGUF file: {}", file_path.display()))?;

        // Load metadata
        for (key, value) in &gguf.metadata {
            let value_type = match value {
                crate::gguf::GGUFValue::U8(_) => "u8",
                crate::gguf::GGUFValue::I8(_) => "i8",
                crate::gguf::GGUFValue::U16(_) => "u16",
                crate::gguf::GGUFValue::I16(_) => "i16",
                crate::gguf::GGUFValue::U32(_) => "u32",
                crate::gguf::GGUFValue::I32(_) => "i32",
                crate::gguf::GGUFValue::F32(_) => "f32",
                crate::gguf::GGUFValue::U64(_) => "u64",
                crate::gguf::GGUFValue::I64(_) => "i64",
                crate::gguf::GGUFValue::F64(_) => "f64",
                crate::gguf::GGUFValue::Bool(_) => "bool",
                crate::gguf::GGUFValue::String(_) => "string",
                crate::gguf::GGUFValue::Array(_) => "array",
            };

            self.metadata.push(MetadataInfo {
                name: key.clone(),
                value: value.to_string(),
                value_type: value_type.to_string(),
            });
        }

        // Load tensors
        for tensor in &gguf.tensors {
            let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
            let dtype = tensor.tensor_type.to_string();

            // Calculate size using the element size from our custom implementation
            let num_elements = shape.iter().product::<usize>();
            let size_bytes =
                (num_elements as f32 * tensor.tensor_type.element_size_bytes()) as usize;

            self.tensors.push(TensorInfo {
                name: tensor.name.clone(),
                dtype,
                shape,
                size_bytes,
                num_elements,
            });
        }

        Ok(())
    }

    fn build_tree(&mut self) {
        if self.metadata.is_empty() {
            self.tree = TreeBuilder::build_tree(&self.tensors);
        } else {
            self.tree = TreeBuilder::build_tree_mixed(&self.tensors, &self.metadata);
        }
        self.flatten_tree();
    }

    fn flatten_tree(&mut self) {
        self.flattened_tree = TreeBuilder::flatten_tree(&self.tree);
    }

    pub fn run(&mut self) -> Result<()> {
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
        self.load_all_files()?;

        loop {
            let title = if self.files.len() == 1 {
                self.files[0].to_string_lossy().to_string()
            } else {
                "SafeTensors Model".to_string()
            };

            self.scroll_offset = UI::draw_screen(
                &self.flattened_tree,
                &title,
                0,
                1,
                self.total_parameters,
                self.selected_idx,
                self.scroll_offset,
            )?;

            if let Event::Key(key_event) = event::read()? {
                match key_event {
                    KeyEvent {
                        code: KeyCode::Char('q'),
                        ..
                    } => break,
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    } => break,
                    KeyEvent {
                        code: KeyCode::Up, ..
                    } => self.move_selection(-1),
                    KeyEvent {
                        code: KeyCode::Down,
                        ..
                    } => self.move_selection(1),
                    KeyEvent {
                        code: KeyCode::Enter,
                        ..
                    }
                    | KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } => {
                        self.handle_selection();
                    }
                    // Remove left/right file navigation since we're showing all files merged
                    _ => {}
                }
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

    fn handle_selection(&mut self) {
        if self.selected_idx < self.flattened_tree.len() {
            let (selected_node, _) = &self.flattened_tree[self.selected_idx];

            match selected_node {
                TreeNode::Group { .. } => {
                    let mut tree_clone = self.tree.clone();
                    let _ = TreeBuilder::toggle_node_by_index(self.selected_idx, &mut tree_clone);
                    self.tree = tree_clone;
                    self.flatten_tree();
                }
                TreeNode::Tensor { info } => {
                    self.show_tensor_detail(info);
                }
                TreeNode::Metadata { info } => {
                    self.show_metadata_detail(info);
                }
            }
        }
    }

    fn show_tensor_detail(&self, tensor: &TensorInfo) {
        if UI::draw_tensor_detail(tensor).is_ok() {
            // Wait for any key press
            let _ = event::read();
        }
    }

    fn show_metadata_detail(&self, metadata: &MetadataInfo) {
        if UI::draw_metadata_detail(metadata).is_ok() {
            // Wait for any key press
            let _ = event::read();
        }
    }
}
