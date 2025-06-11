# SafeTensors Explorer

An interactive terminal-based explorer for SafeTensors files, designed to help you visualize and navigate the structure of machine learning models.

## Features

- 🔍 **Interactive browsing** of SafeTensors file structure
- 📁 **Hierarchical tree view** with expandable/collapsible groups
- 📊 **Tensor details** including shape, data type, and size
- 🔗 **Multi-file support** - automatically merges multiple SafeTensors files into a unified view
- 📏 **Human-readable sizes** (B, KB, MB, GB)
- ⌨️ **Keyboard navigation** for smooth exploration

## Installation

### Prerequisites
- Rust (1.70 or later)

### Build from source
```bash
git clone <repository-url>
cd safetensors_explorer
cargo build --release
```

## Usage

### Basic usage
```bash
# Explore a single SafeTensors file
cargo run -- model.safetensors

# Or use the compiled binary
./target/release/safetensors_explorer model.safetensors
```

### Multi-file exploration
```bash
# Explore multiple files as a unified model
cargo run -- model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors

# Using wildcards
cargo run -- *.safetensors
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate up/down through the tree |
| `Enter` / `Space` | Expand/collapse groups, view tensor details |
| `q` | Quit the application |
| `Ctrl+C` | Force quit |

## Example Output

```
SafeTensors Explorer - model.safetensors (1/1)
Use ↑/↓ to navigate, Enter/Space to expand/collapse, q to quit
================================================================================

▼ 📁 transformer (123 tensors, 1.2 GB)
  ▼ 📁 h (120 tensors, 1.1 GB)
    ▼ 📁 0 (5 tensors, 45.2 MB)
      📄 attn.c_attn.weight [Float16, (4096, 3072), 25.2 MB]
      📄 attn.c_proj.weight [Float16, (1024, 4096), 8.4 MB]
      📄 ln_1.weight [Float16, (4096,), 8.2 KB]
      📄 mlp.c_fc.weight [Float16, (4096, 11008), 90.1 MB]
      📄 mlp.c_proj.weight [Float16, (11008, 4096), 90.1 MB]
    ▶ 📁 1 (5 tensors, 45.2 MB)
    ▶ 📁 2 (5 tensors, 45.2 MB)
    ...
    ▶ 📁 31 (5 tensors, 45.2 MB)
  📄 ln_f.weight [Float16, (4096,), 8.2 KB]
  📄 wte.weight [Float16, (151936, 4096), 1.2 GB]

Selected: 1/342 | Scroll: 0
```

## Technical Details

### Supported Formats
- SafeTensors files (`.safetensors`)
- All tensor data types supported by the SafeTensors format

### Performance
- Memory efficient: Only loads tensor metadata, not the actual tensor data
- Fast startup: Optimized for quick exploration of large models
- Responsive UI: Smooth navigation even with thousands of tensors

## Dependencies

- `safetensors` - For reading SafeTensors files
- `crossterm` - For terminal UI and keyboard input
- `clap` - For command-line argument parsing
- `anyhow` - For error handling

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
