use anyhow::Result;
use crossterm::{
    cursor, execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use std::io::{self, Write};

use crate::tree::{MetadataInfo, TensorInfo, TreeNode};
use crate::utils::{format_parameters, format_shape, format_size};

pub struct DrawConfig<'a> {
    pub tree: &'a [(TreeNode, usize)],
    pub current_file: &'a str,
    pub file_idx: usize,
    pub total_files: usize,
    pub total_parameters: usize,
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub search_mode: bool,
    pub search_query: &'a str,
}

pub struct UI;

impl UI {
    pub fn draw_screen(config: &DrawConfig) -> Result<usize> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        let (_, terminal_height) = terminal::size()?;
        let header_height = 3;
        let footer_height = 2;
        let available_height =
            (terminal_height as usize).saturating_sub(header_height + footer_height);

        // Header
        writeln!(
            stdout,
            "SafeTensors Explorer - {} ({}/{})\r",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        )?;
        if config.search_mode {
            writeln!(
                stdout,
                "SEARCH MODE: {} | Type to search, Enter/Esc to exit search\r",
                if config.search_query.is_empty() {
                    "_"
                } else {
                    config.search_query
                }
            )?;
        } else {
            writeln!(
                stdout,
                "Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, q to quit\r"
            )?;
        }
        writeln!(stdout, "{}\r", "=".repeat(80))?;

        // Calculate scroll offset
        let new_scroll_offset = if config.selected_idx >= config.scroll_offset + available_height {
            config.selected_idx.saturating_sub(available_height - 1)
        } else if config.selected_idx < config.scroll_offset {
            config.selected_idx
        } else {
            config.scroll_offset
        };

        // Draw tree
        for (actual_index, (node, depth)) in config
            .tree
            .iter()
            .enumerate()
            .skip(new_scroll_offset)
            .take(available_height)
        {
            let is_selected = actual_index == config.selected_idx;

            if is_selected {
                execute!(
                    stdout,
                    SetForegroundColor(Color::Black),
                    crossterm::style::SetBackgroundColor(Color::White)
                )?;
            }

            Self::draw_node(node, *depth, &mut stdout)?;

            if is_selected {
                execute!(stdout, ResetColor)?;
            }
        }

        // Footer
        execute!(stdout, cursor::MoveTo(0, terminal_height - 1))?;
        if config.search_mode && config.tree.is_empty() {
            writeln!(
                stdout,
                "No results found for \"{}\" | Press Esc to exit search\r",
                config.search_query
            )?;
        } else {
            writeln!(
                stdout,
                "Total Parameters: {} | Selected: {}/{} | Scroll: {} | Matches: {}\r",
                format_parameters(config.total_parameters),
                config.selected_idx + 1,
                config.tree.len(),
                new_scroll_offset,
                config.tree.len()
            )?;
        }

        stdout.flush()?;
        Ok(new_scroll_offset)
    }

    fn draw_node(node: &TreeNode, depth: usize, stdout: &mut io::Stdout) -> Result<()> {
        let indent = "  ".repeat(depth);

        match node {
            TreeNode::Group {
                name,
                expanded,
                tensor_count,
                total_size,
                ..
            } => {
                let icon = if *expanded { "▼" } else { "▶" };
                writeln!(
                    stdout,
                    "{}{} 📁 {} ({} tensors, {})\r",
                    indent,
                    icon,
                    name,
                    tensor_count,
                    format_size(*total_size)
                )?;
            }
            TreeNode::Tensor { info } => {
                // In search mode (depth 0), show full name; otherwise show short name
                let display_name = if depth == 0 {
                    &info.name
                } else {
                    info.name.split('.').next_back().unwrap_or(&info.name)
                };
                writeln!(
                    stdout,
                    "{}  📄 {} [{}, {}, {}]\r",
                    indent,
                    display_name,
                    info.dtype,
                    format_shape(&info.shape),
                    format_size(info.size_bytes)
                )?;
            }
            TreeNode::Metadata { info } => {
                let truncated_value = if info.value.len() > 50 {
                    format!("{}...", &info.value[..47])
                } else {
                    info.value.clone()
                };
                writeln!(
                    stdout,
                    "{}  🏷️  {} [{}]: {}\r",
                    indent, info.name, info.value_type, truncated_value
                )?;
            }
        }
        Ok(())
    }

    pub fn draw_tensor_detail(tensor: &TensorInfo) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        writeln!(stdout, "Tensor Details\r")?;
        writeln!(stdout, "==============\r")?;
        writeln!(stdout, "Name: {}\r", tensor.name)?;
        writeln!(stdout, "Data Type: {}\r", tensor.dtype)?;
        writeln!(stdout, "Shape: {}\r", format_shape(&tensor.shape))?;
        writeln!(stdout, "Size: {}\r", format_size(tensor.size_bytes))?;
        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;

        stdout.flush()?;
        Ok(())
    }

    pub fn draw_metadata_detail(metadata: &MetadataInfo) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        writeln!(stdout, "Metadata Details\r")?;
        writeln!(stdout, "================\r")?;
        writeln!(stdout, "Key: {}\r", metadata.name)?;
        writeln!(stdout, "Type: {}\r", metadata.value_type)?;
        writeln!(stdout, "Value:\r")?;

        // Handle multi-line values or long values
        let lines = metadata.value.lines();
        for line in lines.take(20) {
            // Limit to 20 lines
            writeln!(stdout, "  {line}\r")?;
        }

        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;

        stdout.flush()?;
        Ok(())
    }
}
