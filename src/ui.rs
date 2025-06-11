use anyhow::Result;
use crossterm::{
    cursor, execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use std::io::{self, Write};

use crate::tree::{TensorInfo, TreeNode};
use crate::utils::{format_shape, format_size};

pub struct UI;

impl UI {
    pub fn draw_screen(
        tree: &[(TreeNode, usize)],
        current_file: &str,
        file_idx: usize,
        total_files: usize,
        selected_idx: usize,
        scroll_offset: usize,
    ) -> Result<usize> {
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
            current_file,
            file_idx + 1,
            total_files
        )?;
        writeln!(
            stdout,
            "Use ↑/↓ to navigate, Enter/Space to expand/collapse, q to quit\r"
        )?;
        writeln!(stdout, "{}\r", "=".repeat(80))?;

        // Calculate scroll offset
        let new_scroll_offset = if selected_idx >= scroll_offset + available_height {
            selected_idx.saturating_sub(available_height - 1)
        } else if selected_idx < scroll_offset {
            selected_idx
        } else {
            scroll_offset
        };

        // Draw tree
        for (actual_index, (node, depth)) in tree
            .iter()
            .enumerate()
            .skip(new_scroll_offset)
            .take(available_height)
        {
            let is_selected = actual_index == selected_idx;

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
        writeln!(
            stdout,
            "Selected: {}/{} | Scroll: {}\r",
            selected_idx + 1,
            tree.len(),
            new_scroll_offset
        )?;

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
                let short_name = info.name.split('.').next_back().unwrap_or(&info.name);
                writeln!(
                    stdout,
                    "{}  📄 {} [{}, {}, {}]\r",
                    indent,
                    short_name,
                    info.dtype,
                    format_shape(&info.shape),
                    format_size(info.size_bytes)
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
}
