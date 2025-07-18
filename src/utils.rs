pub fn format_shape(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn format_size(bytes: usize) -> String {
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

pub fn format_parameters(params: usize) -> String {
    if params < 1_000 {
        format!("{params}")
    } else if params < 1_000_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else if params < 1_000_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    }
}
