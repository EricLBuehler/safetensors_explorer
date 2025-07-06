use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub size_bytes: usize,
    pub num_elements: usize,
}

#[derive(Debug, Clone)]
pub struct MetadataInfo {
    pub name: String,
    pub value: String,
    pub value_type: String,
}

#[derive(Debug, Clone)]
pub enum TreeNode {
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
    Metadata {
        info: MetadataInfo,
    },
}

impl TreeNode {
    pub fn name(&self) -> &str {
        match self {
            TreeNode::Group { name, .. } => name,
            TreeNode::Tensor { info } => &info.name,
            TreeNode::Metadata { info } => &info.name,
        }
    }
}

pub fn natural_sort_key(name: &str) -> Vec<NaturalSortItem> {
    let mut result = Vec::new();
    let mut current_number = String::new();
    let mut current_text = String::new();

    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !current_text.is_empty() {
                result.push(NaturalSortItem::Text(current_text.clone()));
                current_text.clear();
            }
            current_number.push(ch);
        } else {
            if !current_number.is_empty() {
                if let Ok(num) = current_number.parse::<u32>() {
                    result.push(NaturalSortItem::Number(num));
                } else {
                    result.push(NaturalSortItem::Text(current_number.clone()));
                }
                current_number.clear();
            }
            current_text.push(ch);
        }
    }

    if !current_number.is_empty() {
        if let Ok(num) = current_number.parse::<u32>() {
            result.push(NaturalSortItem::Number(num));
        } else {
            result.push(NaturalSortItem::Text(current_number));
        }
    }
    if !current_text.is_empty() {
        result.push(NaturalSortItem::Text(current_text));
    }

    result
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NaturalSortItem {
    Text(String),
    Number(u32),
}

pub struct TreeBuilder;

impl TreeBuilder {
    pub fn build_tree_mixed(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Vec<TreeNode> {
        let mut tree = Vec::new();

        // Add metadata as a separate group
        if !metadata.is_empty() {
            let mut metadata_children = Vec::new();
            for meta in metadata {
                metadata_children.push(TreeNode::Metadata { info: meta.clone() });
            }
            metadata_children.sort_by_key(|a| natural_sort_key(a.name()));

            tree.push(TreeNode::Group {
                name: "🔧 Metadata".to_string(),
                children: metadata_children,
                expanded: false,
                tensor_count: 0,
                total_size: 0,
            });
        }

        // Build tensor tree
        let tensor_tree = Self::build_tree(tensors);
        tree.extend(tensor_tree);

        tree
    }

    pub fn build_tree(tensors: &[TensorInfo]) -> Vec<TreeNode> {
        let mut root_map: HashMap<String, Vec<TensorInfo>> = HashMap::new();

        for tensor in tensors {
            let parts: Vec<&str> = tensor.name.split('.').collect();
            if parts.len() > 1 {
                let prefix = parts[0].to_string();
                root_map.entry(prefix).or_default().push(tensor.clone());
            } else {
                root_map
                    .entry("_root".to_string())
                    .or_default()
                    .push(tensor.clone());
            }
        }

        let mut tree = Vec::new();
        for (prefix, mut tensors) in root_map {
            if prefix == "_root" {
                for tensor in tensors {
                    tree.push(TreeNode::Tensor { info: tensor });
                }
            } else {
                tensors.sort_by(|a, b| natural_sort_key(&a.name).cmp(&natural_sort_key(&b.name)));
                let tensor_count = tensors.len();
                let total_size = tensors.iter().map(|t| t.size_bytes).sum();

                let children = Self::build_subtree(&tensors, &prefix);

                tree.push(TreeNode::Group {
                    name: prefix,
                    children,
                    expanded: true,
                    tensor_count,
                    total_size,
                });
            }
        }

        tree.sort_by_key(|a| natural_sort_key(a.name()));
        tree
    }

    fn build_subtree(tensors: &[TensorInfo], prefix: &str) -> Vec<TreeNode> {
        let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        let mut direct_tensors = Vec::new();

        for tensor in tensors {
            let remaining = tensor
                .name
                .strip_prefix(&format!("{prefix}."))
                .unwrap_or(&tensor.name);
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
            let full_prefix = format!("{prefix}.{group_name}");
            let children = Self::build_subtree(&group_tensors, &full_prefix);

            result.push(TreeNode::Group {
                name: group_name,
                children,
                expanded: false,
                tensor_count,
                total_size,
            });
        }

        result.sort_by_key(|a| natural_sort_key(a.name()));
        result
    }

    pub fn flatten_tree(tree: &[TreeNode]) -> Vec<(TreeNode, usize)> {
        let mut flattened = Vec::new();
        for node in tree {
            Self::flatten_node(node, 0, &mut flattened);
        }
        flattened
    }

    fn flatten_node(node: &TreeNode, depth: usize, flattened: &mut Vec<(TreeNode, usize)>) {
        flattened.push((node.clone(), depth));

        if let TreeNode::Group {
            children, expanded, ..
        } = node
        {
            if *expanded {
                for child in children {
                    Self::flatten_node(child, depth + 1, flattened);
                }
            }
        }
    }

    pub fn toggle_node_by_index(target_idx: usize, nodes: &mut [TreeNode]) -> bool {
        let mut current_idx = 0;
        Self::toggle_node_by_index_recursive(target_idx, nodes, &mut current_idx)
    }

    fn toggle_node_by_index_recursive(
        target_idx: usize,
        nodes: &mut [TreeNode],
        current_idx: &mut usize,
    ) -> bool {
        for node in nodes {
            // Check if this is the target node
            if *current_idx == target_idx {
                if let TreeNode::Group { expanded, .. } = node {
                    *expanded = !*expanded;
                    return true;
                }
                return false; // Target was not a group
            }

            // Increment for this node
            *current_idx += 1;

            // If it's an expanded group, recurse into children
            if let TreeNode::Group {
                children, expanded, ..
            } = node
            {
                if *expanded
                    && Self::toggle_node_by_index_recursive(target_idx, children, current_idx)
                {
                    return true;
                }
            }
        }
        false
    }
}
