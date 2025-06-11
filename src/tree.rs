use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub size_bytes: usize,
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
}

impl TreeNode {
    pub fn name(&self) -> &str {
        match self {
            TreeNode::Group { name, .. } => name,
            TreeNode::Tensor { info } => &info.name,
        }
    }
}

pub struct TreeBuilder;

impl TreeBuilder {
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
                tensors.sort_by(|a, b| a.name.cmp(&b.name));
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

        tree.sort_by(|a, b| a.name().cmp(b.name()));
        tree
    }

    fn build_subtree(tensors: &[TensorInfo], prefix: &str) -> Vec<TreeNode> {
        let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        let mut direct_tensors = Vec::new();

        for tensor in tensors {
            let remaining = tensor
                .name
                .strip_prefix(&format!("{}.", prefix))
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
            let full_prefix = format!("{}.{}", prefix, group_name);
            let children = Self::build_subtree(&group_tensors, &full_prefix);

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

    pub fn toggle_node_by_name(target_name: &str, nodes: &mut [TreeNode]) {
        for node in nodes {
            match node {
                TreeNode::Group {
                    name,
                    expanded,
                    children,
                    ..
                } => {
                    if name == target_name {
                        *expanded = !*expanded;
                        return;
                    }
                    Self::toggle_node_by_name(target_name, children);
                }
                TreeNode::Tensor { .. } => {}
            }
        }
    }
}
