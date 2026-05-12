//! Monte Carlo Tree Search planner using kindle's learned world model.
//!
//! Tree-structured alternative to the CEM/random-shooting `planner.rs`.
//! Per-lane tree, expanded incrementally via UCB1 selection. Each
//! simulation:
//!
//!   1. Select: descend tree from root via UCB1 (Q + c*sqrt(ln(N)/n)) until
//!      hitting a node with an unexpanded valid action.
//!   2. Expand: pick one unexpanded action; defer WM forward to caller
//!      (so we can batch across lanes).
//!   3. Evaluate: caller computes novelty score on the predicted child
//!      latent and passes it back via `backup`.
//!   4. Backup: propagate score up the visited path, incrementing visit
//!      counts at each node.
//!
//! Output: the action at the root with highest child visit count (the
//! "most explored" action — the MCTS canonical policy).

use crate::adapter::MAX_ACTION_DIM;

/// One tree node. Stores its latent state, accumulated visit stats, and
/// per-action child pointers. `children[a] == None` means "action a not
/// yet expanded from this node."
#[derive(Clone)]
pub struct McNode {
    pub z: Vec<f32>,
    pub n: u32,
    pub w: f32,
    /// Index into `McTree::nodes` for each action's child node (when
    /// expanded), else `usize::MAX` sentinel.
    pub children: [usize; MAX_ACTION_DIM],
    /// Mask of valid action indices at this node (subset of
    /// 0..MAX_ACTION_DIM). The selector ignores indices not in this list.
    pub valid: Vec<u8>,
}

impl McNode {
    fn new(z: Vec<f32>, valid: Vec<u8>) -> Self {
        Self {
            z,
            n: 0,
            w: 0.0,
            children: [usize::MAX; MAX_ACTION_DIM],
            valid,
        }
    }

    /// Among VALID actions, pick the first unexpanded one. Returns the
    /// action index, or None if all valid actions are already expanded.
    fn first_unexpanded(&self) -> Option<u8> {
        self.valid
            .iter()
            .copied()
            .find(|&a| self.children[a as usize] == usize::MAX)
    }
}

/// Per-lane MCTS tree. Constructed fresh at each `plan_and_queue` call.
pub struct McTree {
    pub nodes: Vec<McNode>,
}

impl McTree {
    pub fn new(z_root: Vec<f32>, valid: Vec<u8>) -> Self {
        Self {
            nodes: vec![McNode::new(z_root, valid)],
        }
    }

    pub fn root(&self) -> &McNode {
        &self.nodes[0]
    }

    /// Select a path from root to either:
    ///   - a node with an unexpanded valid action (returns path + that action)
    ///   - a fully-expanded leaf with no children (returns path + None)
    ///
    /// During traversal each interior node is chosen by UCB1 (Q + c*sqrt(ln N_parent / n_child)).
    /// Path is the sequence of node indices from root to selected node.
    pub fn select(&self, c_puct: f32) -> (Vec<usize>, Option<u8>) {
        let mut path = vec![0usize]; // start at root
        let mut cur = 0usize;
        loop {
            let node = &self.nodes[cur];
            // If this node has an unexpanded action, return it for expansion.
            if let Some(action) = node.first_unexpanded() {
                return (path, Some(action));
            }
            // All valid actions are expanded; pick UCB1-best child to descend.
            // If the node has no valid actions at all, return (rare; treat as leaf).
            if node.valid.is_empty() {
                return (path, None);
            }
            let parent_n = node.n.max(1) as f32;
            let log_parent = parent_n.ln().max(0.0);
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child: Option<usize> = None;
            for &a in &node.valid {
                let child_idx = node.children[a as usize];
                debug_assert_ne!(child_idx, usize::MAX, "all valid should be expanded here");
                let child = &self.nodes[child_idx];
                let q = if child.n > 0 {
                    child.w / child.n as f32
                } else {
                    0.0
                };
                let u = c_puct * (log_parent / (child.n as f32 + 1.0)).sqrt();
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_child = Some(child_idx);
                }
            }
            match best_child {
                Some(idx) => {
                    cur = idx;
                    path.push(cur);
                }
                None => return (path, None),
            }
        }
    }

    /// Append a new child node under `parent_idx` for `action`. Returns the
    /// new child's index. Caller computes `child_z` via WM forward.
    pub fn add_child(
        &mut self,
        parent_idx: usize,
        action: u8,
        child_z: Vec<f32>,
        child_valid: Vec<u8>,
    ) -> usize {
        let child_idx = self.nodes.len();
        self.nodes.push(McNode::new(child_z, child_valid));
        self.nodes[parent_idx].children[action as usize] = child_idx;
        child_idx
    }

    /// Propagate `value` up the given path (visit count++, sum+=value at
    /// each node).
    pub fn backup(&mut self, path: &[usize], value: f32) {
        for &idx in path {
            let n = &mut self.nodes[idx];
            n.n += 1;
            n.w += value;
        }
    }

    /// Canonical MCTS policy: at the root, return the action with the
    /// highest child visit count. Returns None when no children expanded.
    pub fn best_root_action(&self) -> Option<u8> {
        let root = &self.nodes[0];
        let mut best_a: Option<u8> = None;
        let mut best_n: u32 = 0;
        for &a in &root.valid {
            let ci = root.children[a as usize];
            if ci == usize::MAX {
                continue;
            }
            let n = self.nodes[ci].n;
            if n > best_n {
                best_n = n;
                best_a = Some(a);
            }
        }
        best_a
    }
}
