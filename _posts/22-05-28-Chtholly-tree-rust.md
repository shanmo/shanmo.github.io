---
layout: post
title: "How to publish a package in Rust"
author: "Timothy Shan"
tags: rust
---

## Introduction 

I have been using Rust for about 2 months now, and I think it's a good time for me to contribute to this community. As a relatively new language, Rust adopts many good practices, e.g. when you create a workspace, the version control system is automatically initialized. Another advantage of using Rust is that publishing packages to `crates.io` is very easy, let's go through that in this blog. 

I came across a data structure called `Chtholly Tree` since it's named after an animation character. In my opinion it's kind of similar to `segment tree`, but Chtholly Tree focuses more on updating the values of intervals. For segment tree, when we want to update a range of values in a interval, one normally uses `lazy propagation`. We maintain another tree called lazy tree, to store the values that need to be propagated. Those values are only propagated when we query the corresponding range. However, I find Chtholly Tree easier to understand compared with segment tree with lazy propagation in this particular case. 

There is only one crate for Chtholly Tree in Rust [here](https://docs.rs/chtholly/latest/chtholly/), which does not include a readme, no github repo, and only supports initializing the tree via a vector. Due to the limitations of this crate, I've decided to implement my own version and publish on crates.io. 

## Chtholly Tree

Each node in Chtholly Tree has an interval parameterized by `[left, right]`, and a `value`. The Chtholly Tree structure itself has a set of nodes, and two methods. 

- The first method is `split`, where the input is a position, and its function is to split the intervals based on the given position. For instance, the interval is `[1, 5]`, and the position is `3`, then the interval splits into `[1, 2], [3, 5]`. 

- Another method is `assign`, given an interval `[left, right]` with a value, we first find all intervals of the tree within this range, and then merge them and update their value to the given one. 

The implementation in Python is below:
```python
class TreeNode:
    def __init__(self, l, r, v):
        self.left = l
        self.right = r 
        self.val = v

class ChthollyTree:
    def __init__(self, ):
        self.nodes = []

    def split(self, pos):
        n = len(self.nodes)
        left, right = 0, n-1
        while left <= right: 
            mid = (right - left)//2 + left 
            if self.nodes[mid].left < pos:
                left = mid + 1 
            elif self.nodes[mid].left > pos:
                right = mid - 1 
            else: 
                # no need to split if node.left == pos  
                return mid
        left -= 1 
        l, r, v = self.nodes[left].left, self.nodes[left].right, self.nodes[left].val 
        del self.nodes[left]
        tr = TreeNode(l, pos-1, v)  
        self.nodes.insert(left, tr)
        tr = TreeNode(pos, r, v)
        self.nodes.insert(left+1, tr)
        return left+1

    def assign(self, l, r, v): 
        if len(self.nodes) == 0: 
            tr = TreeNode(l, r, v) 
            self.nodes.append(tr)
            return 0 
        itr = self.split(r+1)
        itl = self.split(l) 
        del self.nodes[itl:itr]
        tr = TreeNode(l, r, v) 
        self.nodes.insert(itl, tr) 
```

The Rust version is:
```rust
/// Chtholly Tree allows queries and updates to be performed in a interval.
/// The input data needs to be random for it to achieve a time complexity of `O(nloglogn)`.
/// Representation of Chtholly Node used to build Chtholly Tree.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeNode {
    left: i32,
    right: i32,
    value: i32,
}

impl TreeNode {
    /// Creates a new TreeNode based on the interval `[left, right]` and `value`.
    pub fn new(left: i32, right: i32, value: i32) -> Self {
        TreeNode {
            left: left,
            right: right,
            value: value,
        }
    }
}

/// Representation of Chtholly Tree. The nodes are sorted based on `left` in interval `[left, right]`.  
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ChthollyTree {
    nodes: Vec<TreeNode>,
}

impl ChthollyTree {
    pub fn new() -> Self {
        ChthollyTree { nodes: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    /// Split the intervals in the tree based on the given position.
    /// E.g. interval is [1, 5] and position is 3, then the interval will be split into [1, 2] and [3, 5].
    pub fn split(&mut self, pos: i32) -> usize {
        let n: usize = self.nodes.len();
        let mut left: usize = 0;
        let mut right: usize = n - 1;

        // use binary search to find the first interval
        // whose left end is smaller than or equal to pos
        // can also use built-in `binary_search`
        while left <= right {
            let mid = (right - left) / 2 + left;
            if self.nodes[mid].left < pos {
                left = mid + 1;
            } else if self.nodes[mid].left > pos {
                right = mid - 1;
            } else {
                return mid;
            }
        }

        left -= 1;
        let l = self.nodes[left].left;
        let r = self.nodes[left].right;
        let v = self.nodes[left].value;

        self.nodes.remove(left);
        let tr = TreeNode::new(l, pos - 1, v);
        self.nodes.insert(left, tr);
        let tr = TreeNode::new(pos, r, v);
        self.nodes.insert(left + 1, tr);
        return left + 1;
    }

    /// Assign the value to given interval,
    /// and remove all intervals in the tree that are covered by the given interval.
    pub fn assign(&mut self, left: i32, right: i32, value: i32) {
        // check whether there are nodes in the tree
        if self.is_empty() {
            let tr = TreeNode::new(left, right, value);
            self.nodes.push(tr);
            return;
        }

        // delete the intervals between itl and itr
        let itr = self.split(right + 1);
        let itl = self.split(left);
        let mut index = 0;
        self.nodes.retain(|_| {
            index += 1;
            index < itl + 1 || index >= itr + 1
        });

        // insert the new interval
        let tr = TreeNode::new(left, right, value);
        self.nodes.insert(itl, tr);
    }
}
```

## Publish the package 

To publish the package, you need to create an account on crates.io, and visit the `Account Settings` page to get a token. Then in your terminal, enter `cargo login` and paste the token. We need to prepare the information below prior to publishing:
```
license or license-file
description
homepage
documentation
repository
readme
keywords
categories
```

What the official Rust guidebook does not mention, is that you also need to have a verified email, which could be added via `https://crates.io/me`.  Moreover, you also need to do a git push so that all changes are committed to the remote repo. 

When you are ready to publish, it's a good idea to test it first, by doing 
```
cargo publish --dry-run
```

The step above will 
- Perform some verification checks on your package.
- Compress your source code into a .crate file.
- Extract the .crate file into a temporary directory and verify that it compiles.

We need to check the `.crate` located in `target/package`, and make sure its size is below `10MB`. To ensure there are no large assests in the file, use 
```
cargo package --list
```

If everything looks OK, then you can publish the package via 
```
cargo publish
```

For my Rust package, the `Cargo.toml` looks like 
```
[package]
name = "chtholly_tree"
description = "Rust bindings for Chtholly Tree"
repository = "https://github.com/shanmo/chtholly_tree.git"
version = "0.1.0"
edition = "2021"
license = "MIT"
readme = "README.md"
keywords = ["interval", "query", "update"]
categories = ["data-structures", "algorithms"]
```

The output for `cargo package --list` is 
```
.cargo_vcs_info.json
.gitignore
Cargo.toml
Cargo.toml.orig
LICENSE
README.md
src/chtholly_tree.rs
src/lib.rs
```

The output for `cargo publish --dry-run` says `warning: aborting upload due to dry run`, and there are no errors, time to publish it! The crate is accessible right after you hit publish. For instance, check out mine [here](https://crates.io/crates/chtholly_tree), and the doc is included [here](https://docs.rs/chtholly_tree/0.1.0/chtholly_tree/). 

You cannot delete or modify the version you published, but you can publish a new version. In my case, I found that the documentation is not generated since I did not put my functions in the `lib.rs` file, so I modified that and published a new one. The update on crates.io is instant but not on `https://docs.rs/`.  After about 5 minutes, the doc is updated [here](https://docs.rs/chtholly_tree/1.0.0/chtholly_tree/#). 

## Reference 

- [Publishing on crates.io](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [Crate chtholly](https://docs.rs/chtholly/latest/chtholly/)
