---
https://www.luogu.com.cn/blog/blaze/solution-cf896clayout: post
title: "Introduction to ROS2 with Rust"
author: "Timothy Shan"
tags: ros2
---

## Introduction 

In May 2022, ROS 2 Humble Hawksbill (humble) was released, which supports Ubuntu 22.04. Since I am still using Ubuntu 20.04, this blog will focus on `ROS 2 Galactic Geochelone` (galactic) instead, released in May, 2021. 

First thing first, how to install ROS2? Because our team is migrating from ROS1 to ROS2, I need to use both for now. My current way is to install ROS2 on my OS following the [official guide](https://docs.ros.org/en/galactic/Installation.html), and install ROS1 via `mamba` using [RoboStack](https://robostack.github.io/). Even though RoboStack also provides `ros-galactic-desktop`, I do not recommend it since the package support is not complete, and does not work well with Python APIs. 

One may wonder whether there is a way to just taste the flavour of ROS2, see whether one likes it, before diving into the full-scale installation. There are [docker files](https://github.com/athackst/dockerfiles) for those who have nvidia GPU, whereas [this repo](https://github.com/Tiryoh/docker-ros2-desktop-vnc) provides docker files for CPU and VNC. The one used in this blog is the VNC version. After pulling and building the docker file, use a browser to connect to `http://127.0.0.1:6080/`, which shows the desktop. Also need to take care of the the permission issue via `sudo chmod 777 -R ~/.ros/` and you are good to go. 

For the Rust interface, I am going to use [r2r](https://github.com/sequenceplanner/r2r), which has examples on how to use `tokio`. Other Rust interfaces are also available, such as [ros2_rust](https://github.com/ros2-rust/ros2_rust), which in active development, but does not support tokio yet. The code for this blog is in [this repo](https://github.com/shanmo/learn-ros2). 

## Hello World

Let's begin with the good old `hello world` example. First, create a cargo binary package

```rust
cargo new hello_world --bin --vcs none
```

In the `src/main.rs`, add the following

```rust
use r2r::QosProfile;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "testnode", "")?;
    let duration = std::time::Duration::from_millis(2500);

    let mut timer = node.create_wall_timer(duration)?;
    let publisher =
        node.create_publisher::<r2r::std_msgs::msg::String>("/hw_topic", QosProfile::default())?;

    task::spawn(async move {
        loop {
            timer.tick().await.unwrap();
            let msg = r2r::std_msgs::msg::String {
                data: "hello world".to_string(),
            };
            publisher.publish(&msg).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }); 

    // here we spin the node in its own thread (but we could just busy wait in this thread)
    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
```

The `Cargo.toml` looks like this 

```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
r2r = "0.6.2"
tokio = { version = "1.15.0", features = ["full"] }
```

Next run `Cargo run` and checkout the topics via `ros2 topic list`, output is 

```
/hw_topic
/parameter_events
/rosout
```

To check the data, use `ros2 topic echo /hw_topic`

```
data: hello world
---
data: hello world
---
...
```

## A simple publisher

I am a fan of `Sherlock Holmes` so I will use that as an example, especially the [one filed by BBC](https://www.bbc.co.uk/programmes/b018ttws).  

## Reference 

- [ROS 2 workshop](https://ros2-industrial-workshop.readthedocs.io/en/latest/)
- [d2l-ros2](https://github.com/fishros/d2l-ros2) a ROS2 course in Chinese 
