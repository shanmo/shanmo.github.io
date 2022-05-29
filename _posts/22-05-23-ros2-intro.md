---
layout: post
title: "Introduction to ROS2 with Rust"
author: "Timothy Shan"
tags: ros2
---

## Introduction 

In May 2022, ROS 2 Humble Hawksbill (humble) was released, which supports Ubuntu 22.04. Since I am still using Ubuntu 20.04, this blog will focus on `foxy`. 

First thing first, how to install ROS2? Because our team is migrating from ROS1 to ROS2, I need to use both for now. My current way is to install ROS2 on my OS following the [official guide](https://docs.ros.org/en/galactic/Installation.html), and install ROS1 via `mamba` using [RoboStack](https://robostack.github.io/). Even though RoboStack also provides `ros-galactic-desktop`, I do not recommend it since the package support is not complete, and does not work well with Python APIs. 

One may wonder whether there is a way to just taste the flavour of ROS2, see whether one likes it, before diving into the full-scale installation. There are [docker files](https://github.com/athackst/dockerfiles) for those who have nvidia GPU, whereas [this repo](https://github.com/Tiryoh/docker-ros2-desktop-vnc) provides docker files for CPU and VNC. The one used in this blog is the VNC version. After pulling and building the docker file, use a browser to connect to `http://127.0.0.1:6080/`, which shows the desktop. Also need to take care of the the permission issue via `sudo chmod 777 -R ~/.ros/` and you are good to go. 

## Hello World

For the Rust interface of this section, I am going to use [r2r](https://github.com/sequenceplanner/r2r), which has examples on how to use `tokio`. Other Rust interfaces are also available, such as [ros2_rust](https://github.com/ros2-rust/ros2_rust), which in active development, but does not support tokio yet. The code for this blog is in [this repo](https://github.com/shanmo/learn-ros2). 

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

I am a fan of `Sherlock Holmes` so I will use that as an example, especially the [one filed by BBC](https://www.bbc.co.uk/programmes/b018ttws). In the TV series, John Watson writes blogs about the cases Sherlock is dealing with. So let's create a publisher to publish Watson's blog, by 

```
cargo new watson --bin
```

The Rust interface used in this section is [rclrust](https://github.com/rclrust/rclrust), wich also supports `tokio`. Note that we can also create a Rust Client from scratch as shown [here](https://marshalshi.medium.com/create-a-rust-client-for-ros2-from-scratch-part-1-1-create-the-dynamic-library-via-cmake-empy-a93f78ae90d1). We need to define the dependencies in the Cargo.toml as 

```rust
[package]
name = "watson"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
rclrust = { git = "https://github.com/rclrust/rclrust.git", features = ["foxy"] }
rclrust-msg = { git = "https://github.com/rclrust/rclrust.git" }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

In code above, we can specify the ROS2 version for `rclrust` via cargo features. We also need to include `rclrust-msg`. 

The publisher code is 

```rust
use std::{thread::sleep, time::Duration};

use anyhow::Result;
use rclrust::{qos::QoSProfile, rclrust_info};
use rclrust_msg::std_msgs::msg::String as String_;

fn main() -> Result<()> {
    let ctx = rclrust::init()?;
    let node = ctx.create_node("watson_blog")?;
    let logger = node.logger();
    let publisher = node.create_publisher::<String_>("blog", &QoSProfile::default())?;

    let mut count = 1; 
    loop {
        publisher.publish(&String_ {
            data: format!("Watson's {}th blog", count),
        })?;
        rclrust_info!(logger, "Watson's {}th blog published", count);
        count += 1; 
        sleep(Duration::from_millis(100));
    }

    Ok(())
}
```

Output will be 

```
[INFO] [1653823142.928885221] [watson_blog]: Watson's 1th blog published
[INFO] [1653823143.029225096] [watson_blog]: Watson's 2th blog published
[INFO] [1653823143.129426721] [watson_blog]: Watson's 3th blog published
[INFO] [1653823143.230213513] [watson_blog]: Watson's 4th blog published
[INFO] [1653823143.330655138] [watson_blog]: Watson's 5th blog published
```

More examples could be found in [rclrust-examples](https://github.com/rclrust/rclrust-examples). 

## A simple subscriber 

A young man named Billy who lives at Baker Street enjoys reading Watson's blogs very much, so let's write a subscriber for Billy. The code is below 

```rust
use std::sync::Arc;

use rclrust::{qos::QoSProfile, rclrust_info};
use rclrust_msg::std_msgs::msg::String as String_;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = rclrust::init()?;
    let mut node = ctx.create_node("billy_reader")?;
    let logger = node.logger();

    let _subscription = node.create_subscription(
        "blog",
        move |msg: Arc<String_>| {
            rclrust_info!(logger, "I read: {}", msg.data);
        },
        &QoSProfile::default(),
    )?;

    node.wait();
    Ok(())
}
```

We need to run both publisher and subscriber, the output is 

```
[INFO] [1653826256.633754176] [billy_reader]: I read: Watson's 2th blog
[INFO] [1653826256.734067551] [billy_reader]: I read: Watson's 3th blog
[INFO] [1653826256.835288093] [billy_reader]: I read: Watson's 4th blog
```

Billy likes the blogs so much that he pays a little reward after reading each one. We need to publish an `u8` type to handle this feature as follows

```rust
use std::sync::Arc;

use rclrust::{qos::QoSProfile, rclrust_info};
use rclrust_msg::std_msgs::msg::String as String_;
use rclrust_msg::std_msgs::msg::UInt8 as u8_;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = rclrust::init()?;
    let mut node = ctx.create_node("billy_reader")?;
    let logger = node.logger();
    let publisher = node.create_publisher::<u8_>("reward", &QoSProfile::default())?;
    let reward: u8 = 10; 

    let _subscription = node.create_subscription(
        "blog",
        move |msg: Arc<String_>| {
            rclrust_info!(logger, "I read: {}", msg.data);
            publisher.publish(&u8_ {
                data: reward,
            }).unwrap();
            rclrust_info!(logger, "I paid: {} for the blog", reward);
        },
        &QoSProfile::default(),
    )?;

    node.wait();
    Ok(())
}
```

The blog publisher node needs to be modified to receive the reward as follows 

```rust
use std::{thread::sleep, time::Duration};
use std::sync::Arc;

use anyhow::Result;
use rclrust::{qos::QoSProfile, rclrust_info};
use rclrust_msg::std_msgs::msg::String as String_;
use rclrust_msg::std_msgs::msg::UInt8 as u8_;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = rclrust::init()?;
    let mut node = ctx.create_node("watson_blog")?;
    let logger = node.logger();
    let publisher = node.create_publisher::<String_>("blog", &QoSProfile::default())?;

    let _subscription = node.create_subscription(
        "reward",
        move |msg: Arc<u8_>| {
            rclrust_info!(logger, "I received ${} reward", msg.data);
        },
        &QoSProfile::default(),
    )?;

    let logger = node.logger();
    let mut count = 1; 
    loop {
        publisher.publish(&String_ {
            data: format!("Watson's {}th blog", count),
        })?;
        rclrust_info!(logger, "Watson's {}th blog published", count);
        count += 1; 
        sleep(Duration::from_millis(100));
    }

    Ok(())
}
```

And the output includes the reward message 

```
[INFO] [1653829848.327928005] [watson_blog]: Watson's 79th blog published
[INFO] [1653829848.329881922] [watson_blog]: I received $10 reward
```

## Summary 

That's it for a quick introduction, which covers how to write publishers and subscribers. One thing to note is that unlike the Python or C++ examples for ROS/ROS2, there are no callback functions in the Rust ones, since the function is async, as mentioned [here](https://github.com/adnanademovic/rosrust/issues/121). 

## Reference 

- [ROS 2 workshop](https://ros2-industrial-workshop.readthedocs.io/en/latest/)
- [d2l-ros2](https://github.com/fishros/d2l-ros2) a ROS2 course in Chinese 
