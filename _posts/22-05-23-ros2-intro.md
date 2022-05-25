---
layout: post
title: "Introduction to ROS2"
author: "Mo Shan"
tags: vscode
---

## Introduction 

In May 2022, ROS 2 Humble Hawksbill (humble) was released, which supports Ubuntu 22.04. Since I am still using Ubuntu 20.04, this blog will focus on `ROS 2 Galactic Geochelone` (galactic) instead, released in May, 2021. 

First thing first, how to install ROS2? Because our team is migrating from ROS1 to ROS2, I need to use both for now. My current way is to install ROS2 on my OS following the [official guide](https://docs.ros.org/en/galactic/Installation.html), and install ROS1 via `mamba` using [RoboStack](https://robostack.github.io/). Even though RoboStack also provides `ros-galactic-desktop`, I do not recommend it since the package support is not complete, and does not work well with Python APIs. 

One may wonder whether there is a way to just taste the flavour of ROS2, see whether one likes it, before diving into the full-scale installation. There are [docker files](https://github.com/athackst/dockerfiles) for those who have nvidia GPU, whereas [this repo](https://github.com/Tiryoh/docker-ros2-desktop-vnc) provides docker files for CPU and VNC. The one used in this blog is the VNC version. After pulling and building the docker file, use a browser to connect to `http://127.0.0.1:6080/`, which shows the desktop. Also need to take care of the the permission issue via `sudo chmod 777 -R ~/.ros/` and you are good to go. 



## Reference 

- [ROS 2 workshop](https://ros2-industrial-workshop.readthedocs.io/en/latest/)
- [d2l-ros2](https://github.com/fishros/d2l-ros2) a ROS2 course in Chinese 
