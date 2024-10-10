---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "CROWS"
date:   2024-09-16 09:00:00 -0400
description: >- # Supports markdown
  Conformalized Reachable Sets for Obstacle Avoidance With Spheres
show-description: true

# Add page-specifi mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: true

# Preview image for social media cards
image:
  path: 
  height: 800
  width: 600
  alt: CROWS Main Figure - Bookshelf Scenario

# Only the first author is supported by twitter metadata
authors:
  - name: Yongseok Kwon
    email: kwonys [at] umich.edu
  - name: Jonathan Michaux
    email: jmichaux [at] umich.edu
  - name: Seth Isaacson
    email: sethgi [at] umich.edu
  - name: Bohao Zhang
    email: jimzhang [at] umich.edu
  - name: Matthew Ejakov
    email: ejakovm [at] umich.edu
  - name: Ram Vasudevan
    email: ramv [at] umich.edu


# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes:
  <br> All authors affiliated with the department of Mechanical Engineering and Department of Robotics of the University of Michigan, Ann Arbor.
links:
  - icon: arxiv
    icon-library: simpleicons
    text: Arxiv (Coming Soon!)
#    url: https://arxiv.org/
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/crows
  - icon: bi-file-earmark-text-fill
    icon-library: bootstrap-icons
    text: Appendix
    url: assets/documents/CROWS_Appendix.pdf
  - icon: googledrive
    icon-library: simpleicons
    text: Dataset
    url: https://drive.google.com/drive/folders/1y82zpWuKaZmejr7AXxctpPKblB2tOSW9?usp=sharing

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---
# [Overview Videos](#overview-videos)

<!-- BEGIN OVERVIEW VIDEOS -->
<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      class="autoplay-on-load"
      preload="none"
      controls
      disablepictureinpicture
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto;"
      poster="assets/images/bookshelf_thumb.jpg">
      <source src="assets/videos/bookshelf_single_take.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>CROWS performing trajectory planning between two bookshelfs</p>
  </div>
</div> <!-- END OVERVIEW VIDEOS -->


<!-- BEGIN ABSTRACT -->
<div markdown="1" class="content-block justify grey">

# [Abstract](#abstract)
Safe motion planning algorithms are necessary for deploying autonomous robots in unstructured environments. Motion plans must be safe to ensure that the robot does not harm humans or damage any nearby objects. Generating these motion plans in real-time is also important to ensure that the robot can adapt to sudden changes in its environment. Many trajectory optimization methods introduce heuristics that balance safety and real-time performance, potentially increasing the risk of the robot colliding with its environment. This paper addresses this challenge by proposing Conformalized Reachable Sets for Obstacle Avoidance With Spheres (CROWS). CROWS is a novel real-time, receding-horizon trajectory planner that generates probalistically-safe motion plans. Offline, CROWS learns a novel neural network-based representation of a spherebased reachable set that overapproximates the swept volume of the robot’s motion. CROWS then uses conformal prediction to compute a confidence bound that provides a *probabilistic safety guarantee on the learned reachable set*. At runtime, CROWS performs trajectory optimization to select a trajectory that is probabilstically-guaranteed to be collision-free. We demonstrate that CROWS outperforms a variety of state-of-the-art methods in solving challenging motion planning tasks in cluttered environments while remaining collision-free. Code, data, and video demonstrations can be found at [roahmlab/crows](https://github.com/roahmlab/crows).

</div> <!-- END ABSTRACT -->

<!-- BEGIN METHOD -->
<div markdown="1" class="justify">

# [Method](#method)

![link_construction](./assets/images/method_overview.png)
{: class="fullwidth"}


<!-- # Contributions -->
This paper proposes Conformalized Reachable Sets for Obstacle Avoidance With Spheres (CROWS), a neural network-based safety representation that can be efficiently integrated into a trajectory optimization algorithm. CROWS extends SPARROWS<sup>[1](https://roahmlab.github.io/sparrows/)</sup>  by learning an overapproximation of the swept volume (i.e. reachable set) of a serial robot manipulator that is composed entirely of spheres. Prior to planning, a neural network is trained to approximate the sphere-based reachable set. Then, CROWS applies conformal prediction to compute a confidence bound that provides a probabilistic safety guarantee. Finally, CROWS uses the conformalized reachable set and its learned gradient to solve an optimization problem to generate probabilistically-safe trajectories online.

![link_construction](./assets/images/conformal_prediction.png)
{: style="width: 70%; display: block; margin: 0 auto;" }



Panel (A) compares the ground truth (left, purple), predicted (middle, orange), and conformalized (right, pink) reachable sets. To construct the conformalized reachable set, CROWS first defines the nonconformity score, which is the minimum buffer (blue) to ensure that the predicted sphere (orange) encloses the ground truth sphere (purple) (Panel C). The distribution of the nonconformity scores for joint 4 over a time interval $$T_i$$ with the values defining the quantiles indicated in blue (Panel B). Next, conformal prediction computes a confidence bound that upper bounds the nonconformity scores with a probability of $$1 − \epsilon$$. The predicted joint sphere (A, middle) is then expanded by the size of this confidence bound to give the conformalized joint sphere (A, right). Finally, applying this procedure for all of the joint spheres gives the conformalized neural (A, right) reachable set that is guaranteed to cover the ground truth reachable set with probability greater than $$(1 − \epsilon)^{n_q+1}$$, where $$n_q$$ is the number of actuated joints (Thm. 5).



</div><!-- END METHOD -->

<!-- START RESULTS -->
<div markdown="1" class="content-block grey justify">

# [Results](#results)

## [Simulation Results](#simulation-results)
<!-- START SIMULATION VIDEOS -->
<div class="video-container">
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/realistic_1_thumb.jpg">
      <source src="assets/videos/realistic_1.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/realistic_2_thumb.jpg">
      <source src="assets/videos/realistic_2.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/realistic_3_thumb.jpg">
      <source src="assets/videos/realistic_3.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/realistic_4_thumb.jpg">
      <source src="assets/videos/realistic_4.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/realistic_5_thumb.jpg">
      <source src="assets/videos/realistic_5.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div><!-- END SIMULATION VIDEOS -->

## [Hardware Results](#hardware-results)
<!-- START HARDWARE VIDEOS -->
<div class="video-container">
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/hardware_demo_1_thumb.jpg">
      <source src="assets/videos/hardware_demo_1.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/hardware_demo_2_thumb.jpg">
      <source src="assets/videos/hardware_demo_2.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      poster="assets/images/hardware_demo_3_thumb.jpg">
      <source src="assets/videos/hardware_demo_3.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div><!-- END HARDWARE VIDEOS -->



</div><!-- END RESULTS -->

<div markdown="1" class="justify">
  
# [Related Projects](#related-projects)


* [Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres](https://roahmlab.github.io/sparrows/)
* [Let’s Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat](https://roahmlab.github.io/splanning/)
* [Reachability-based Trajectory Design with Neural Implicit Safety Constraints](https://roahmlab.github.io/RDF/)

<div markdown="1" class="content-block grey justify">
  
# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at the University of Michigan - Ann Arbor.

```bibtex
@article{kwon2024crows,
  title={Conformalized Reachable Sets for Obstacle Avoidance With Spheres},
  author={Yongseok Kwon and Jonathan Michaux and and Seth Isaacson and Bohao Zhang and Matthew Ejakov and Ram Vasudevan},
  journal={},
  year={2024},
  volume={},
  url={}}
```
</div>


<!-- below are some special scripts -->
<script>
window.addEventListener("load", function() {
  // Get all video elements and auto pause/play them depending on how in frame or not they are
  let videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.25 });

    observer.observe(video);
  });

  // document.addEventListener("DOMContentLoaded", function() {
  videos = document.querySelectorAll('.autoplay-on-load');

  videos.forEach(video => {
    video.play();
  });
});
</script>
