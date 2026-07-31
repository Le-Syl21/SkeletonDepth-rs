# skeleton-depth

Pure-Rust **upper-body skeleton tracking from a depth silhouette** — no OpenCV,
no OpenNI, no machine-learning model, no external dependencies.

Feed a depth frame (millimeters) and get back the **head, shoulders and hands**
of the person standing closest to the camera. It's cheap enough to run every
frame on one core, and the head is anchored to the body — not just "the nearest
blob" — so a nearer object off to the side doesn't get mistaken for the head.

```rust
use skeleton_depth::{Config, Tracker};

let (w, h) = (512, 424);                 // Kinect v2 depth resolution
let mut tracker = Tracker::new(w, h, Config::default());

// each frame: depth in u16 millimeters, 0 = no data
let skel = tracker.track(&depth);
if let Some(head) = skel.head {
    println!("head @ ({}, {}) — {} mm", head.x, head.y, head.z_mm);
}
if let (Some(l), Some(r)) = (skel.left_hand, skel.right_hand) {
    // e.g. is the player the one whose hands are on the lockbar?
}
```

## How it works

1. **Segment** (`segment`): find the nearest valid depth pixel and keep the near
   slab `[closest, closest + slab_mm]` as foreground → a binary silhouette. This
   is the only stage that touches depth.
2. **Isolate** (`Mask::keep_largest_region`): keep the largest connected
   component, dropping bystanders and speckle.
3. **Extremities** (`detect::locate_extremities`): relative to the body's centre
   column, the **head** is the highest silhouette pixel, the **hands** are the
   outermost pixels in the left/right quadrants; **shoulders** are found by
   dropping from the head column to the silhouette edge.
4. **Smooth** (`smooth::MedianRing`): a short per-joint temporal median rejects
   single-frame outliers and rides through brief dropouts.

Joints come out as full-resolution pixels plus depth (`Joint { x, y, z_mm }`);
`Joint::to_metric` deprojects to camera-space millimeters through pinhole
intrinsics.

### Depth-less / webcam path

Only step 1 needs depth. A webcam with a smooth background can produce the same
silhouette by background subtraction and feed it straight to
`Tracker::track_mask` — the identical joint rules then find head + hands in 2D
(pass `depth_mm: None` for `z_mm = 0`, or a registered depth frame to fill `z`).

## Status / roadmap

Working: **head, shoulders, hands, body centre**, with temporal smoothing.

Not yet ported:
- **Elbows** — the original tracks the arm curve by angle; deferred.
- **Adaptive slab** — the segmentation slab is currently fixed; the original
  widens it away from the closest point.
- **Geodesic head selection** — an optional, more robust head pick (connectivity
  to the body mass, à la Skeltrack) is being considered as an alternative to the
  centre-column heuristic.
- Constants (`afa`, shoulder offsets, slab) mirror the original's frontal-camera
  tuning and will want retuning for very different placements (e.g. a camera
  looking *up* at the player).

## Credit & license

MIT. This is a **from-scratch Rust reimplementation** of the upper-body
skeleton-detection method in
[`derzu/BodySkeletonTracker`](https://github.com/derzu/BodySkeletonTracker)
(MIT, © 2018 Derzu) — no original source was copied. The underlying idea traces
to Andreas Baak's *"A Data-Driven Approach for Real-Time Full Body Pose
Reconstruction from a Depth Camera"*.
