# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial upper-body skeleton tracker from a depth silhouette, pure Rust with
  zero dependencies (a from-scratch reimplementation of the method in
  [derzu/BodySkeletonTracker](https://github.com/derzu/BodySkeletonTracker), MIT).
- `Tracker::track` — detect from a depth frame (`&[u16]`, millimeters).
- `Tracker::track_mask` — detect from a pre-built silhouette (webcam / 2D path).
- Joints: head, shoulders, hands, body centre, each with pixel + depth and
  `Joint::to_metric` deprojection through pinhole intrinsics.
- Per-joint temporal median smoothing.

### Not yet implemented
- Elbows (arm-curve angle tracking).
- Adaptive depth slab.
- Optional geodesic head selection.
