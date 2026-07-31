# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

### Added
- **Elbows** — completes the upper-body joint set (`left_elbow` / `right_elbow`):
  straight-arm midpoint, or the arm-skeleton bend point when the arm is folded.
- Zhang-Suen thinning (`Mask::thinned`) and small-region pruning
  (`Mask::remove_small_regions`); the extremity scan and arm tracing now run on
  the thinned skeleton, matching the original pipeline.
- `Config::min_region_px` to tune skeleton spur pruning.

### Changed
- Connected-component labelling is now 8-connected (required for the diagonal
  skeleton lines).

## [0.1.0]

### Added
- Initial upper-body skeleton tracker from a depth silhouette, pure Rust with
  zero dependencies (a from-scratch reimplementation of the method in
  [derzu/BodySkeletonTracker](https://github.com/derzu/BodySkeletonTracker), MIT).
- `Tracker::track` — detect from a depth frame (`&[u16]`, millimeters).
- `Tracker::track_mask` — detect from a pre-built silhouette (webcam / 2D path).
- Joints: head, shoulders, hands, body centre, each with pixel + depth and
  `Joint::to_metric` deprojection through pinhole intrinsics.
- Per-joint temporal median smoothing.
