//! Upper-body skeleton tracking from a depth silhouette — pure Rust, no OpenCV,
//! no OpenNI, no machine-learning model.
//!
//! Feed a depth frame (millimeters) and get back the **head, shoulders and
//! hands** of the person standing closest to the camera. The pipeline is:
//!
//! 1. [`segment::segment`]: keep the near depth slab → a binary silhouette
//!    [`mask::Mask`].
//! 2. [`mask::Mask::keep_largest_region`]: drop bystanders and speckle.
//! 3. [`detect::locate_extremities`]: the head is the highest silhouette pixel
//!    above the body centre; the hands are the outermost pixels in the
//!    left/right quadrants.
//! 4. per-joint temporal median smoothing ([`smooth::MedianRing`]).
//!
//! It is a from-scratch Rust reimplementation of the method in
//! [`derzu/BodySkeletonTracker`](https://github.com/derzu/BodySkeletonTracker)
//! (MIT). No original source was copied.
//!
//! # Example
//! ```no_run
//! use skeleton_depth::{Config, Tracker};
//! // 512×424 Kinect v2 depth frame in millimeters (0 = no data).
//! let (w, h) = (512, 424);
//! let depth = vec![0u16; w * h];
//! let mut tracker = Tracker::new(w, h, Config::default());
//! let skel = tracker.track(&depth);
//! if let Some(head) = skel.head {
//!     println!("head at pixel ({}, {}), {} mm deep", head.x, head.y, head.z_mm);
//! }
//! ```

pub mod detect;
pub mod geom;
pub mod mask;
pub mod segment;
pub mod smooth;

use detect::{
    is_present, locate_extremities, locate_shoulders, pick_elbow, pick_left_hand, pick_right_hand,
    scale, skeleton_arm,
};
use geom::Pt;
use mask::Mask;
use segment::DepthMap;
use smooth::MedianRing;

/// Pinhole depth-camera intrinsics, used only by [`Joint::to_metric`] to turn a
/// pixel + depth into metric camera-space millimeters.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// A detected joint: full-resolution pixel plus its depth in millimeters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Joint {
    pub x: i32,
    pub y: i32,
    pub z_mm: u16,
}

impl Joint {
    /// Deproject to camera-space millimeters `[X, Y, Z]` through pinhole
    /// intrinsics: `X = (x - cx) * z / fx`, `Y = (y - cy) * z / fy`, `Z = z`.
    pub fn to_metric(&self, intr: &Intrinsics) -> [f32; 3] {
        let z = self.z_mm as f32;
        [
            (self.x as f32 - intr.cx) * z / intr.fx,
            (self.y as f32 - intr.cy) * z / intr.fy,
            z,
        ]
    }
}

/// The upper-body skeleton for one frame. Every joint is `Option` — absent when
/// that part wasn't found. Elbows are not computed yet (always `None`); see the
/// crate README roadmap.
#[derive(Clone, Copy, Debug, Default)]
pub struct Skeleton {
    pub head: Option<Joint>,
    pub left_shoulder: Option<Joint>,
    pub right_shoulder: Option<Joint>,
    pub left_elbow: Option<Joint>,
    pub right_elbow: Option<Joint>,
    pub left_hand: Option<Joint>,
    pub right_hand: Option<Joint>,
    /// Silhouette centroid — the body's centre of mass, handy as a torso anchor.
    pub center: Option<Joint>,
}

/// Tuning knobs. [`Config::default`] targets a Kinect-class sensor with a 4×
/// working sub-sample; the offsets mirror the original and will want retuning
/// for a very different camera placement.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Working sub-sample factor: the silhouette is built at `1/subsample`
    /// resolution. Higher = cheaper and smoother, coarser joints.
    pub subsample: usize,
    /// Depth slab thickness behind the nearest pixel, millimeters. The person is
    /// segmented as everything within `[closest, closest + slab_mm]`.
    pub slab_mm: u16,
    /// Central-band half-width (sub-sampled pixels) that defines "the head
    /// column". `0` = auto (`70 / subsample`, as in the original).
    pub afa: i32,
    /// Rows below the head (sub-sampled) at which the shoulder scan starts.
    pub shoulder_drop: i32,
    /// "Go inside the arm" vertical nudge applied to shoulders (sub-sampled rows).
    pub shoulder_inset: i32,
    /// Thinned-skeleton components smaller than this (pixels) are pruned as spurs
    /// before the extremity scan (`removeSmallsRegions`).
    pub min_region_px: u32,
    /// Temporal median window (frames) per joint.
    pub history: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            subsample: 4,
            slab_mm: 600,
            afa: 0,
            shoulder_drop: 30,
            shoulder_inset: 10,
            min_region_px: 6,
            history: 5,
        }
    }
}

/// Horizontal "shift" tolerance used by the hand-selection rules (full-res px),
/// matching the original's `shift = 50`.
const SHIFT: i32 = 50;

/// Stateful tracker. Holds per-joint smoothing history across frames; create one
/// per camera and call [`Tracker::track`] each frame.
pub struct Tracker {
    w: usize,
    h: usize,
    cfg: Config,
    afa: i32,
    afa28: f32,
    rings: [MedianRing; 8],
    skeleton: Skeleton,
}

/// Stable joint ordering for the smoothing ring array.
mod jid {
    pub const HEAD: usize = 0;
    pub const L_SHOULDER: usize = 1;
    pub const R_SHOULDER: usize = 2;
    pub const L_ELBOW: usize = 3;
    pub const R_ELBOW: usize = 4;
    pub const L_HAND: usize = 5;
    pub const R_HAND: usize = 6;
    pub const CENTER: usize = 7;
}

impl Tracker {
    /// Build a tracker for `w × h` depth frames.
    pub fn new(w: usize, h: usize, cfg: Config) -> Self {
        let sub = cfg.subsample.max(1) as i32;
        let afa = if cfg.afa > 0 { cfg.afa } else { 70 / sub };
        let afa28 = afa as f32 * sub as f32 * 2.8;
        let rings = std::array::from_fn(|_| MedianRing::new(cfg.history));
        Self {
            w,
            h,
            cfg,
            afa,
            afa28,
            rings,
            skeleton: Skeleton::default(),
        }
    }

    /// Detect the upper-body skeleton in one depth frame (`w*h` u16 millimeters,
    /// `0` = no data) and return the smoothed result. The reference is valid
    /// until the next call.
    pub fn track(&mut self, depth_mm: &[u16]) -> &Skeleton {
        assert_eq!(depth_mm.len(), self.w * self.h, "depth frame size mismatch");
        let sub = self.cfg.subsample.max(1);
        // Silhouette from the near depth slab, then the shared detection core.
        match segment::segment(depth_mm, self.w, self.h, sub, self.cfg.slab_mm) {
            Some(seg) => self.detect_from_mask(seg.mask, Some(depth_mm)),
            None => self.publish(RawJoints::default()),
        }
    }

    /// Detect from a pre-built silhouette instead of a depth frame — e.g. a
    /// webcam background-subtraction mask, which lets the same joint rules serve
    /// a depth-less camera. `mask` must be at the working (sub-sampled)
    /// resolution the tracker expects; `depth_mm` (full resolution, optional) is
    /// read only for each joint's `z`, so pass `None` for a purely 2D skeleton
    /// (all `z_mm` stay 0).
    pub fn track_mask(&mut self, mask: Mask, depth_mm: Option<&[u16]>) -> &Skeleton {
        self.detect_from_mask(mask, depth_mm)
    }

    /// Shared core: silhouette → joints. Isolates the dominant body, runs the
    /// extremity scan and the head/shoulder/hand rules, then smooths + stores.
    fn detect_from_mask(&mut self, mut mask: Mask, depth_mm: Option<&[u16]>) -> &Skeleton {
        let subi = self.cfg.subsample.max(1) as i32;
        mask.keep_largest_region();

        let Some(center_sub) = mask.centroid() else {
            return self.publish(RawJoints::default());
        };
        let center_ws = center_sub.x; // sub-sampled centre column
        let center_full = Pt::new(center_sub.x * subi, center_sub.y * subi);

        // Thin the silhouette to a 1-px skeleton whose endpoints are the limb
        // tips; the extremity scan and the arm tracing both run on it (the
        // shoulders and elbows still read the solid `mask`).
        let mut skel = mask.thinned();
        skel.remove_small_regions(self.cfg.min_region_px);

        // Extremities → head + hand candidates (scaled to full resolution).
        let e = locate_extremities(&skel, center_ws, self.afa);
        let max_right = scale(e.right, subi);
        let max_left = scale(e.left, subi);
        let max_top_center = scale(e.top_center, subi);
        let max_top_right = scale(e.top_right, subi);
        let max_bottom_right = scale(e.bottom_right, subi);
        let max_top_left = scale(e.top_left, subi);
        let max_bottom_left = scale(e.bottom_left, subi);

        let head = max_top_center;

        // Shoulders (need the head row to start the scan); scanned on the solid mask.
        let (r_shoulder, l_shoulder) = locate_shoulders(
            &mask,
            center_ws,
            self.afa,
            e.top_center.y,
            self.cfg.shoulder_drop,
            self.cfg.shoulder_inset,
            subi,
        );

        // Hands (need the shoulders for the "top vs elbow" disambiguation).
        let r_hand = pick_right_hand(
            max_right,
            max_top_right,
            max_bottom_right,
            center_full,
            r_shoulder,
            self.afa28,
            SHIFT,
        );
        let l_hand = pick_left_hand(
            max_left,
            max_top_left,
            max_bottom_left,
            center_full,
            l_shoulder,
            self.afa28,
            SHIFT,
        );

        // Elbows: straight-arm midpoint, or the arm-skeleton bend point.
        let right_arm = skeleton_arm(&skel, center_ws, self.afa, true);
        let left_arm = skeleton_arm(&skel, center_ws, self.afa, false);
        let r_elbow = pick_elbow(&right_arm, r_hand, r_shoulder, &mask, subi);
        let l_elbow = pick_elbow(&left_arm, l_hand, l_shoulder, &mask, subi);

        self.publish(RawJoints {
            head,
            l_shoulder,
            r_shoulder,
            l_elbow,
            r_elbow,
            l_hand,
            r_hand,
            center: center_full,
            dmap: depth_mm,
        })
    }

    /// Read depth for each present joint, run smoothing, and store the result.
    fn publish(&mut self, raw: RawJoints) -> &Skeleton {
        let depth = raw.dmap;
        let (w, h) = (self.w, self.h);
        let z_of = |p: Pt| -> u16 {
            match depth {
                Some(d) if is_present(p) => DepthMap { depth: d, w, h }.mean_z(p.x, p.y),
                _ => 0,
            }
        };
        let smooth = |ring: &mut MedianRing, p: Pt| -> Option<Joint> {
            let sample = if is_present(p) {
                Some([p.x, p.y, z_of(p) as i32])
            } else {
                None
            };
            ring.update(sample).map(|m| Joint {
                x: m[0],
                y: m[1],
                z_mm: m[2].max(0) as u16,
            })
        };

        self.skeleton = Skeleton {
            head: smooth(&mut self.rings[jid::HEAD], raw.head),
            left_shoulder: smooth(&mut self.rings[jid::L_SHOULDER], raw.l_shoulder),
            right_shoulder: smooth(&mut self.rings[jid::R_SHOULDER], raw.r_shoulder),
            left_elbow: smooth(&mut self.rings[jid::L_ELBOW], raw.l_elbow),
            right_elbow: smooth(&mut self.rings[jid::R_ELBOW], raw.r_elbow),
            left_hand: smooth(&mut self.rings[jid::L_HAND], raw.l_hand),
            right_hand: smooth(&mut self.rings[jid::R_HAND], raw.r_hand),
            center: smooth(&mut self.rings[jid::CENTER], raw.center),
        };
        &self.skeleton
    }

    /// The most recent skeleton, without re-running detection.
    pub fn last(&self) -> &Skeleton {
        &self.skeleton
    }
}

/// Raw (pre-smoothing) joints handed to `publish`. `Pt(0,0)` means absent.
#[derive(Default)]
struct RawJoints<'a> {
    head: Pt,
    l_shoulder: Pt,
    r_shoulder: Pt,
    l_elbow: Pt,
    r_elbow: Pt,
    l_hand: Pt,
    r_hand: Pt,
    center: Pt,
    dmap: Option<&'a [u16]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic depth frame with a head-and-shoulders blob and check
    /// the head lands near the top-centre of that blob.
    #[test]
    fn detects_head_above_body() {
        let (w, h) = (160, 160);
        let mut depth = vec![0u16; w * h];
        // Head: circle-ish block near top centre.
        for y in 20..44 {
            for x in 68..92 {
                depth[y * w + x] = 1500;
            }
        }
        // Torso: wider block below.
        for y in 44..120 {
            for x in 50..110 {
                depth[y * w + x] = 1550;
            }
        }
        let mut t = Tracker::new(w, h, Config::default());
        // A couple of frames to prime the median history.
        for _ in 0..3 {
            t.track(&depth);
        }
        let s = t.track(&depth);
        let head = s.head.expect("head found");
        assert!((head.x - 80).abs() < 20, "head x near centre: {}", head.x);
        assert!(head.y < 60, "head near the top: {}", head.y);
        assert!(s.center.is_some(), "center found");
    }

    #[test]
    fn empty_frame_yields_no_head() {
        let (w, h) = (64, 64);
        let mut t = Tracker::new(w, h, Config::default());
        let s = t.track(&vec![0u16; w * h]);
        assert!(s.head.is_none());
    }
}
