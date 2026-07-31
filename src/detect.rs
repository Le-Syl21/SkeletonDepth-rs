//! Extremity scan and upper-body joint rules — the heart of the method
//! (`locateMaximus` / `setMaximus` / `locateMainBodyPoints` / `locateShoulders`
//! in the original). Everything runs on the sub-sampled silhouette; joint
//! coordinates are scaled back to full resolution by the caller.
//!
//! The idea is deliberately simple and cheap: relative to the body's centre
//! column, the head is the highest silhouette pixel, and the hands are the
//! outermost pixels in the left/right quadrants. No model, no training data.

use crate::geom::{dist, Pt};
use crate::mask::Mask;

/// `0` in a coordinate means "not found", matching the original's zero-init
/// convention (the body never sits exactly on column/row 0 in practice).
const ABSENT: Pt = Pt { x: 0, y: 0 };

/// Silhouette extremities relative to the centre column, in sub-sampled pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct Extremities {
    pub right: Pt,
    pub left: Pt,
    pub top_center: Pt,
    pub bottom_center: Pt,
    pub top_right: Pt,
    pub bottom_right: Pt,
    pub top_left: Pt,
    pub bottom_left: Pt,
}

/// Scan the mask once and collect the extremities around `center_ws` (the body
/// centre column, in sub-sampled pixels). `afa` is the half-width of the central
/// band that defines "the head column" (`70 / subsample` in the original).
pub fn locate_extremities(mask: &Mask, center_ws: i32, afa: i32) -> Extremities {
    let mut e = Extremities {
        right: ABSENT,
        left: Pt::new(mask.w as i32, 0),
        top_center: Pt::new(0, mask.h as i32),
        bottom_center: ABSENT,
        top_right: Pt::new(0, mask.h as i32),
        bottom_right: ABSENT,
        top_left: Pt::new(0, mask.h as i32),
        bottom_left: ABSENT,
    };
    let band_r12 = center_ws + (afa as f32 * 1.2) as i32;
    let band_r13 = center_ws + (afa as f32 * 1.3) as i32;
    let band_l12 = center_ws - (afa as f32 * 1.2) as i32;
    let band_l13 = center_ws - (afa as f32 * 1.3) as i32;

    for y in 0..mask.h as i32 {
        for x in 0..mask.w as i32 {
            if !mask.is_fg(x, y) {
                continue;
            }
            if x >= e.right.x {
                e.right = Pt::new(x, y);
            }
            if x < e.left.x {
                e.left = Pt::new(x, y);
            }
            if x >= center_ws - afa && x <= center_ws + afa {
                if y < e.top_center.y {
                    e.top_center = Pt::new(x, y);
                }
                if y > e.bottom_center.y {
                    e.bottom_center = Pt::new(x, y);
                }
            }
            if x > band_r12 && y <= e.top_right.y {
                e.top_right = Pt::new(x, y);
            }
            if x > band_r13 && y >= e.bottom_right.y {
                e.bottom_right = Pt::new(x, y);
            }
            if x < band_l12 && y < e.top_left.y {
                e.top_left = Pt::new(x, y);
            }
            if x < band_l13 && y > e.bottom_left.y {
                e.bottom_left = Pt::new(x, y);
            }
        }
    }

    // Normalise "sentinel" values that never matched to ABSENT.
    if e.right.x == 0 && e.right.y == 0 {
        e.right = ABSENT;
    }
    if e.left.x == mask.w as i32 {
        e.left = ABSENT;
    }
    if e.top_center.y == mask.h as i32 {
        e.top_center = ABSENT;
    }
    if e.top_right.y == mask.h as i32 {
        e.top_right = ABSENT;
    }
    if e.top_left.y == mask.h as i32 {
        e.top_left = ABSENT;
    }
    e
}

/// Scale a sub-sampled extremity to full-resolution pixels, preserving the
/// "absent" (0,0) marker.
pub fn scale(p: Pt, sub: i32) -> Pt {
    if p == ABSENT {
        ABSENT
    } else {
        Pt::new(p.x * sub, p.y * sub)
    }
}

/// Locate the two shoulders by dropping from the head column and stepping
/// outward until the silhouette edge (`locateShoulders`). Scans the sub-sampled
/// mask; returns full-resolution shoulder points (`ABSENT` if not found).
///
/// * `center_ws` — body centre column (sub-sampled).
/// * `afa` — central-band half-width (sub-sampled); shoulders sit at `±(afa-2)`.
/// * `head_y_sub` — head row (sub-sampled) to start the scan just below.
/// * `drop`, `inset` — start offset below the head and the "go inside the arm"
///   vertical nudge, both in sub-sampled rows.
pub fn locate_shoulders(
    mask: &Mask,
    center_ws: i32,
    afa: i32,
    head_y_sub: i32,
    drop: i32,
    inset: i32,
    sub: i32,
) -> (Pt, Pt) {
    let aff = afa - 2;
    let y_end = (2 * mask.h as i32) / 3;
    let (mut right, mut left) = (ABSENT, ABSENT);
    let mut found_r = false;
    let mut found_l = false;
    let mut y = head_y_sub + drop;
    while y < y_end {
        if !found_r && center_ws + aff < mask.w as i32 && mask.is_fg(center_ws + aff, y) {
            found_r = true;
            right = Pt::new((center_ws + aff) * sub, (y + inset) * sub);
            if found_l {
                break;
            }
        }
        if !found_l && center_ws - aff > 0 && mask.is_fg(center_ws - aff, y) {
            found_l = true;
            left = Pt::new((center_ws - aff) * sub, (y + inset) * sub);
            if found_r {
                break;
            }
        }
        y += 1;
    }
    (right, left)
}

#[inline]
fn present(p: Pt) -> bool {
    p.x != 0
}

/// Pick the right hand from the right-side extremities (full-resolution),
/// mirroring the original's three-case rule:
/// 1. the rightmost point is far past the centre (can't be an elbow) — or its
///    top/bottom candidates agree in `y` → it's the hand;
/// 2. otherwise the top-right point, if it's outside the bottom-right one or
///    clearly above the shoulder → hand;
/// 3. otherwise fall back to the bottom-right point.
pub fn pick_right_hand(
    max_right: Pt,
    max_top_right: Pt,
    max_bottom_right: Pt,
    center: Pt,
    right_shoulder: Pt,
    afa28: f32,
    shift: i32,
) -> Pt {
    if present(max_right)
        && ((max_right.x - center.x) as f32 > afa28
            || ((max_right.y - max_bottom_right.y).abs() < 30
                && (max_right.y - max_top_right.y).abs() < 30))
    {
        max_right
    } else if present(max_top_right)
        && (max_top_right.x > max_bottom_right.x
            || (max_bottom_right.y < center.y + shift
                && max_top_right.y < right_shoulder.y + shift
                && dist(max_top_right, right_shoulder) > 50.0))
    {
        max_top_right
    } else if present(max_bottom_right) {
        max_bottom_right
    } else {
        ABSENT
    }
}

/// Left-hand counterpart of [`pick_right_hand`] (mirrored in x).
pub fn pick_left_hand(
    max_left: Pt,
    max_top_left: Pt,
    max_bottom_left: Pt,
    center: Pt,
    left_shoulder: Pt,
    afa28: f32,
    shift: i32,
) -> Pt {
    if present(max_left)
        && ((center.x - max_left.x) as f32 > afa28
            || ((max_left.y - max_bottom_left.y).abs() < 30
                && (max_left.y - max_top_left.y).abs() < 30))
    {
        max_left
    } else if present(max_top_left)
        && (max_top_left.x < max_bottom_left.x
            || (max_bottom_left.y < center.y + shift
                && max_top_left.y < left_shoulder.y + shift
                && dist(max_top_left, left_shoulder) > 50.0))
    {
        max_top_left
    } else if present(max_bottom_left) {
        max_bottom_left
    } else {
        ABSENT
    }
}

/// Whether a (possibly `ABSENT`) point was actually found.
pub fn is_present(p: Pt) -> bool {
    present(p)
}
