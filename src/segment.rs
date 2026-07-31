//! Depth → binary silhouette. This is the only stage that touches the depth
//! sensor; everything downstream works on the resulting [`Mask`]. A port of
//! `SkeletonDepth` from the original: find the nearest valid depth pixel, then
//! keep the near slab `[closest, closest + slab_mm]` as foreground — i.e. the
//! person standing closest to the camera, cut away from the background wall.
//!
//! Because the silhouette method is depth-agnostic, a webcam path with a smooth
//! background can produce the same [`Mask`] by other means and feed the rest of
//! the pipeline unchanged (see [`crate::Tracker::track_mask`]).

use crate::geom::Pt;
use crate::mask::{Mask, FG};

/// Result of segmenting one depth frame.
pub struct Segmented {
    /// Silhouette at the sub-sampled working resolution.
    pub mask: Mask,
    /// Nearest foreground pixel, in full-resolution coordinates.
    pub closest: Pt,
    /// Depth of `closest`, millimeters.
    pub closest_z: u16,
}

/// Turn a full-resolution depth frame (millimeters, `0` = no data) into a
/// sub-sampled silhouette. `subsample` matches the value the [`crate::Tracker`]
/// was built with. Returns `None` if the frame holds no valid depth.
///
/// The slab is *adaptive*: a pixel is foreground when its depth lies in
/// `[closest, closest + slab_mm + (|dx| + |dy|) / spread_divisor]`, where
/// `dx, dy` are its image-space offset from the nearest pixel. The slab thus
/// thickens away from the closest point, letting the body spread in depth as it
/// widens (arms, shoulders). `spread_divisor == 0` disables it (fixed slab).
pub fn segment(
    depth_mm: &[u16],
    w: usize,
    h: usize,
    subsample: usize,
    slab_mm: u16,
    spread_divisor: u32,
) -> Option<Segmented> {
    debug_assert_eq!(depth_mm.len(), w * h);
    let sub = subsample.max(1);

    // Pass 1 — nearest valid depth pixel (the seed of the near slab).
    let mut best_z = u16::MAX;
    let mut best_i = usize::MAX;
    for (i, &z) in depth_mm.iter().enumerate() {
        if z != 0 && z < best_z {
            best_z = z;
            best_i = i;
        }
    }
    if best_i == usize::MAX {
        return None;
    }
    let closest = Pt::new((best_i % w) as i32, (best_i / w) as i32);
    let near = u32::from(best_z);
    let base_far = near + u32::from(slab_mm);

    // Pass 2 — sub-sampled silhouette: a working pixel is foreground when the
    // depth sampled at its top-left source pixel lies in the (adaptive) slab.
    let mw = w / sub;
    let mh = h / sub;
    let mut mask = Mask::new(mw, mh);
    for my in 0..mh {
        let sy = my * sub;
        for mx in 0..mw {
            let sx = mx * sub;
            let z = u32::from(depth_mm[sy * w + sx]);
            if z < near {
                continue;
            }
            // Adaptive thickening; `checked_div` folds in the `divisor == 0`
            // (fixed slab) case as a zero spread.
            let dx = (closest.x - sx as i32).unsigned_abs();
            let dy = (closest.y - sy as i32).unsigned_abs();
            let far = base_far + (dx + dy).checked_div(spread_divisor).unwrap_or(0);
            if z <= far {
                mask.set(mx, my, FG);
            }
        }
    }

    Some(Segmented {
        mask,
        closest,
        closest_z: best_z,
    })
}

/// A full-resolution depth frame kept around so joint pixels can read their
/// `z` back (`getMeanDepthValue`): the mean of a 5×5 window, ignoring holes.
pub struct DepthMap<'a> {
    pub depth: &'a [u16],
    pub w: usize,
    pub h: usize,
}

impl DepthMap<'_> {
    /// Mean depth (mm) over a 5×5 window centred on the full-resolution pixel
    /// `(x, y)`, ignoring zero (no-data) samples. `0` if the window is empty or
    /// too close to the border.
    pub fn mean_z(&self, x: i32, y: i32) -> u16 {
        if x < 2 || y < 2 || x as usize + 2 >= self.w || y as usize + 2 >= self.h {
            return 0;
        }
        let (cx, cy) = (x as usize, y as usize);
        let (mut sum, mut n) = (0u32, 0u32);
        for yy in (cy - 2)..=(cy + 2) {
            let row = yy * self.w;
            for xx in (cx - 2)..=(cx + 2) {
                let v = self.depth[row + xx];
                if v != 0 {
                    sum += v as u32;
                    n += 1;
                }
            }
        }
        sum.checked_div(n).map_or(0, |mean| mean as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_keeps_near_object_only() {
        // 8×8 frame: a near 3×3 block at 1000 mm, background at 3000 mm.
        let (w, h) = (8, 8);
        let mut depth = vec![3000u16; w * h];
        for y in 1..4 {
            for x in 1..4 {
                depth[y * w + x] = 1000;
            }
        }
        let seg = segment(&depth, w, h, 1, 600, 5).unwrap();
        assert_eq!(seg.closest_z, 1000);
        assert!(seg.mask.is_fg(2, 2), "near block is foreground");
        assert!(!seg.mask.is_fg(6, 6), "far background is cut away");
    }

    #[test]
    fn adaptive_slab_widens_with_distance() {
        // Near seed at 1000 mm; a pixel 50 px away sitting at 1620 mm — just
        // beyond the fixed 600 slab (far 1600). At divisor 2 the spread adds
        // 50/2 = 25 mm (far 1625), pulling it in.
        let (w, h) = (60, 4);
        let mut depth = vec![0u16; w * h];
        depth[0] = 1000; // closest at (0,0)
        depth[50] = 1620; // 50 px away, 620 mm behind
        assert!(
            !segment(&depth, w, h, 1, 600, 0).unwrap().mask.is_fg(50, 0),
            "fixed slab cuts it"
        );
        assert!(
            segment(&depth, w, h, 1, 600, 2).unwrap().mask.is_fg(50, 0),
            "adaptive slab keeps it"
        );
    }

    #[test]
    fn empty_depth_is_none() {
        assert!(segment(&[0u16; 16], 4, 4, 1, 600, 5).is_none());
    }
}
