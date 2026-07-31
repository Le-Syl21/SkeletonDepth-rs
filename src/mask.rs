//! A binary silhouette mask (`FG` = foreground / person, `BG` = background),
//! stored row-major. This replaces the `cv::Mat` byte buffer of the original —
//! the algorithm never used an OpenCV *operation*, only a 2D byte container plus
//! a handful of hand-rolled passes (largest-region flood fill, centroid, a
//! line-vs-silhouette test), all of which live here in safe Rust.

use crate::geom::{line, Pt};

/// Foreground pixel value (a person pixel).
pub const FG: u8 = 255;
/// Background pixel value.
pub const BG: u8 = 0;

/// A binary image at the (sub-sampled) working resolution.
#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Mask {
    /// A fresh all-background mask of the given size.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![BG; w * h],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    /// Foreground test, background outside the image bounds.
    #[inline]
    pub fn is_fg(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return false;
        }
        self.data[self.idx(x as usize, y as usize)] == FG
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        let i = self.idx(x, y);
        self.data[i] = v;
    }

    /// Number of foreground pixels.
    pub fn count_fg(&self) -> usize {
        self.data.iter().filter(|&&v| v == FG).count()
    }

    /// Mean position of the foreground pixels — the body's centre column/row
    /// (`mediaPoint` in the original). `None` when the mask is empty.
    pub fn centroid(&self) -> Option<Pt> {
        let (mut sx, mut sy, mut n) = (0i64, 0i64, 0i64);
        for y in 0..self.h {
            let row = y * self.w;
            for x in 0..self.w {
                if self.data[row + x] == FG {
                    sx += x as i64;
                    sy += y as i64;
                    n += 1;
                }
            }
        }
        if n == 0 {
            return None;
        }
        Some(Pt::new((sx / n) as i32, (sy / n) as i32))
    }

    /// Keep only the largest 8-connected foreground component, clearing the rest
    /// (`detectBiggerRegion`). This drops bystanders and speckle so the extremity
    /// scan sees a single body.
    pub fn keep_largest_region(&mut self) {
        let (label, sizes) = label_components(&self.data, self.w, self.h);
        // Winner = component with the most pixels (id 0 is background).
        let best = (1..sizes.len())
            .max_by_key(|&i| sizes[i])
            .map(|i| i as u32)
            .unwrap_or(0);
        if best == 0 {
            return;
        }
        for (px, &lab) in self.data.iter_mut().zip(label.iter()) {
            if lab != best {
                *px = BG;
            }
        }
    }

    /// Clear every 8-connected component smaller than `min_px`
    /// (`removeSmallsRegions`). Applied to the thinned skeleton to prune stray
    /// short spurs before the extremity scan.
    pub fn remove_small_regions(&mut self, min_px: u32) {
        let (label, sizes) = label_components(&self.data, self.w, self.h);
        for (px, &lab) in self.data.iter_mut().zip(label.iter()) {
            if lab != 0 && sizes[lab as usize] < min_px {
                *px = BG;
            }
        }
    }

    /// Zhang-Suen skeletonization (`DrawAux::thinning`): erode the silhouette to
    /// a 1-pixel-wide medial skeleton whose endpoints are the limb tips (head,
    /// hands). Returns a new mask; `self` is untouched. Border pixels are never
    /// removed, matching the reference.
    pub fn thinned(&self) -> Mask {
        let (w, h) = (self.w, self.h);
        // Work on a 0/1 buffer.
        let mut img: Vec<u8> = self.data.iter().map(|&v| u8::from(v == FG)).collect();
        if w >= 3 && h >= 3 {
            let mut marker = vec![0u8; w * h];
            loop {
                let mut changed = false;
                for iter in 0..2 {
                    marker.iter_mut().for_each(|m| *m = 0);
                    for y in 1..h - 1 {
                        for x in 1..w - 1 {
                            let at = |dx: isize, dy: isize| -> u8 {
                                img[((y as isize + dy) as usize) * w + (x as isize + dx) as usize]
                            };
                            let no = at(0, -1);
                            let ne = at(1, -1);
                            let ea = at(1, 0);
                            let se = at(1, 1);
                            let so = at(0, 1);
                            let sw = at(-1, 1);
                            let we = at(-1, 0);
                            let nw = at(-1, -1);
                            // A = number of 0→1 transitions in the ordered ring.
                            let a = u32::from(no == 0 && ne == 1)
                                + u32::from(ne == 0 && ea == 1)
                                + u32::from(ea == 0 && se == 1)
                                + u32::from(se == 0 && so == 1)
                                + u32::from(so == 0 && sw == 1)
                                + u32::from(sw == 0 && we == 1)
                                + u32::from(we == 0 && nw == 1)
                                + u32::from(nw == 0 && no == 1);
                            // B = number of foreground neighbours.
                            let b = u32::from(no)
                                + u32::from(ne)
                                + u32::from(ea)
                                + u32::from(se)
                                + u32::from(so)
                                + u32::from(sw)
                                + u32::from(we)
                                + u32::from(nw);
                            let (m1, m2) = if iter == 0 {
                                (no * ea * so, ea * so * we)
                            } else {
                                (no * ea * we, no * so * we)
                            };
                            if a == 1 && (2..=6).contains(&b) && m1 == 0 && m2 == 0 {
                                marker[y * w + x] = 1;
                            }
                        }
                    }
                    for (v, &m) in img.iter_mut().zip(marker.iter()) {
                        if m == 1 && *v == 1 {
                            *v = 0;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        Mask {
            w,
            h,
            data: img.iter().map(|&v| if v == 1 { FG } else { BG }).collect(),
        }
    }

    /// Count how many pixels of the straight segment `a`→`b` fall *outside* the
    /// silhouette (`qPointsLineOutside`). Coordinates are in this mask's
    /// (sub-sampled) space. Used to test whether a candidate limb segment stays
    /// on the body.
    pub fn count_line_outside(&self, a: Pt, b: Pt) -> usize {
        line(a, b)
            .into_iter()
            .filter(|p| !self.is_fg(p.x, p.y))
            .count()
    }
}

/// Label 8-connected foreground components. Returns `(label per pixel, sizes)`
/// where label `0` is background and `sizes[k]` is the pixel count of component
/// `k` (`sizes[0]` unused). Iterative flood fill — no recursion, so a full-frame
/// blob can't overflow the stack.
fn label_components(data: &[u8], w: usize, h: usize) -> (Vec<u32>, Vec<u32>) {
    let n = w * h;
    let mut label = vec![0u32; n];
    let mut sizes: Vec<u32> = vec![0];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if data[start] != FG || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0u32;
        stack.push(start);
        label[start] = id;
        while let Some(p) = stack.pop() {
            size += 1;
            let (x, y) = (p % w, p / w);
            let (x0, x1, y0, y1) = (x > 0, x + 1 < w, y > 0, y + 1 < h);
            let mut visit = |q: usize| {
                if data[q] == FG && label[q] == 0 {
                    label[q] = id;
                    stack.push(q);
                }
            };
            if x0 {
                visit(p - 1);
            }
            if x1 {
                visit(p + 1);
            }
            if y0 {
                visit(p - w);
                if x0 {
                    visit(p - w - 1);
                }
                if x1 {
                    visit(p - w + 1);
                }
            }
            if y1 {
                visit(p + w);
                if x0 {
                    visit(p + w - 1);
                }
                if x1 {
                    visit(p + w + 1);
                }
            }
        }
        sizes.push(size);
    }
    (label, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled_rect(m: &mut Mask, x0: usize, y0: usize, x1: usize, y1: usize) {
        for y in y0..y1 {
            for x in x0..x1 {
                m.set(x, y, FG);
            }
        }
    }

    #[test]
    fn largest_region_wins() {
        let mut m = Mask::new(40, 40);
        filled_rect(&mut m, 2, 2, 6, 6); // 16 px
        filled_rect(&mut m, 20, 20, 34, 34); // 196 px
        m.keep_largest_region();
        assert!(!m.is_fg(3, 3), "small blob should be cleared");
        assert!(m.is_fg(25, 25), "big blob should survive");
        assert_eq!(m.count_fg(), 196);
    }

    #[test]
    fn centroid_center() {
        let mut m = Mask::new(10, 10);
        filled_rect(&mut m, 4, 4, 6, 6);
        assert_eq!(m.centroid(), Some(Pt::new(4, 4)));
    }

    #[test]
    fn thinning_reduces_and_stays_inside() {
        let mut m = Mask::new(15, 15);
        filled_rect(&mut m, 3, 3, 12, 12); // 9×9 solid = 81 px
        let before = m.count_fg();
        let t = m.thinned();
        let after = t.count_fg();
        assert!(
            after > 0 && after < before / 2,
            "skeleton thinner: {after}/{before}"
        );
        // Skeleton must stay within the original shape.
        for y in 0..15 {
            for x in 0..15 {
                if t.is_fg(x, y) {
                    assert!(m.is_fg(x, y), "skeleton pixel ({x},{y}) outside the shape");
                }
            }
        }
    }

    #[test]
    fn remove_small_regions_prunes_speckle() {
        let mut m = Mask::new(20, 20);
        m.set(1, 1, FG); // isolated 1-px speckle
        filled_rect(&mut m, 8, 8, 13, 13); // 25-px block
        m.remove_small_regions(6);
        assert!(!m.is_fg(1, 1), "speckle pruned");
        assert!(m.is_fg(10, 10), "block kept");
        assert_eq!(m.count_fg(), 25);
    }
}
