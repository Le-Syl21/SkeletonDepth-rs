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

    /// Keep only the largest 4-connected foreground component, clearing the rest
    /// (`detectBiggerRegion`). This drops bystanders and speckle so the extremity
    /// scan sees a single body. Iterative flood fill — no recursion, so a
    /// full-frame blob can't overflow the stack.
    pub fn keep_largest_region(&mut self) {
        let n = self.w * self.h;
        // Component id per pixel: 0 = background / unvisited-bg, >0 = component.
        let mut label = vec![0u32; n];
        let mut sizes: Vec<u32> = vec![0]; // sizes[0] unused (background)
        let mut stack: Vec<usize> = Vec::new();

        for start in 0..n {
            if self.data[start] != FG || label[start] != 0 {
                continue;
            }
            let id = sizes.len() as u32;
            let mut size = 0u32;
            stack.push(start);
            label[start] = id;
            while let Some(p) = stack.pop() {
                size += 1;
                let (x, y) = (p % self.w, p / self.w);
                // 4-neighbours
                if x > 0 {
                    push_if(&self.data, &mut label, &mut stack, p - 1, id);
                }
                if x + 1 < self.w {
                    push_if(&self.data, &mut label, &mut stack, p + 1, id);
                }
                if y > 0 {
                    push_if(&self.data, &mut label, &mut stack, p - self.w, id);
                }
                if y + 1 < self.h {
                    push_if(&self.data, &mut label, &mut stack, p + self.w, id);
                }
            }
            sizes.push(size);
        }

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

#[inline]
fn push_if(data: &[u8], label: &mut [u32], stack: &mut Vec<usize>, p: usize, id: u32) {
    if data[p] == FG && label[p] == 0 {
        label[p] = id;
        stack.push(p);
    }
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
}
