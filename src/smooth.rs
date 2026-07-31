//! Per-joint temporal smoothing. The original kept a short circular history per
//! joint and replaced each raw joint with the per-axis median of that history
//! (`SkeletonPoints::computePoint`). A median over the last few frames rejects
//! the occasional single-frame outlier and rides through a one-frame dropout
//! without collapsing to the origin.

/// A bounded history of `[x, y, z]` samples with a per-axis, zero-ignoring
/// median. `z == 0` (or a missing sample) is treated as "no reading" for that
/// axis, matching the original.
pub struct MedianRing {
    cap: usize,
    buf: Vec<[i32; 3]>,
}

impl MedianRing {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: Vec::with_capacity(cap.max(1)),
        }
    }

    /// Feed the current raw sample (`None`, or a point with `x == 0`, means "not
    /// detected this frame" and is not stored) and return the smoothed joint —
    /// the per-axis median of the retained history, or `None` while the history
    /// is still empty.
    pub fn update(&mut self, sample: Option<[i32; 3]>) -> Option<[i32; 3]> {
        if let Some(s) = sample {
            if s[0] != 0 {
                if self.buf.len() == self.cap {
                    self.buf.remove(0);
                }
                self.buf.push(s);
            }
        }
        if self.buf.is_empty() {
            return None;
        }
        Some([
            self.axis_median(0),
            self.axis_median(1),
            self.axis_median(2),
        ])
    }

    fn axis_median(&self, axis: usize) -> i32 {
        let mut vals: Vec<i32> = self
            .buf
            .iter()
            .map(|s| s[axis])
            .filter(|&v| v != 0)
            .collect();
        if vals.is_empty() {
            return 0;
        }
        vals.sort_unstable();
        vals[vals.len() / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_rejects_outlier() {
        let mut r = MedianRing::new(5);
        r.update(Some([10, 10, 100]));
        r.update(Some([11, 11, 101]));
        r.update(Some([999, 999, 999])); // outlier
        r.update(Some([12, 12, 102]));
        let m = r.update(Some([10, 10, 100])).unwrap();
        assert!(m[0] < 100 && m[1] < 100, "median ignores the spike");
    }

    #[test]
    fn holds_through_dropout() {
        let mut r = MedianRing::new(3);
        r.update(Some([5, 6, 700]));
        let held = r.update(None).expect("history keeps last value");
        assert_eq!(held, [5, 6, 700]);
    }
}
