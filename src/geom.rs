//! Minimal integer geometry. The original C++ subclassed `cv::Point`; here we
//! keep a plain 2D pixel point and carry depth (`z`) separately in [`crate::Joint`].
//! No OpenCV, no floating-point image warping — just what the silhouette passes need.

/// A 2D point in image pixels.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Pt {
    pub x: i32,
    pub y: i32,
}

impl Pt {
    #[inline]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Euclidean distance between two pixels (`DrawAux::euclideanDist` in the original).
#[inline]
pub fn dist(a: Pt, b: Pt) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

/// Rasterize the straight line from `a` to `b` (inclusive) with Bresenham's
/// algorithm, returning every pixel it crosses. Used by the "is this segment
/// inside the silhouette?" checks that refine arm/elbow joints.
pub fn line(a: Pt, b: Pt) -> Vec<Pt> {
    let mut pts = Vec::new();
    let (mut x0, mut y0) = (a.x, a.y);
    let (x1, y1) = (b.x, b.y);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        pts.push(Pt::new(x0, y0));
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_endpoints_and_diagonal() {
        let l = line(Pt::new(0, 0), Pt::new(3, 3));
        assert_eq!(*l.first().unwrap(), Pt::new(0, 0));
        assert_eq!(*l.last().unwrap(), Pt::new(3, 3));
        assert_eq!(l.len(), 4);
    }

    #[test]
    fn dist_pythagoras() {
        assert!((dist(Pt::new(0, 0), Pt::new(3, 4)) - 5.0).abs() < 1e-4);
    }
}
