//! Mandelbrot explorer with deep zoom.
//!
//! Plain `f64` runs out of precision at about 1e-15 zoom. To go deeper, one reference
//! orbit (the screen centre) is computed in arbitrary precision, and every pixel only
//! tracks its small difference from that orbit in `f64` ("perturbation"). Glitches are
//! avoided by rebasing a pixel onto the start of the reference orbit whenever its own
//! value gets closer to zero than its difference is (Zhuoran's rebasing).
//!
//! Controls: scroll = zoom at cursor · left click = zoom in 2× · right click = zoom out 2×
//! · left drag = pan · + / - = more / fewer iterations · P = print location · R = reset
//! · H = full Retina resolution (slower) · F = full screen · Esc = quit

use dashu_float::FBig;
use fearless_simd::{dispatch, f64x4, mask64x4, prelude::*, Level};
use fearless_simd_macros::simd;
use rayon::prelude::*;
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Initial window size.
const WIDTH: usize = 960;
const HEIGHT: usize = 640;
const BAILOUT: f64 = 256.0 * 256.0;
/// Pixel size (complex-plane units per pixel) that zoom depth is measured against.
const START_SCALE: f64 = 3.2 / WIDTH as f64;
/// Deltas are plain f64; below this they start to lose range (f64 bottoms out near 1e-308).
const MIN_SCALE: f64 = 1e-290;

type Big = FBig;

fn big(x: f64, prec: usize) -> Big {
    Big::try_from(x).unwrap().with_precision(prec).value()
}

/// The value in decimal, to about `digits` significant digits.
fn decimal_string(x: &Big, digits: usize) -> String {
    let d: dashu_float::DBig = x.clone().with_base_and_precision::<10>(digits).value().with_rounding();
    d.to_string()
}

fn widen(x: &Big, prec: usize) -> Big {
    if x.precision() >= prec {
        x.clone()
    } else {
        x.clone().with_precision(prec).value()
    }
}

/// Bits of precision the centre needs at a given pixel size, with headroom.
fn precision_for(scale: f64) -> usize {
    (64.0 - scale.log2()).max(64.0) as usize
}

#[derive(Clone)]
struct View {
    re: Big,
    im: Big,
    /// Complex-plane units per pixel.
    scale: f64,
    /// Size in pixels.
    width: usize,
    height: usize,
    /// User multiplier on the automatic iteration count.
    iter_factor: f64,
    /// Iteration count earlier renders found necessary; deeper views need at least as many.
    iter_floor: usize,
}

impl View {
    /// The whole set, fitted to a window of the given size.
    fn home(width: usize, height: usize) -> Self {
        let scale = (3.2 / width as f64).max(2.4 / height as f64);
        let prec = precision_for(scale);
        View { re: big(-0.6, prec), im: big(0.0, prec), scale, width, height, iter_factor: 1.0, iter_floor: 0 }
    }

    /// Zoom depth in decimal digits (0 at the start).
    fn depth(&self) -> f64 {
        (START_SCALE / self.scale).log10()
    }

    fn max_iter(&self) -> usize {
        let auto = 300.0 + 250.0 * self.depth().max(0.0);
        (auto * self.iter_factor).max(self.iter_floor as f64).clamp(64.0, ITER_CAP as f64) as usize
    }

    /// Complex offset of a screen pixel from the centre.
    fn offset(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.width as f64 / 2.0) * self.scale, (self.height as f64 / 2.0 - y) * self.scale)
    }

    fn shift_centre(&mut self, dre: f64, dim: f64) {
        let prec = precision_for(self.scale);
        self.re = widen(&self.re, prec) + big(dre, prec);
        self.im = widen(&self.im, prec) + big(dim, prec);
    }

    /// Zoom by `factor` (>1 zooms in) keeping the point under (x, y) fixed.
    fn zoom_at(&mut self, x: f64, y: f64, factor: f64) {
        let new_scale = (self.scale / factor).clamp(MIN_SCALE, START_SCALE * 4.0);
        let actual = self.scale / new_scale;
        let (dre, dim) = self.offset(x, y);
        self.scale = new_scale;
        self.shift_centre(dre * (1.0 - 1.0 / actual), dim * (1.0 - 1.0 / actual));
    }

    fn pan_pixels(&mut self, dx: f64, dy: f64) {
        self.shift_centre(-dx * self.scale, dy * self.scale);
    }
}

/// The centre's orbit Z_0 = 0, Z_{n+1} = Z_n² + C, computed exactly and stored as f64
/// (the values themselves are small, only their differences need precision).
/// Returns None if the render was cancelled.
fn reference_orbit(view: &View, max_iter: usize, cancel: &dyn Fn() -> bool) -> Option<Vec<(f64, f64)>> {
    let prec = precision_for(view.scale);
    let cr = widen(&view.re, prec);
    let ci = widen(&view.im, prec);
    let two = big(2.0, prec);
    let mut zr = big(0.0, prec);
    let mut zi = big(0.0, prec);
    let mut orbit = Vec::with_capacity(max_iter + 1);
    orbit.push((0.0, 0.0));
    for n in 0..max_iter {
        if n % 1024 == 0 && cancel() {
            return None;
        }
        let zr2 = zr.sqr();
        let zi2 = zi.sqr();
        let new_zi = &two * &zr * &zi + &ci;
        zr = zr2 - zi2 + &cr;
        zi = new_zi;
        let (fr, fi) = (zr.to_f64().value(), zi.to_f64().value());
        if fr * fr + fi * fi > 4.0 {
            break;
        }
        orbit.push((fr, fi));
    }
    Some(orbit)
}

/// What happened to one pixel's orbit.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fate {
    /// Escaped at iteration `n` with |z|² = `mag`; `de` is the estimated distance to
    /// the set, in pixels.
    Escaped { n: usize, mag: f64, de: f64 },
    /// Proven inside the set: the orbit is contracting onto a cycle.
    Inside,
    /// Neither, within the iteration limit: more iterations are needed to tell.
    Unknown,
}

/// Minimal complex arithmetic for the jump table.
#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ONE: C = C { re: 1.0, im: 0.0 };

    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }

    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// How large the dropped δ² term may be, relative to the linear term, for a jump to be
/// taken. Smaller is more faithful and skips less. At 2⁻³⁰ a 1e30-deep view renders ~16×
/// faster; the pixels whose colour changes are isolated ones in chaotic regions, where the
/// colour is effectively noise anyway, and inside/outside never changed in testing.
const JUMP_EPS: f64 = 1.0 / (1u64 << 30) as f64;

/// Many iterations at once: while |δ| < r, running `len` iterations from reference index
/// m is (to within JUMP_EPS) the linear map δ ← Aδ + B·δc.
#[derive(Clone, Copy)]
struct Jump {
    a: C,
    b: C,
    r: f64,
}

/// Jumps over blocks of 2, 4, 8, … iterations along the reference orbit ("bilinear
/// approximation"). `levels[k][j]` covers 2^(k+1) iterations starting at reference
/// index 1 + j·2^(k+1).
struct Jumps {
    levels: Vec<Vec<Jump>>,
}

impl Jumps {
    /// No jumps: plain iteration.
    #[cfg(test)]
    fn none() -> Self {
        Jumps { levels: Vec::new() }
    }

    /// `dc_max` is the largest |δc| of any pixel in the image.
    fn build(orbit: &[(f64, f64)], dc_max: f64, eps: f64) -> Self {
        // One step from index m: δ ← 2Z_m·δ + δc, exact while |δ|² ≤ ε·|2Z_m·δ|.
        let single = |&(re, im): &(f64, f64)| {
            let a = C { re: 2.0 * re, im: 2.0 * im };
            Jump { a, b: C::ONE, r: eps * a.abs() }
        };
        // x then y: compose the maps; x's validity must also keep δ inside y's radius.
        let merge = |x: &Jump, y: &Jump| Jump {
            a: y.a.mul(x.a),
            b: y.a.mul(x.b).add(y.b),
            r: x.r.min((y.r - x.b.abs() * dc_max) / x.a.abs()).max(0.0),
        };
        // Single steps from index 1 to len-2 (each needs the next orbit value).
        let mut level: Vec<Jump> = orbit[1..orbit.len().saturating_sub(1).max(1)].iter().map(single).collect();
        let mut levels = Vec::new();
        while level.len() >= 2 {
            level = level.chunks_exact(2).map(|p| merge(&p[0], &p[1])).collect();
            levels.push(level.clone());
        }
        Jumps { levels }
    }

    /// The longest valid jump from reference index m ≥ 1 for |δ|² = `d2`, at most `room`
    /// iterations long.
    fn find(&self, m: usize, d2: f64, room: usize) -> Option<(&Jump, usize)> {
        let j = m - 1;
        // A block of 2^(k+1) iterations must start at a multiple of its length.
        let top = (j.trailing_zeros() as usize).min(self.levels.len());
        for k in (0..top).rev() {
            let len = 2usize << k;
            if len > room {
                continue;
            }
            let Some(jump) = self.levels[k].get(j >> (k + 1)) else { continue };
            if d2 < jump.r * jump.r {
                return Some((jump, len));
            }
        }
        None
    }
}

/// Iterate the point at offset (dcr, dci) from the reference, `scale` units per pixel.
/// The renderer runs `iterate_many`, which does the same for several pixels at once; this
/// one-pixel version is the reference it is tested against.
#[cfg(test)]
fn iterate(orbit: &[(f64, f64)], jumps: &Jumps, dcr: f64, dci: f64, max_iter: usize, scale: f64) -> Fate {
    let last = orbit.len() - 1;
    let dc = C { re: dcr, im: dci };
    let (mut dr, mut di) = (0.0f64, 0.0f64);
    // Full z = Z + δ, starting at z₀ = 0.
    let (mut fr, mut fi) = (0.0f64, 0.0f64);
    // dz/dpixel, for the distance estimate (scaled by the pixel size so it stays in range).
    let (mut pr, mut pi) = (0.0f64, 0.0f64);
    // dzₙ/dz₁: shrinks towards zero exactly when the orbit is drawn into an attracting cycle.
    let (mut qr, mut qi) = (1.0f64, 0.0f64);
    let mut m = 0;
    let mut n = 0; // iterations done
    while n < max_iter {
        let jump = if m > 0 { jumps.find(m, dr * dr + di * di, max_iter - n) } else { None };
        if let Some((jump, len)) = jump {
            // Skip `len` iterations; the derivatives follow the same linear map
            // (z = Z + δ, and Z does not depend on the pixel).
            let d = jump.a.mul(C { re: dr, im: di }).add(jump.b.mul(dc));
            let p = jump.a.mul(C { re: pr, im: pi }).add(C { re: jump.b.re * scale, im: jump.b.im * scale });
            let q = jump.a.mul(C { re: qr, im: qi });
            (dr, di, pr, pi, qr, qi) = (d.re, d.im, p.re, p.im, q.re, q.im);
            m += len;
            n += len;
        } else {
            // Derivatives use the full z, so they carry across rebasing unchanged.
            let npr = 2.0 * (fr * pr - fi * pi) + scale;
            pi = 2.0 * (fr * pi + fi * pr);
            pr = npr;
            if n > 0 {
                let nqr = 2.0 * (fr * qr - fi * qi);
                qi = 2.0 * (fr * qi + fi * qr);
                qr = nqr;
            }

            let (zr, zi) = orbit[m];
            // δ ← 2Zδ + δ² + δc
            let ndr = 2.0 * (zr * dr - zi * di) + (dr * dr - di * di) + dcr;
            let ndi = 2.0 * (zr * di + zi * dr) + 2.0 * dr * di + dci;
            dr = ndr;
            di = ndi;
            m += 1;
            n += 1;
        }
        if qr * qr + qi * qi < 1e-12 {
            return Fate::Inside;
        }

        let (zr, zi) = orbit[m];
        fr = zr + dr;
        fi = zi + di;
        let mag = fr * fr + fi * fi;
        if mag > BAILOUT {
            let de = mag.sqrt() * mag.ln() / (pr * pr + pi * pi).sqrt();
            return Fate::Escaped { n: n - 1, mag, de };
        }
        // Rebase: continue from the start of the reference orbit with the full value.
        if mag < dr * dr + di * di || m == last {
            dr = fr;
            di = fi;
            m = 0;
        }
    }
    Fate::Unknown
}

/// Pixels iterated side by side, one per SIMD lane.
const LANES: usize = 4;
/// Marks a lane with no pixel in it.
const IDLE: usize = usize::MAX;

/// How far one pixel's orbit has got, so a raised iteration limit can carry on from there
/// instead of starting over.
#[derive(Clone, Copy, Debug)]
struct Pixel {
    dcr: f64,
    dci: f64,
    dr: f64,
    di: f64,
    fr: f64,
    fi: f64,
    pr: f64,
    pi: f64,
    qr: f64,
    qi: f64,
    m: usize,
    n: usize,
}

impl Pixel {
    /// A pixel at offset (dcr, dci) from the reference, not iterated yet.
    fn new((dcr, dci): (f64, f64)) -> Self {
        Pixel { dcr, dci, dr: 0.0, di: 0.0, fr: 0.0, fi: 0.0, pr: 0.0, pi: 0.0, qr: 1.0, qi: 0.0, m: 0, n: 0 }
    }
}

/// The state of `iterate`, for `LANES` pixels at once.
#[derive(Default)]
struct Lanes {
    pixel: [usize; LANES],
    dcr: [f64; LANES],
    dci: [f64; LANES],
    dr: [f64; LANES],
    di: [f64; LANES],
    fr: [f64; LANES],
    fi: [f64; LANES],
    pr: [f64; LANES],
    pi: [f64; LANES],
    qr: [f64; LANES],
    qi: [f64; LANES],
    m: [usize; LANES],
    n: [usize; LANES],
}

impl Lanes {
    /// Put the next of `pixels` into lane `l`, or leave it idle if none are left.
    fn start(&mut self, l: usize, next: &mut usize, pixels: &[Pixel]) {
        self.pixel[l] = if *next < pixels.len() { *next } else { IDLE };
        let p = pixels.get(*next).copied().unwrap_or(Pixel::new((0.0, 0.0)));
        *next += 1;
        (self.dcr[l], self.dci[l], self.dr[l], self.di[l], self.fr[l], self.fi[l]) = (p.dcr, p.dci, p.dr, p.di, p.fr, p.fi);
        (self.pr[l], self.pi[l], self.qr[l], self.qi[l], self.m[l], self.n[l]) = (p.pr, p.pi, p.qr, p.qi, p.m, p.n);
    }

    /// The pixel in lane `l`, as far as it has got.
    fn save(&self, l: usize) -> Pixel {
        Pixel {
            dcr: self.dcr[l],
            dci: self.dci[l],
            dr: self.dr[l],
            di: self.di[l],
            fr: self.fr[l],
            fi: self.fi[l],
            pr: self.pr[l],
            pi: self.pi[l],
            qr: self.qr[l],
            qi: self.qi[l],
            m: self.m[l],
            n: self.n[l],
        }
    }
}

/// `iterate` for every one of `pixels`, `LANES` at a time: the plain steps run in SIMD,
/// jumps, rebasing and escape stay per pixel. A lane takes the next pixel as soon as its
/// own is decided, so lanes do not wait on each other. Gives the same results as `iterate`.
/// Pixels left undecided at `max_iter` are written back as far as they got, to continue.
#[simd]
fn iterate_many<S: Simd>(
    simd: S,
    orbit: &[(f64, f64)],
    jumps: &Jumps,
    pixels: &mut [Pixel],
    max_iter: usize,
    scale: f64,
    out: &mut [Fate],
) {
    let last = orbit.len() - 1;
    let v = |a: [f64; LANES]| -> f64x4<S> { a.simd_into(simd) };
    let (two, scale_v) = (f64x4::splat(simd, 2.0), f64x4::splat(simd, scale));
    let mut s = Lanes::default();
    let mut next = 0;
    for l in 0..LANES {
        s.start(l, &mut next, pixels);
    }
    while s.pixel.iter().any(|&p| p != IDLE) {
        // Per pixel: take a jump if one is valid, otherwise queue a plain step.
        let (mut step, mut later) = ([0i64; LANES], [0i64; LANES]);
        let (mut zr, mut zi) = ([0.0; LANES], [0.0; LANES]);
        for l in 0..LANES {
            if s.pixel[l] == IDLE {
                continue;
            }
            let (m, n) = (s.m[l], s.n[l]);
            let d2 = s.dr[l] * s.dr[l] + s.di[l] * s.di[l];
            let jump = if m > 0 { jumps.find(m, d2, max_iter - n) } else { None };
            if let Some((jump, len)) = jump {
                let d = jump.a.mul(C { re: s.dr[l], im: s.di[l] }).add(jump.b.mul(C { re: s.dcr[l], im: s.dci[l] }));
                let p = jump.a.mul(C { re: s.pr[l], im: s.pi[l] }).add(C { re: jump.b.re * scale, im: jump.b.im * scale });
                let q = jump.a.mul(C { re: s.qr[l], im: s.qi[l] });
                (s.dr[l], s.di[l], s.pr[l], s.pi[l], s.qr[l], s.qi[l]) = (d.re, d.im, p.re, p.im, q.re, q.im);
                s.m[l] += len;
                s.n[l] += len;
            } else {
                step[l] = -1;
                later[l] = if n > 0 { -1 } else { 0 };
                (zr[l], zi[l]) = orbit[m];
                s.m[l] += 1;
                s.n[l] += 1;
            }
        }

        // The plain steps, all lanes at once; lanes that jumped keep their values.
        let step: mask64x4<S> = step.simd_into(simd);
        let later: mask64x4<S> = later.simd_into(simd);
        let (fr, fi, pr, pi, qr, qi) = (v(s.fr), v(s.fi), v(s.pr), v(s.pi), v(s.qr), v(s.qi));
        let (dr, di, zr, zi) = (v(s.dr), v(s.di), v(zr), v(zi));
        let npr = two * (fr * pr - fi * pi) + scale_v;
        let npi = two * (fr * pi + fi * pr);
        let nqr = later.select(two * (fr * qr - fi * qi), qr);
        let nqi = later.select(two * (fr * qi + fi * qr), qi);
        let ndr = two * (zr * dr - zi * di) + (dr * dr - di * di) + v(s.dcr);
        let ndi = two * (zr * di + zi * dr) + two * dr * di + v(s.dci);
        s.pr = step.select(npr, pr).into();
        s.pi = step.select(npi, pi).into();
        s.qr = step.select(nqr, qr).into();
        s.qi = step.select(nqi, qi).into();
        s.dr = step.select(ndr, dr).into();
        s.di = step.select(ndi, di).into();

        // Per pixel: decide, rebase, or move on to the next pixel.
        for l in 0..LANES {
            if s.pixel[l] == IDLE {
                continue;
            }
            let (qr, qi) = (s.qr[l], s.qi[l]);
            let fate = if qr * qr + qi * qi < 1e-12 {
                Some(Fate::Inside)
            } else {
                let (zr, zi) = orbit[s.m[l]];
                let (dr, di) = (s.dr[l], s.di[l]);
                let (fr, fi) = (zr + dr, zi + di);
                (s.fr[l], s.fi[l]) = (fr, fi);
                let mag = fr * fr + fi * fi;
                if mag > BAILOUT {
                    let (pr, pi) = (s.pr[l], s.pi[l]);
                    let de = mag.sqrt() * mag.ln() / (pr * pr + pi * pi).sqrt();
                    Some(Fate::Escaped { n: s.n[l] - 1, mag, de })
                } else {
                    // At the limit the reference orbit ends only because the limit does: leave
                    // the pixel where it is, so a longer orbit can carry it on.
                    let limit = s.n[l] >= max_iter;
                    if mag < dr * dr + di * di || (s.m[l] == last && !limit) {
                        (s.dr[l], s.di[l], s.m[l]) = (fr, fi, 0);
                    }
                    limit.then_some(Fate::Unknown)
                }
            };
            if let Some(fate) = fate {
                if fate == Fate::Unknown {
                    pixels[s.pixel[l]] = s.save(l);
                }
                out[s.pixel[l]] = fate;
                s.start(l, &mut next, pixels);
            }
        }
    }
}

fn colour(fate: Fate) -> u32 {
    let Fate::Escaped { n, mag, de } = fate else { return 0 };
    // Smooth (fractional) iteration count, so colour bands blend instead of stepping.
    let nu = n as f64 + 2.0 - (0.5 * mag.ln()).log2();
    // Cycle a cosine palette at a fixed rate, so detail stays visible at any depth.
    let t = nu * 0.035;
    // Darken towards the boundary: filaments come out as crisp dark lines instead of colour
    // noise, but never fully black, so that black stays reserved for the set itself.
    let shade = 0.22 + 0.78 * (de / 2.0).clamp(0.0, 1.0).powf(0.5);
    let channel = |phase: f64| {
        let v = 0.5 + 0.5 * (std::f64::consts::TAU * (t + phase)).cos();
        (v * shade * 255.0) as u32
    };
    (channel(0.0) << 16) | (channel(0.15) << 8) | channel(0.30)
}

struct Frame {
    generation: u64,
    pixels: Vec<u32>,
    width: usize,
    height: usize,
    final_pass: bool,
    max_iter: usize,
    seconds: f64,
}

/// Share of pixels left undecided above which the iteration limit is raised.
const UNKNOWN_LIMIT: f64 = 0.002;
const ITER_CAP: usize = 2_000_000;

/// The pixels of one render pass, `step` screen pixels apart.
struct Pass {
    width: usize,
    step: usize,
    fates: Vec<Fate>,
    /// Indices into `fates` of the undecided pixels, and how far each got.
    undecided: Vec<usize>,
    states: Vec<Pixel>,
}

/// Every `step`-th pixel of `view`, iterated up to `max_iter`. None if cancelled.
fn first_pass(view: &View, orbit: &[(f64, f64)], jumps: &Jumps, level: Level, max_iter: usize, step: usize, cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Pass> {
    let (w, h) = (view.width.div_ceil(step), view.height.div_ceil(step));
    let scale = view.scale * step as f64;
    let rows: Vec<(Vec<Fate>, Vec<Pixel>)> = (0..h)
        .into_par_iter()
        .map(|row| {
            if cancelled() {
                return None;
            }
            let y = (row * step) as f64 + step as f64 / 2.0;
            let mut pixels: Vec<Pixel> =
                (0..w).map(|col| Pixel::new(view.offset((col * step) as f64 + step as f64 / 2.0, y))).collect();
            let mut fates = vec![Fate::Unknown; w];
            dispatch!(level, simd => iterate_many(simd, orbit, jumps, &mut pixels, max_iter, scale, &mut fates));
            Some((fates, pixels))
        })
        .collect::<Option<_>>()?;
    let mut pass = Pass { width: w, step, fates: Vec::with_capacity(w * h), undecided: Vec::new(), states: Vec::new() };
    for (fates, pixels) in rows {
        for (fate, pixel) in fates.into_iter().zip(pixels) {
            if fate == Fate::Unknown {
                pass.undecided.push(pass.fates.len());
                pass.states.push(pixel);
            }
            pass.fates.push(fate);
        }
    }
    Some(pass)
}

/// Continue the undecided pixels of `pass` up to `max_iter`. False if cancelled.
fn continue_pass(pass: &mut Pass, view: &View, orbit: &[(f64, f64)], jumps: &Jumps, level: Level, max_iter: usize, cancelled: &(dyn Fn() -> bool + Sync)) -> bool {
    let scale = view.scale * pass.step as f64;
    let mut fates = vec![Fate::Unknown; pass.states.len()];
    let complete = pass.states.par_chunks_mut(64).zip(fates.par_chunks_mut(64)).all(|(pixels, fates)| {
        if cancelled() {
            return false;
        }
        dispatch!(level, simd => iterate_many(simd, orbit, jumps, pixels, max_iter, scale, fates));
        true
    });
    if !complete {
        return false;
    }
    let (mut undecided, mut states) = (Vec::new(), Vec::new());
    for ((i, state), fate) in pass.undecided.iter().zip(&pass.states).zip(fates) {
        pass.fates[*i] = fate;
        if fate == Fate::Unknown {
            undecided.push(*i);
            states.push(*state);
        }
    }
    (pass.undecided, pass.states) = (undecided, states);
    true
}

/// Render `view` into frames: a coarse preview first, then full resolution. When too
/// many pixels are left undecided, raise the iteration limit and carry on with those.
fn render(view: View, generation: u64, latest: Arc<AtomicU64>, out: Sender<Frame>, wake: impl Fn()) {
    let started = Instant::now();
    let cancelled = || latest.load(Ordering::Relaxed) != generation;
    let mut max_iter = view.max_iter();
    let level = Level::new();
    let dc_max = (view.width as f64).hypot(view.height as f64) / 2.0 * view.scale;
    let Some(mut orbit) = reference_orbit(&view, max_iter, &cancelled) else { return };
    let mut jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);

    let send = |pass: &Pass, max_iter: usize, final_pass: bool| {
        let (width, step) = (view.width, pass.step);
        let colours: Vec<u32> = pass.fates.iter().map(|&f| colour(f)).collect();
        let pixels = (0..width * view.height).map(|i| colours[(i / width / step) * pass.width + (i % width) / step]).collect();
        let frame = Frame { generation, pixels, width, height: view.height, final_pass, max_iter, seconds: started.elapsed().as_secs_f64() };
        let sent = out.send(frame).is_ok();
        wake();
        sent
    };

    let Some(preview) = first_pass(&view, &orbit, &jumps, level, max_iter, 4, &cancelled) else { return };
    if !send(&preview, max_iter, false) {
        return;
    }
    let Some(mut pass) = first_pass(&view, &orbit, &jumps, level, max_iter, 1, &cancelled) else { return };
    let mut previous_undecided = 1.0;
    loop {
        let undecided = pass.undecided.len() as f64 / pass.fates.len() as f64;
        // Raise while most of the view is undecided (the limit is below the period of what
        // fills it), then only while doubling still settles a good share of the undecided
        // pixels: near cusps some stay undecided for any practical limit.
        let settling = undecided > 0.5 || undecided < 0.7 * previous_undecided;
        let raise = undecided > UNKNOWN_LIMIT && settling && max_iter < ITER_CAP;
        previous_undecided = undecided;
        if !send(&pass, max_iter, !raise) || !raise {
            return;
        }
        max_iter = (max_iter * 2).min(ITER_CAP);
        let Some(longer) = reference_orbit(&view, max_iter, &cancelled) else { return };
        orbit = longer;
        jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
        if !continue_pass(&mut pass, &view, &orbit, &jumps, level, max_iter, &cancelled) {
            return;
        }
    }
}

/// Maps screen pixels onto the last rendered frame, so zooms and pans show instantly
/// (stretched) while the real render catches up: frame pixel = offset + screen pixel · k.
#[derive(Clone, Copy)]
struct Warp {
    ox: f64,
    oy: f64,
    k: f64,
}

impl Warp {
    const IDENTITY: Warp = Warp { ox: 0.0, oy: 0.0, k: 1.0 };

    fn zoom_at(&mut self, x: f64, y: f64, factor: f64) {
        self.ox += self.k * x * (1.0 - 1.0 / factor);
        self.oy += self.k * y * (1.0 - 1.0 / factor);
        self.k /= factor;
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        self.ox -= self.k * dx;
        self.oy -= self.k * dy;
    }

    /// The window grew by (dw, dh) pixels around a fixed centre.
    fn resize(&mut self, dw: f64, dh: f64) {
        self.ox -= self.k * dw / 2.0;
        self.oy -= self.k * dh / 2.0;
    }

    /// Fill a screen `width` physical pixels wide, `d` physical pixels per view pixel.
    fn apply(&self, frame: &Frame, screen: &mut [u32], width: usize, d: f64) {
        screen.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
            let fy = (self.oy + (y as f64 + 0.5) / d * self.k).floor();
            for (x, px) in line.iter_mut().enumerate() {
                let fx = (self.ox + (x as f64 + 0.5) / d * self.k).floor();
                let inside = fx >= 0.0 && fy >= 0.0 && (fx as usize) < frame.width && (fy as usize) < frame.height;
                *px = if inside { frame.pixels[fy as usize * frame.width + fx as usize] } else { 0x101010 };
            }
        });
    }
}

struct Gfx {
    window: Rc<Window>,
    _context: Context<Rc<Window>>,
    surface: Surface<Rc<Window>, Rc<Window>>,
}

struct App {
    view: View,
    latest: Arc<AtomicU64>,
    tx: Sender<Frame>,
    rx: Receiver<Frame>,
    proxy: EventLoopProxy<()>,
    frame: Frame,
    warp: Warp,
    status: String,
    gfx: Option<Gfx>,
    /// Render one pixel per physical screen pixel (sharp on Retina, ~4× slower).
    sharp: bool,
    /// Mouse state, in view pixels.
    cursor: Option<(f64, f64)>,
    press: Option<(f64, f64)>,
    dragged: bool,
}

impl App {
    fn request_render(&mut self) {
        let generation = self.latest.fetch_add(1, Ordering::Relaxed) + 1;
        let (view, latest, tx, proxy) =
            (self.view.clone(), self.latest.clone(), self.tx.clone(), self.proxy.clone());
        thread::spawn(move || {
            render(view, generation, latest, tx, move || {
                let _ = proxy.send_event(());
            })
        });
        self.status = "rendering…".into();
        self.redraw();
    }

    fn redraw(&self) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }

    /// Physical screen pixels per view pixel.
    fn divisor(&self) -> f64 {
        match (&self.gfx, self.sharp) {
            (Some(gfx), false) => gfx.window.scale_factor(),
            _ => 1.0,
        }
    }

    /// Match the view to the window: keep the centre and pixel size, show more or less
    /// around it.
    fn sync_size(&mut self) {
        let Some(gfx) = &mut self.gfx else { return };
        let size = gfx.window.inner_size();
        let (Some(pw), Some(ph)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return; // minimised
        };
        gfx.surface.resize(pw, ph).expect("could not resize the drawing surface");
        let d = self.divisor();
        let width = (size.width as f64 / d).ceil() as usize;
        let height = (size.height as f64 / d).ceil() as usize;
        if (width, height) != (self.view.width, self.view.height) {
            self.warp.resize(width as f64 - self.view.width as f64, height as f64 - self.view.height as f64);
            self.view.width = width;
            self.view.height = height;
            self.request_render();
        }
        self.redraw();
    }

    fn zoom(&mut self, x: f64, y: f64, factor: f64) {
        let before = self.view.scale;
        self.view.zoom_at(x, y, factor);
        let actual = before / self.view.scale;
        if actual != 1.0 {
            self.warp.zoom_at(x, y, actual);
            self.request_render();
        }
    }

    fn print_location(&self) {
        let digits = (self.view.depth() + 20.0).max(20.0) as usize;
        println!(
            "re = {}\nim = {}\npixel = {:e}  zoom = 1e{:.1}  iterations = {}\n",
            decimal_string(&self.view.re, digits),
            decimal_string(&self.view.im, digits),
            self.view.scale,
            self.view.depth(),
            self.view.max_iter()
        );
    }

    fn title(&self) -> String {
        format!(
            "Mandelbrot — zoom 1e{:.1} · {} iterations · {}",
            self.view.depth(),
            self.view.max_iter(),
            self.status
        )
    }

    fn draw(&mut self) {
        let d = self.divisor();
        let title = self.title();
        let Some(gfx) = &mut self.gfx else { return };
        gfx.window.set_title(&title);
        let size = gfx.window.inner_size();
        let Ok(mut buffer) = gfx.surface.buffer_mut() else { return };
        self.warp.apply(&self.frame, &mut buffer, size.width as usize, d);
        let _ = buffer.present();
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, key: &Key, repeat: bool) {
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Character(c) => match c.as_str() {
                "+" | "=" | "]" => {
                    self.view.iter_factor *= 1.5;
                    self.request_render();
                }
                "-" | "[" => {
                    self.view.iter_factor /= 1.5;
                    self.view.iter_floor = (self.view.iter_floor as f64 / 1.5) as usize;
                    self.request_render();
                }
                _ if repeat => {}
                "p" | "P" => self.print_location(),
                "r" | "R" => {
                    self.view = View::home(self.view.width, self.view.height);
                    self.warp = Warp::IDENTITY;
                    self.request_render();
                }
                "h" | "H" => {
                    self.sharp = !self.sharp;
                    // The pixel grid changes: keep the same area on screen.
                    let (old_w, old_scale) = (self.view.width as f64, self.view.scale);
                    self.view.width = 0;
                    self.view.height = 0;
                    self.sync_size();
                    self.view.scale = old_scale * old_w / self.view.width as f64;
                    self.warp = Warp { ox: 0.0, oy: 0.0, k: old_w / self.view.width as f64 };
                    self.request_render();
                }
                "f" | "F" => {
                    if let Some(gfx) = &self.gfx {
                        let full = gfx.window.fullscreen().is_some();
                        gfx.window.set_fullscreen(if full { None } else { Some(Fullscreen::Borderless(None)) });
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Mandelbrot")
            .with_inner_size(LogicalSize::new(WIDTH as f64, HEIGHT as f64));
        let window = Rc::new(event_loop.create_window(attributes).expect("could not open a window"));
        let context = Context::new(window.clone()).expect("could not start drawing");
        let surface = Surface::new(&context, window.clone()).expect("could not create a drawing surface");
        self.gfx = Some(Gfx { window, _context: context, surface });
        self.sync_size();
        self.request_render();
    }

    fn user_event(&mut self, _: &ActiveEventLoop, _: ()) {
        // Take the newest frame for the current view, drop stale ones.
        let current = self.latest.load(Ordering::Relaxed);
        while let Ok(frame) = self.rx.try_recv() {
            if frame.generation == current {
                // Iterations this view turned out to need carry over to deeper ones.
                self.view.iter_floor = self.view.iter_floor.max(frame.max_iter);
                self.status = if frame.final_pass {
                    format!("{:.2}s", frame.seconds)
                } else {
                    "refining…".into()
                };
                self.frame = frame;
                self.warp = Warp::IDENTITY;
            }
        }
        self.redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => self.sync_size(),
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.key(event_loop, &event.logical_key, event.repeat)
            }
            WindowEvent::CursorMoved { position, .. } => {
                let d = self.divisor();
                let now = (position.x / d, position.y / d);
                if let (Some(start), Some(prev)) = (self.press, self.cursor) {
                    if (now.0 - start.0).hypot(now.1 - start.1) > 4.0 {
                        self.dragged = true;
                    }
                    if self.dragged {
                        let (dx, dy) = (now.0 - prev.0, now.1 - prev.1);
                        self.view.pan_pixels(dx, dy);
                        self.warp.pan(dx, dy);
                        self.request_render();
                    }
                }
                self.cursor = Some(now);
            }
            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    self.press = self.cursor;
                    self.dragged = false;
                }
                (MouseButton::Left, ElementState::Released) => {
                    if let (Some((x, y)), false) = (self.press.take(), self.dragged) {
                        self.zoom(x, y, 2.0);
                    }
                }
                (MouseButton::Right, ElementState::Pressed) => {
                    if let Some((x, y)) = self.cursor {
                        self.zoom(x, y, 0.5);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    // Trackpads: about 100 points of scrolling per zoom step.
                    MouseScrollDelta::PixelDelta(p) => {
                        p.y / self.gfx.as_ref().map_or(1.0, |g| g.window.scale_factor()) / 100.0
                    }
                };
                if let Some((x, y)) = self.cursor {
                    self.zoom(x, y, 1.25f64.powf(steps.clamp(-6.0, 6.0)));
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::<()>::with_user_event().build().expect("could not start");
    event_loop.set_control_flow(ControlFlow::Wait);
    let (tx, rx) = channel();
    let mut app = App {
        view: View::home(WIDTH, HEIGHT),
        latest: Arc::new(AtomicU64::new(0)),
        tx,
        rx,
        proxy: event_loop.create_proxy(),
        frame: Frame {
            generation: 0,
            pixels: vec![0; WIDTH * HEIGHT],
            width: WIDTH,
            height: HEIGHT,
            final_pass: false,
            max_iter: 0,
            seconds: 0.0,
        },
        warp: Warp::IDENTITY,
        status: String::new(),
        gfx: None,
        sharp: false,
        cursor: None,
        press: None,
        dragged: false,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashu_float::DBig;
    use std::str::FromStr;

    fn decimal(s: &str, prec: usize) -> Big {
        DBig::from_str(s).unwrap().with_base_and_precision::<2>(prec).value().with_rounding()
    }

    /// Escape count of c = centre + (dre, dim), iterated directly in big floats.
    fn direct(view: &View, dre: f64, dim: f64, max_iter: usize) -> Option<usize> {
        let prec = precision_for(view.scale);
        let cr = widen(&view.re, prec) + big(dre, prec);
        let ci = widen(&view.im, prec) + big(dim, prec);
        let (mut zr, mut zi) = (big(0.0, prec), big(0.0, prec));
        for n in 0..max_iter {
            let new_zi = big(2.0, prec) * &zr * &zi + &ci;
            zr = zr.sqr() - zi.sqr() + &cr;
            zi = new_zi;
            let (fr, fi) = (zr.to_f64().value(), zi.to_f64().value());
            if fr * fr + fi * fi > BAILOUT {
                return Some(n);
            }
        }
        None
    }

    fn assert_matches(view: &View, max_iter: usize) {
        let orbit = reference_orbit(view, max_iter, &|| false).unwrap();
        let dc_max = (view.width as f64).hypot(view.height as f64) / 2.0 * view.scale;
        let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
        for (x, y) in [(10.0, 20.0), (480.0, 320.0), (700.0, 100.0), (123.0, 600.0), (900.0, 500.0)] {
            let (dre, dim) = view.offset(x, y);
            let expected = direct(view, dre, dim, max_iter);
            let got = match iterate(&orbit, &jumps, dre, dim, max_iter, view.scale) {
                Fate::Escaped { n, .. } => Some(n as i64),
                _ => None,
            };
            match expected {
                None => assert_eq!(got, None, "pixel ({x},{y}) should not escape"),
                Some(n) => {
                    let g = got.expect("pixel escaped directly but not via perturbation");
                    assert!((g - n as i64).abs() <= 1, "pixel ({x},{y}): direct {n}, perturbed {g}");
                }
            }
        }
    }

    #[test]
    fn perturbation_matches_direct_at_start() {
        assert_matches(&View::home(WIDTH, HEIGHT), 500);
    }

    #[test]
    fn perturbation_matches_direct_deep() {
        // The neck at -0.75 + 0.001i (escapes after ~3000 iterations), 1e-30 wide pixels,
        // centre given to 50 digits: far past what f64 alone can resolve.
        let prec = 200;
        let view = View {
            re: decimal("-0.75000000000000000000000000000000731946192183019577", prec),
            im: decimal("0.00100000000000000000000000000012345678901234567890", prec),
            scale: 1e-30,
            width: WIDTH,
            height: HEIGHT,
            iter_factor: 1.0,
            iter_floor: 0,
        };
        assert_matches(&view, 20_000);
    }

    #[test]
    fn zoom_keeps_point_under_cursor() {
        let mut view = View::home(WIDTH, HEIGHT);
        let (x, y) = (200.0, 150.0);
        let point = |v: &View| {
            let (dre, dim) = v.offset(x, y);
            ((v.re.clone() + big(dre, 128)).to_f64().value(), (v.im.clone() + big(dim, 128)).to_f64().value())
        };
        let before = point(&view);
        view.zoom_at(x, y, 8.0);
        let after = point(&view);
        assert!((before.0 - after.0).abs() < 1e-12 && (before.1 - after.1).abs() < 1e-12);
    }
}

#[cfg(test)]
mod jump_tests {
    use super::*;

    /// Render with and without jumps. Returns the share of pixels that switch between
    /// inside / outside / undecided, the share whose colour changes visibly, and the speed-up.
    fn compare(view: &View, max_iter: usize) -> (f64, f64, f64) {
        let orbit = reference_orbit(view, max_iter, &|| false).unwrap();
        let dc_max = (view.width as f64).hypot(view.height as f64) / 2.0 * view.scale;
        let run = |jumps: &Jumps| {
            let t = Instant::now();
            let out: Vec<Fate> = (0..view.width * view.height)
                .into_par_iter()
                .map(|i| {
                    let (dcr, dci) = view.offset((i % view.width) as f64 + 0.5, (i / view.width) as f64 + 0.5);
                    iterate(&orbit, jumps, dcr, dci, max_iter, view.scale)
                })
                .collect();
            (out, t.elapsed().as_secs_f64())
        };
        let (plain, t_plain) = run(&Jumps::none());
        let (fast, t_fast) = run(&Jumps::build(&orbit, dc_max, JUMP_EPS));
        let px = plain.len() as f64;
        let class = plain.iter().zip(&fast).filter(|(a, b)| std::mem::discriminant(*a) != std::mem::discriminant(*b));
        let visible = plain.iter().zip(&fast).filter(|(a, b)| {
            let (a, b) = (colour(**a), colour(**b));
            (0..3).any(|k| (((a >> (8 * k)) & 255) as i32 - ((b >> (8 * k)) & 255) as i32).abs() > 24)
        });
        (class.count() as f64 / px, visible.count() as f64 / px, t_plain / t_fast)
    }

    /// The classic deep-zoom location in the seahorse valley.
    const SEAHORSE: (&str, &str) = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");

    /// The period-8007 copy of the set shown in the README.
    pub(super) const MINIBROT: (&str, &str) = (
        "-0.74364388703715870475219150611477977821525620794818",
        "0.13182590420531197049313205638514067897295227932892",
    );

    fn view(re: &str, im: &str, scale: f64) -> View {
        let prec = precision_for(scale);
        let parse = |s: &str| {
            use std::str::FromStr;
            dashu_float::DBig::from_str(s).unwrap().with_base_and_precision::<2>(prec).value().with_rounding()
        };
        View { re: parse(re), im: parse(im), scale, width: 480, height: 320, iter_factor: 1.0, iter_floor: 0 }
    }

    #[test]
    fn lanes_agree_with_iterate() {
        for v in [
            view("-0.6", "0", 3.2 / 480.0),
            view(SEAHORSE.0, SEAHORSE.1, 1e-14 / 480.0),
            view(MINIBROT.0, MINIBROT.1, START_SCALE / 5e28 * 2.0),
        ] {
            let max_iter = 20_000;
            let orbit = reference_orbit(&v, max_iter, &|| false).unwrap();
            let dc_max = (v.width as f64).hypot(v.height as f64) / 2.0 * v.scale;
            let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
            let dc: Vec<_> = (0..v.width * v.height)
                .map(|i| v.offset((i % v.width) as f64 + 0.5, (i / v.width) as f64 + 0.5))
                .collect();
            let mut lanes = vec![Fate::Unknown; dc.len()];
            let mut pixels: Vec<Pixel> = dc.iter().map(|&d| Pixel::new(d)).collect();
            dispatch!(Level::new(), simd => iterate_many(simd, &orbit, &jumps, &mut pixels, max_iter, v.scale, &mut lanes));
            for (i, &(dcr, dci)) in dc.iter().enumerate() {
                assert_eq!(lanes[i], iterate(&orbit, &jumps, dcr, dci, max_iter, v.scale), "pixel {i}");
            }
        }
    }

    #[test]
    fn continuing_matches_starting_over() {
        for (v, low, high) in [
            (view(SEAHORSE.0, SEAHORSE.1, 1e-6 / 480.0), 2_000, 16_000),
            (view(MINIBROT.0, MINIBROT.1, START_SCALE / 5e31 * 2.0), 8_224, 16_448),
        ] {
            continues(&v, low, high);
        }
    }

    fn continues(v: &View, low: usize, high: usize) {
        let dc_max = (v.width as f64).hypot(v.height as f64) / 2.0 * v.scale;
        let setup = |max_iter| {
            let orbit = reference_orbit(v, max_iter, &|| false).unwrap();
            let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
            (orbit, jumps)
        };
        let (orbit, jumps) = setup(low);
        let mut resumed = first_pass(v, &orbit, &jumps, Level::new(), low, 1, &|| false).unwrap();
        assert!(!resumed.undecided.is_empty());
        let (orbit, jumps) = setup(high);
        assert!(continue_pass(&mut resumed, v, &orbit, &jumps, Level::new(), high, &|| false));
        let fresh = first_pass(v, &orbit, &jumps, Level::new(), high, 1, &|| false).unwrap();
        let class = |f: &Fate| std::mem::discriminant(f);
        let differ = resumed.fates.iter().zip(&fresh.fates).filter(|(a, b)| class(a) != class(b)).count();
        assert!(differ * 1000 < fresh.fates.len(), "{differ} pixels differ");
        assert_eq!(resumed.undecided.len(), fresh.undecided.len());
    }

    #[test]
    fn raises_the_limit_when_nothing_is_decided() {
        // The default limit here is below the period of the copy filling the view, so the
        // first full pass decides nothing.
        let v = view(MINIBROT.0, MINIBROT.1, START_SCALE / 5e28 * 2.0);
        let (tx, rx) = channel();
        render(v, 1, Arc::new(AtomicU64::new(1)), tx, || {});
        let last = rx.try_iter().last().unwrap();
        assert!(last.final_pass);
        let coloured = last.pixels.iter().filter(|&&p| p != 0).count();
        assert!(coloured * 2 > last.pixels.len(), "only {coloured} of {} pixels coloured", last.pixels.len());
    }

    #[test]
    fn jumps_agree_with_plain_iteration() {
        for (v, iters) in [
            (view("-0.7436439", "0.1318259", 1e-4 / 480.0), 20_000),
            (view("-0.75000000000000000000000000000000731946192183019577", "0.00100000000000000000000000000012345678901234567890", 1e-30), 20_000),
            (view("-1.7549", "0", 0.03 / 480.0), 5_000),
            (view(SEAHORSE.0, SEAHORSE.1, 1e-10 / 480.0), 50_000),
            (view(SEAHORSE.0, SEAHORSE.1, 1e-14 / 480.0), 50_000),
            (view(SEAHORSE.0, SEAHORSE.1, 1e-20 / 480.0), 100_000),
            (view(SEAHORSE.0, SEAHORSE.1, 1e-30 / 480.0), 200_000),
        ] {
            let (class, visible, speedup) = compare(&v, iters);
            eprintln!("{speedup:.1}x faster; inside/outside differs {:.3}%, colour {:.2}%", class * 100.0, visible * 100.0);
            assert!(class < 0.001, "inside/outside differs for {:.3}% of pixels", class * 100.0);
            assert!(visible < 0.04, "colour differs visibly for {:.2}% of pixels", visible * 100.0);
        }
    }
}


#[cfg(test)]
mod bench {
    use super::*;

    /// Best of three runs, in seconds.
    fn time(render: impl Fn() -> Vec<u32>) -> f64 {
        (0..3)
            .map(|_| {
                let t = Instant::now();
                std::hint::black_box(render());
                t.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min)
    }

    /// Time full 960×640 renders at a few depths, one pixel at a time and in SIMD lanes:
    /// `cargo test --release bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn render_speed() {
        let parse = |s: &str, prec| {
            use std::str::FromStr;
            dashu_float::DBig::from_str(s).unwrap().with_base_and_precision::<2>(prec).value().with_rounding()
        };
        let seahorse = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");
        for (name, re, im, scale) in [
            ("home", "-0.6", "0", 3.2 / WIDTH as f64),
            ("1e-6 seahorse", seahorse.0, seahorse.1, 1e-6 / WIDTH as f64),
            ("1e-14 seahorse", seahorse.0, seahorse.1, 1e-14 / WIDTH as f64),
            ("5e28 minibrot", jump_tests::MINIBROT.0, jump_tests::MINIBROT.1, START_SCALE / 5e28),
        ] {
            let prec = precision_for(scale);
            let view = View { re: parse(re, prec), im: parse(im, prec), scale, width: WIDTH, height: HEIGHT, iter_factor: 1.0, iter_floor: 0 };
            let max_iter = view.max_iter() * 4;
            let orbit = reference_orbit(&view, max_iter, &|| false).unwrap();
            let dc_max = (view.width as f64).hypot(view.height as f64) / 2.0 * view.scale;
            let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
            let level = Level::new();
            let lanes = time(|| {
                (0..HEIGHT)
                    .into_par_iter()
                    .flat_map_iter(|row| {
                        let mut pixels: Vec<Pixel> = (0..WIDTH).map(|col| Pixel::new(view.offset(col as f64 + 0.5, row as f64 + 0.5))).collect();
                        let mut fates = vec![Fate::Unknown; WIDTH];
                        dispatch!(level, simd => iterate_many(simd, &orbit, &jumps, &mut pixels, max_iter, view.scale, &mut fates));
                        fates.into_iter().map(colour)
                    })
                    .collect()
            });
            let scalar = time(|| {
                (0..HEIGHT)
                    .into_par_iter()
                    .flat_map_iter(|row| {
                        let (view, orbit, jumps) = (&view, &orbit, &jumps);
                        (0..WIDTH).map(move |col| {
                            let (dcr, dci) = view.offset(col as f64 + 0.5, row as f64 + 0.5);
                            colour(iterate(orbit, jumps, dcr, dci, max_iter, view.scale))
                        })
                    })
                    .collect()
            });
            eprintln!("{name:>16}: scalar {:7.1} ms · lanes {:7.1} ms · {:.2}× ({max_iter} iterations)", scalar * 1e3, lanes * 1e3, scalar / lanes);
        }
    }
}

#[cfg(test)]
mod diag {
    use super::*;

    /// `iterate`, counting plain steps and jumps.
    fn counted(orbit: &[(f64, f64)], jumps: &Jumps, dcr: f64, dci: f64, max_iter: usize, scale: f64) -> (Fate, usize, usize) {
        let last = orbit.len() - 1;
        let dc = C { re: dcr, im: dci };
        let (mut dr, mut di, mut fr, mut fi, mut pr, mut pi, mut qr, mut qi) = (0.0, 0.0, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 1.0f64, 0.0f64);
        let (mut m, mut n, mut steps, mut js) = (0, 0, 0, 0);
        while n < max_iter {
            let jump = if m > 0 { jumps.find(m, dr * dr + di * di, max_iter - n) } else { None };
            if let Some((jump, len)) = jump {
                let d = jump.a.mul(C { re: dr, im: di }).add(jump.b.mul(dc));
                let p = jump.a.mul(C { re: pr, im: pi }).add(C { re: jump.b.re * scale, im: jump.b.im * scale });
                let q = jump.a.mul(C { re: qr, im: qi });
                (dr, di, pr, pi, qr, qi) = (d.re, d.im, p.re, p.im, q.re, q.im);
                m += len; n += len; js += 1;
            } else {
                let npr = 2.0 * (fr * pr - fi * pi) + scale;
                pi = 2.0 * (fr * pi + fi * pr); pr = npr;
                if n > 0 { let nqr = 2.0 * (fr * qr - fi * qi); qi = 2.0 * (fr * qi + fi * qr); qr = nqr; }
                let (zr, zi) = orbit[m];
                let ndr = 2.0 * (zr * dr - zi * di) + (dr * dr - di * di) + dcr;
                let ndi = 2.0 * (zr * di + zi * dr) + 2.0 * dr * di + dci;
                dr = ndr; di = ndi; m += 1; n += 1; steps += 1;
            }
            if qr * qr + qi * qi < 1e-12 { return (Fate::Inside, steps, js); }
            let (zr, zi) = orbit[m];
            fr = zr + dr; fi = zi + di;
            let mag = fr * fr + fi * fi;
            if mag > BAILOUT { return (Fate::Escaped { n: n - 1, mag, de: 0.0 }, steps, js); }
            if mag < dr * dr + di * di || m == last { dr = fr; di = fi; m = 0; }
        }
        (Fate::Unknown, steps, js)
    }

    #[test]
    #[ignore]
    fn what_raises_decide() {
        let v = jump_tests_view(START_SCALE / 1e6);
        let dc_max = (v.width as f64).hypot(v.height as f64) / 2.0 * v.scale;
        let mut max_iter = v.max_iter();
        let orbit = reference_orbit(&v, max_iter, &|| false).unwrap();
        let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
        let mut pass = first_pass(&v, &orbit, &jumps, Level::new(), max_iter, 1, &|| false).unwrap();
        while max_iter < 921_600 {
            let before: Vec<usize> = pass.undecided.clone();
            max_iter *= 2;
            let orbit = reference_orbit(&v, max_iter, &|| false).unwrap();
            let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
            let t = Instant::now();
            continue_pass(&mut pass, &v, &orbit, &jumps, Level::new(), max_iter, &|| false);
            let (mut inside, mut escaped) = (0, 0);
            for i in before {
                match pass.fates[i] { Fate::Inside => inside += 1, Fate::Escaped { .. } => escaped += 1, _ => {} }
            }
            eprintln!("to {max_iter:>7}: {:.2}s, {inside:>5} proven inside, {escaped:>5} escaped, {:>5} still undecided", t.elapsed().as_secs_f64(), pass.undecided.len());
        }
    }

    fn jump_tests_view(scale: f64) -> View {
        let prec = precision_for(scale);
        let parse = |s: &str| {
            use std::str::FromStr;
            dashu_float::DBig::from_str(s).unwrap().with_base_and_precision::<2>(prec).value().with_rounding()
        };
        View { re: parse("-0.743643887037158704752191506114774"), im: parse("0.131825904205311970493132056385139"), scale, width: WIDTH, height: HEIGHT, iter_factor: 1.0, iter_floor: 0 }
    }

    #[test]
    #[ignore]
    fn where_time_goes() {
        let parse = |s: &str, prec| {
            use std::str::FromStr;
            dashu_float::DBig::from_str(s).unwrap().with_base_and_precision::<2>(prec).value().with_rounding()
        };
        let sea = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");
        for (name, re, im, scale) in [
            ("home", "-0.6", "0", 3.2 / WIDTH as f64),
            ("1e3 seahorse", sea.0, sea.1, START_SCALE / 1e3),
            ("1e6 seahorse", sea.0, sea.1, START_SCALE / 1e6),
            ("1e14 seahorse", sea.0, sea.1, START_SCALE / 1e14),
            ("5e28 minibrot", jump_tests::MINIBROT.0, jump_tests::MINIBROT.1, START_SCALE / 5e28),
            ("5e31 minibrot", jump_tests::MINIBROT.0, jump_tests::MINIBROT.1, START_SCALE / 5e31),
        ] {
            let prec = precision_for(scale);
            let view = View { re: parse(re, prec), im: parse(im, prec), scale, width: WIDTH, height: HEIGHT, iter_factor: 1.0, iter_floor: 0 };
            // The app's own pipeline, end to end.
            let (tx, rx) = channel();
            let t = Instant::now();
            render(view.clone(), 1, Arc::new(AtomicU64::new(1)), tx, || {});
            let total = t.elapsed().as_secs_f64();
            let frames: Vec<Frame> = rx.try_iter().collect();
            if let Ok(dir) = std::env::var("SNAP_DIR") {
                let f = frames.last().unwrap();
                let mut ppm = format!("P6 {} {} 255\n", f.width, f.height).into_bytes();
                ppm.extend(f.pixels.iter().flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8]));
                std::fs::write(format!("{dir}/{}.ppm", name.replace(' ', "_")), ppm).unwrap();
            }
            let max_iter = frames.last().unwrap().max_iter;
            let passes: Vec<String> = frames.iter().map(|f| format!("{}@{:.2}s:{}", f.max_iter, f.seconds, f.pixels.iter().filter(|&&p| p != 0).count())).collect();

            let t = Instant::now();
            let orbit = reference_orbit(&view, max_iter, &|| false).unwrap();
            let t_orbit = t.elapsed().as_secs_f64();
            let dc_max = (view.width as f64).hypot(view.height as f64) / 2.0 * view.scale;
            let t = Instant::now();
            let jumps = Jumps::build(&orbit, dc_max, JUMP_EPS);
            let t_jumps = t.elapsed().as_secs_f64();

            // Work per kind of pixel at the final limit: [count, plain steps, jumps].
            let stats = (0..WIDTH * HEIGHT)
                .into_par_iter()
                .map(|i| {
                    let (dcr, dci) = view.offset((i % WIDTH) as f64 + 0.5, (i / WIDTH) as f64 + 0.5);
                    let (fate, s, j) = counted(&orbit, &jumps, dcr, dci, max_iter, view.scale);
                    let k = match fate { Fate::Escaped { .. } => 0, Fate::Inside => 1, Fate::Unknown => 2 };
                    let mut a = [[0u64; 3]; 3];
                    a[k] = [1, s as u64, j as u64];
                    a
                })
                .reduce(|| [[0u64; 3]; 3], |mut a, b| { for k in 0..3 { for x in 0..3 { a[k][x] += b[k][x]; } } a });
            let work: u64 = stats.iter().map(|s| s[1] + s[2]).sum();
            // Boundary fill: interior pixels whose 8 neighbours are all inside need no work.
            let px: Vec<(Fate, u64)> = (0..WIDTH * HEIGHT).into_par_iter().map(|i| {
                let (dcr, dci) = view.offset((i % WIDTH) as f64 + 0.5, (i / WIDTH) as f64 + 0.5);
                let (f, s, j) = counted(&orbit, &jumps, dcr, dci, max_iter, view.scale);
                (f, (s + j) as u64)
            }).collect();
            let inner = |i: usize| {
                let (x, y) = ((i % WIDTH) as i64, (i / WIDTH) as i64);
                (-1..=1).all(|dy| (-1..=1).all(|dx| {
                    let (nx, ny) = (x + dx, y + dy);
                    nx >= 0 && ny >= 0 && nx < WIDTH as i64 && ny < HEIGHT as i64 && px[(ny as usize) * WIDTH + nx as usize].0 == Fate::Inside
                }))
            };
            let skippable: u64 = (0..px.len()).filter(|&i| inner(i)).map(|i| px[i].1).sum();
            eprintln!("  boundary fill could skip {:.1}% of the final pass's work", 100.0 * skippable as f64 / work as f64);
            eprintln!("\n{name}: app total {total:.2}s, passes {passes:?}, orbit {} its {t_orbit:.3}s, jumps build {t_jumps:.3}s", orbit.len());
            for (k, label) in ["escaped", "inside", "undecided"].iter().enumerate() {
                let [c, s, j] = stats[k];
                eprintln!("  {label:>9}: {:5.1}% of pixels, {:5.1}% of work, avg {:.0} steps + {:.0} jumps",
                    100.0 * c as f64 / (WIDTH * HEIGHT) as f64, 100.0 * (s + j) as f64 / work as f64,
                    s as f64 / c.max(1) as f64, j as f64 / c.max(1) as f64);
            }
        }
    }
}
