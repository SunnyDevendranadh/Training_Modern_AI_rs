//! Server-side SVG chart rendering.
//!
//! Pure-Rust generation of line and bar charts. Output is a self-contained
//! `<svg>` element that can be embedded directly in an HTML response — no
//! JavaScript, no external libraries.

use std::fmt::Write;

const W: f64 = 720.0;
const H: f64 = 360.0;
const PAD_L: f64 = 64.0;
const PAD_R: f64 = 24.0;
const PAD_T: f64 = 24.0;
const PAD_B: f64 = 56.0;

/// A single named series for a line chart.
pub struct Series<'a> {
    pub name: &'a str,
    pub color: &'a str,
    pub points: Vec<(f64, f64)>,
}

/// Axis scaling options.
#[derive(Clone, Copy)]
pub enum Scale {
    Linear,
    Log10,
}

pub struct LineChart<'a> {
    pub title: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub x_scale: Scale,
    pub y_scale: Scale,
    pub series: Vec<Series<'a>>,
    pub markers: Vec<Marker<'a>>,
}

/// A vertical or horizontal annotation line.
pub struct Marker<'a> {
    pub axis: Axis,
    pub value: f64,
    pub label: &'a str,
    pub color: &'a str,
}

#[derive(Clone, Copy)]
pub enum Axis {
    X,
    Y,
}

fn transform(v: f64, scale: Scale) -> f64 {
    match scale {
        Scale::Linear => v,
        Scale::Log10 => {
            if v <= 0.0 {
                f64::NEG_INFINITY
            } else {
                v.log10()
            }
        }
    }
}

fn data_range(values: impl Iterator<Item = f64>, scale: Scale) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        let t = transform(v, scale);
        if t.is_finite() {
            if t < lo {
                lo = t;
            }
            if t > hi {
                hi = t;
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if (hi - lo).abs() < f64::EPSILON {
        let pad = if hi.abs() < f64::EPSILON {
            1.0
        } else {
            hi.abs() * 0.1
        };
        return (lo - pad, hi + pad);
    }
    let pad = (hi - lo) * 0.05;
    (lo - pad, hi + pad)
}

fn tick_values(lo: f64, hi: f64, scale: Scale, count: usize) -> Vec<f64> {
    match scale {
        Scale::Linear => {
            let step = (hi - lo) / count as f64;
            (0..=count).map(|i| lo + step * i as f64).collect()
        }
        Scale::Log10 => {
            let lo_e = lo.floor() as i32;
            let hi_e = hi.ceil() as i32;
            (lo_e..=hi_e).map(|e| e as f64).collect()
        }
    }
}

fn fmt_tick(v: f64, scale: Scale) -> String {
    match scale {
        Scale::Linear => fmt_short(v),
        Scale::Log10 => fmt_short(10f64.powf(v)),
    }
}

fn fmt_short(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        return "0".into();
    }
    if a >= 1e12 {
        return format!("{:.1}T", v / 1e12);
    }
    if a >= 1e9 {
        return format!("{:.1}G", v / 1e9);
    }
    if a >= 1e6 {
        return format!("{:.1}M", v / 1e6);
    }
    if a >= 1e3 {
        return format!("{:.1}k", v / 1e3);
    }
    if a >= 1.0 {
        return format!("{:.2}", v);
    }
    if a >= 1e-3 {
        return format!("{:.3}", v);
    }
    if a >= 1e-6 {
        return format!("{:.1}µ", v * 1e6);
    }
    format!("{:.2e}", v)
}

impl<'a> LineChart<'a> {
    pub fn render(&self) -> String {
        let plot_w = W - PAD_L - PAD_R;
        let plot_h = H - PAD_T - PAD_B;

        let all_x = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|(x, _)| *x));
        let all_y = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|(_, y)| *y));
        let (x_lo, x_hi) = data_range(all_x, self.x_scale);
        let (y_lo, y_hi) = data_range(all_y, self.y_scale);

        let project = |x: f64, y: f64| -> (f64, f64) {
            let tx = transform(x, self.x_scale);
            let ty = transform(y, self.y_scale);
            let nx = (tx - x_lo) / (x_hi - x_lo);
            let ny = (ty - y_lo) / (y_hi - y_lo);
            (PAD_L + nx * plot_w, PAD_T + (1.0 - ny) * plot_h)
        };

        let mut s = String::new();
        let _ = write!(
            s,
            r##"<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">"##,
            W = W as i64,
            H = H as i64,
            title = html_escape(self.title),
        );
        let _ = write!(
            s,
            r##"<rect width="{}" height="{}" fill="#0f1117"/>"##,
            W as i64, H as i64
        );

        // Title
        let _ = write!(
            s,
            r##"<text x="{cx}" y="18" text-anchor="middle" fill="#e0e0e0" font-family="system-ui, sans-serif" font-size="14" font-weight="600">{t}</text>"##,
            cx = (W / 2.0) as i64,
            t = html_escape(self.title),
        );

        // Plot frame
        let _ = write!(
            s,
            r##"<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#2a2d3a" stroke-width="1"/>"##,
            x = PAD_L as i64,
            y = PAD_T as i64,
            w = plot_w as i64,
            h = plot_h as i64,
        );

        // Y grid + ticks
        for tv in tick_values(y_lo, y_hi, self.y_scale, 5) {
            let ny = (tv - y_lo) / (y_hi - y_lo);
            if !(0.0..=1.0).contains(&ny) {
                continue;
            }
            let py = PAD_T + (1.0 - ny) * plot_h;
            let _ = write!(
                s,
                r##"<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="#1c1f2a" stroke-width="1"/>"##,
                x1 = PAD_L as i64,
                x2 = (PAD_L + plot_w) as i64,
                y = py,
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" text-anchor="end" fill="#a0a0b0" font-family="ui-monospace, monospace" font-size="11">{t}</text>"##,
                x = (PAD_L - 8.0) as i64,
                y = py + 4.0,
                t = fmt_tick(tv, self.y_scale),
            );
        }

        // X grid + ticks
        for tv in tick_values(x_lo, x_hi, self.x_scale, 6) {
            let nx = (tv - x_lo) / (x_hi - x_lo);
            if !(0.0..=1.0).contains(&nx) {
                continue;
            }
            let px = PAD_L + nx * plot_w;
            let _ = write!(
                s,
                r##"<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" stroke="#1c1f2a" stroke-width="1"/>"##,
                x = px,
                y1 = PAD_T as i64,
                y2 = (PAD_T + plot_h) as i64,
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#a0a0b0" font-family="ui-monospace, monospace" font-size="11">{t}</text>"##,
                x = px,
                y = (PAD_T + plot_h + 16.0) as i64,
                t = fmt_tick(tv, self.x_scale),
            );
        }

        // Axis labels
        let _ = write!(
            s,
            r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#a0a0b0" font-family="system-ui, sans-serif" font-size="12">{t}</text>"##,
            x = (PAD_L + plot_w / 2.0) as i64,
            y = (H - 14.0) as i64,
            t = html_escape(self.x_label),
        );
        let _ = write!(
            s,
            r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#a0a0b0" font-family="system-ui, sans-serif" font-size="12" transform="rotate(-90, {x}, {y})">{t}</text>"##,
            x = 18,
            y = (PAD_T + plot_h / 2.0) as i64,
            t = html_escape(self.y_label),
        );

        // Series polylines
        for series in &self.series {
            let mut points = String::new();
            for (x, y) in &series.points {
                let (px, py) = project(*x, *y);
                if px.is_finite() && py.is_finite() {
                    let _ = write!(points, "{:.1},{:.1} ", px, py);
                }
            }
            let _ = write!(
                s,
                r##"<polyline fill="none" stroke="{c}" stroke-width="2" points="{p}"/>"##,
                c = series.color,
                p = points.trim(),
            );
        }

        // Markers
        for m in &self.markers {
            match m.axis {
                Axis::X => {
                    let tx = transform(m.value, self.x_scale);
                    if (x_lo..=x_hi).contains(&tx) {
                        let nx = (tx - x_lo) / (x_hi - x_lo);
                        let px = PAD_L + nx * plot_w;
                        let _ = write!(
                            s,
                            r##"<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" stroke="{c}" stroke-width="1.5" stroke-dasharray="4 4"/>"##,
                            x = px,
                            y1 = PAD_T as i64,
                            y2 = (PAD_T + plot_h) as i64,
                            c = m.color,
                        );
                        let _ = write!(
                            s,
                            r##"<text x="{x}" y="{y}" fill="{c}" font-family="ui-monospace, monospace" font-size="11" text-anchor="start">{t}</text>"##,
                            x = (px + 6.0) as i64,
                            y = (PAD_T + 14.0) as i64,
                            c = m.color,
                            t = html_escape(m.label),
                        );
                    }
                }
                Axis::Y => {
                    let ty = transform(m.value, self.y_scale);
                    if (y_lo..=y_hi).contains(&ty) {
                        let ny = (ty - y_lo) / (y_hi - y_lo);
                        let py = PAD_T + (1.0 - ny) * plot_h;
                        let _ = write!(
                            s,
                            r##"<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="1.5" stroke-dasharray="4 4"/>"##,
                            x1 = PAD_L as i64,
                            x2 = (PAD_L + plot_w) as i64,
                            y = py,
                            c = m.color,
                        );
                        let _ = write!(
                            s,
                            r##"<text x="{x}" y="{y}" fill="{c}" font-family="ui-monospace, monospace" font-size="11">{t}</text>"##,
                            x = (PAD_L + 6.0) as i64,
                            y = (py - 4.0) as i64,
                            c = m.color,
                            t = html_escape(m.label),
                        );
                    }
                }
            }
        }

        // Legend
        let mut lx = PAD_L + 8.0;
        let ly = PAD_T + 8.0;
        for series in &self.series {
            let _ = write!(
                s,
                r##"<rect x="{x}" y="{y}" width="12" height="3" fill="{c}"/>"##,
                x = lx,
                y = ly,
                c = series.color,
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" fill="#e0e0e0" font-family="system-ui, sans-serif" font-size="11">{t}</text>"##,
                x = (lx + 16.0) as i64,
                y = (ly + 5.0) as i64,
                t = html_escape(series.name),
            );
            lx += (series.name.len() as f64 * 6.5) + 36.0;
        }

        s.push_str("</svg>");
        s
    }
}

/// Bar chart with named categories.
pub struct BarChart<'a> {
    pub title: &'a str,
    pub y_label: &'a str,
    pub bars: Vec<(&'a str, f64, &'a str)>,
}

impl<'a> BarChart<'a> {
    pub fn render(&self) -> String {
        let plot_w = W - PAD_L - PAD_R;
        let plot_h = H - PAD_T - PAD_B;

        let max = self
            .bars
            .iter()
            .map(|(_, v, _)| *v)
            .fold(0.0_f64, f64::max)
            .max(1e-9);
        let n = self.bars.len().max(1) as f64;
        let bw = plot_w / n * 0.7;
        let gap = plot_w / n * 0.3;

        let mut s = String::new();
        let _ = write!(
            s,
            r##"<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">"##,
            W = W as i64,
            H = H as i64,
            title = html_escape(self.title),
        );
        let _ = write!(
            s,
            r##"<rect width="{}" height="{}" fill="#0f1117"/>"##,
            W as i64, H as i64
        );
        let _ = write!(
            s,
            r##"<text x="{cx}" y="18" text-anchor="middle" fill="#e0e0e0" font-family="system-ui, sans-serif" font-size="14" font-weight="600">{t}</text>"##,
            cx = (W / 2.0) as i64,
            t = html_escape(self.title),
        );

        // Y grid (5 ticks)
        for i in 0..=5 {
            let v = max * i as f64 / 5.0;
            let py = PAD_T + (1.0 - i as f64 / 5.0) * plot_h;
            let _ = write!(
                s,
                r##"<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="#1c1f2a"/>"##,
                x1 = PAD_L as i64,
                x2 = (PAD_L + plot_w) as i64,
                y = py,
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" text-anchor="end" fill="#a0a0b0" font-family="ui-monospace, monospace" font-size="11">{t}</text>"##,
                x = (PAD_L - 8.0) as i64,
                y = py + 4.0,
                t = fmt_short(v),
            );
        }

        // Bars
        for (i, (label, value, color)) in self.bars.iter().enumerate() {
            let bh = (value / max) * plot_h;
            let bx = PAD_L + (gap / 2.0) + (i as f64) * (bw + gap);
            let by = PAD_T + plot_h - bh;
            let _ = write!(
                s,
                r##"<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}"/>"##,
                x = bx,
                y = by,
                w = bw,
                h = bh,
                c = color,
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#e0e0e0" font-family="ui-monospace, monospace" font-size="11">{t}</text>"##,
                x = (bx + bw / 2.0) as i64,
                y = (by - 6.0) as i64,
                t = fmt_short(*value),
            );
            let _ = write!(
                s,
                r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#a0a0b0" font-family="system-ui, sans-serif" font-size="11">{t}</text>"##,
                x = (bx + bw / 2.0) as i64,
                y = (PAD_T + plot_h + 16.0) as i64,
                t = html_escape(label),
            );
        }

        // Y axis label
        let _ = write!(
            s,
            r##"<text x="{x}" y="{y}" text-anchor="middle" fill="#a0a0b0" font-family="system-ui, sans-serif" font-size="12" transform="rotate(-90, {x}, {y})">{t}</text>"##,
            x = 18,
            y = (PAD_T + plot_h / 2.0) as i64,
            t = html_escape(self.y_label),
        );

        s.push_str("</svg>");
        s
    }
}

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
