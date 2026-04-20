#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod onnx_assist;

use egui::{
    widgets::color_picker::{color_picker_color32, show_color, show_color_at, Alpha},
    Align2, Area, CollapsingHeader, Color32, ComboBox, Context, CursorIcon, FontFamily, Frame, Id,
    Key, Label, Order, PointerButton, Pos2, Rect, Response, RichText, Sense, Shape, Stroke, Ui,
    UiKind, Vec2, WidgetInfo, WidgetType,
};
use image::{GenericImageView, RgbaImage};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// Windows 下子进程不弹出黑色控制台（`python.exe` / `conda` / `cmd`）。
#[cfg(windows)]
fn command_hide_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn command_hide_console(_cmd: &mut Command) {}

#[cfg(windows)]
fn kill_training_process_tree(pid: u32) {
    if pid == 0 {
        return;
    }
    let mut k = Command::new("taskkill");
    k.args(["/PID", &pid.to_string(), "/T", "/F"]);
    command_hide_console(&mut k);
    let _ = k.status();
}

#[cfg(not(windows))]
fn kill_training_process_tree(pid: u32) {
    if pid == 0 {
        return;
    }
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .status();
}

#[derive(Clone)]
struct AssistPred {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    model_class_id: usize,
    conf: f32,
}

/// 交换类别 id 时的临时标记（不会与真实类别数冲突）。
const CLASS_SWAP_SENTINEL: usize = usize::MAX - 2;

/// 与 `image_root` 同级：按行记录类别名；第 1 条非注释行为索引 0，顺次对应 YOLO 标签中的 class 列。
const CLASS_LOG_FILENAME: &str = "class_log.txt";

fn class_log_file_header() -> &'static str {
    concat!(
        "# 类别索引说明：自上而下，第 1 条「非注释、非空」行为类别索引 0，第 2 条为索引 1，以此类推。\n",
        "# 与每张图同名 .txt 标签里每行开头的 class 数字一致。\n",
        "# 以 # 开头的整行会被程序忽略，可自用备注。\n",
        "\n",
    )
}

/// 界面强调色与表面色（深色主题）。
mod theme {
    use egui::Color32;

    pub const ACCENT: Color32 = Color32::from_rgb(92, 189, 255);
    pub const ACCENT_DIM: Color32 = Color32::from_rgb(43, 111, 171);
    pub const OK: Color32 = Color32::from_rgb(122, 214, 149);
    pub const WARN: Color32 = Color32::from_rgb(242, 191, 92);
    pub const DANGER: Color32 = Color32::from_rgb(228, 102, 114);
    pub const SURFACE: Color32 = Color32::from_rgb(17, 21, 28);
    pub const SURFACE_ELEVATED: Color32 = Color32::from_rgb(24, 30, 39);
    pub const SURFACE_SOFT: Color32 = Color32::from_rgb(31, 38, 49);
    pub const SURFACE_DEEP: Color32 = Color32::from_rgb(11, 14, 20);
    pub const BORDER: Color32 = Color32::from_rgb(58, 71, 90);
    pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(42, 51, 65);
    pub const TEXT: Color32 = Color32::from_rgb(232, 237, 244);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(150, 161, 180);
}

#[inline]
fn color_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

fn app_card(fill: Color32) -> Frame {
    Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin::same(14.0))
        .rounding(egui::Rounding::same(14.0))
        .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
}

fn status_chip(ui: &mut Ui, text: &str, tint: Color32) {
    Frame::default()
        .fill(color_alpha(tint, 34))
        .inner_margin(egui::Margin::symmetric(10.0, 5.0))
        .rounding(egui::Rounding::same(999.0))
        .stroke(Stroke::new(1.0, color_alpha(tint, 120)))
        .show(ui, |ui| {
            ui.label(RichText::new(text).small().strong().color(tint));
        });
}

fn metric_card(ui: &mut Ui, label: &str, value: impl Into<String>, detail: impl Into<String>, tint: Color32) {
    app_card(theme::SURFACE_SOFT).show(ui, |ui| {
        ui.set_min_height(68.0);
        ui.vertical(|ui| {
            ui.label(RichText::new(label).small().color(theme::TEXT_MUTED));
            ui.add_space(4.0);
            ui.label(
                RichText::new(value.into())
                    .size(21.0)
                    .strong()
                    .color(theme::TEXT),
            );
            ui.label(RichText::new(detail.into()).small().color(tint));
        });
    });
}

fn section_accordion<R>(
    ui: &mut Ui,
    section_id: u8,
    open_section: &mut Option<u8>,
    index_label: &str,
    title: &str,
    subtitle: &str,
    fill: Color32,
    accent: Color32,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let is_open = *open_section == Some(section_id);
    let header_inner = app_card(fill)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                Frame::default()
                    .fill(color_alpha(accent, 24))
                    .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                    .rounding(egui::Rounding::same(999.0))
                    .stroke(Stroke::new(1.0, color_alpha(accent, 90)))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(index_label)
                                .small()
                                .strong()
                                .color(accent),
                        );
                    });
                ui.vertical(|ui| {
                    ui.label(RichText::new(title).strong().size(17.0).color(theme::TEXT));
                    ui.label(RichText::new(subtitle).small().color(theme::TEXT_MUTED));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(if is_open { "收起" } else { "展开" })
                            .small()
                            .color(accent),
                    );
                });
            });
        });
    let header_rect = header_inner.response.rect;
    let header_id = ui.id().with("sidebar_accordion_header").with(section_id);
    let header_resp = ui
        .interact(header_rect, header_id, Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);
    if header_resp.clicked() {
        if is_open {
            *open_section = None;
        } else {
            *open_section = Some(section_id);
        }
    }
    if is_open {
        ui.add_space(8.0);
        Some(
            app_card(fill)
                .show(ui, |ui| add_contents(ui))
                .inner,
        )
    } else {
        None
    }
}

#[inline]
fn label_draft_textedit_id() -> Id {
    Id::new("yolo_label_draft_te")
}

#[inline]
fn smoothstep64(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// 选中框填充：「有填充 / 无填充」各持续 `half_period_secs`，交界处用 `transition_secs` 做 smoothstep 渐变。
#[inline]
fn selection_fill_blink_alpha(
    time_secs: f64,
    half_period_secs: f64,
    alpha_max: f64,
    transition_secs: f64,
) -> u8 {
    let period = half_period_secs * 2.0;
    if period <= f64::EPSILON || alpha_max <= f64::EPSILON {
        return 0;
    }
    let d = transition_secs
        .clamp(0.0, 1.0)
        .min(half_period_secs * 0.35);
    let p = time_secs.rem_euclid(period);
    let h = half_period_secs;
    let a = if p < d {
        smoothstep64(p / d)
    } else if p < h - d {
        1.0
    } else if p < h + d {
        1.0 - smoothstep64((p - (h - d)) / (2.0 * d))
    } else if p < period - d {
        0.0
    } else {
        smoothstep64((p - (period - d)) / d)
    };
    ((a * alpha_max * 255.0).round() as i32).clamp(0, 255) as u8
}

/// 角柄命中半径（屏幕像素）、绘制半径、边命中条带宽度等。
const CORNER_HANDLE_PX: f32 = 10.0;
const CORNER_DRAW_RADIUS: f32 = 5.0;
/// 悬停角点时相对基础半径的放大倍数（1.0 = 不放大）。
const CORNER_HOVER_RADIUS_SCALE: f32 = 2.15;
const EDGE_HIT_PX: f32 = 7.0;
/// 边上命中区两端留白，避免与角柄冲突（应大于 `CORNER_HANDLE_PX`）。
const EDGE_CORNER_SKIP_SCREEN: f32 = 14.0;
const EDGE_HOVER_THICK_EXTRA: f32 = 6.0;
const STROKE_ENDPOINT_INSET: f32 = 5.0;
const CROSSHAIR_DASH: f32 = 6.0;
const CROSSHAIR_GAP: f32 = 5.0;

#[inline]
fn dist_point_segment_2d(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let len_sq = ab.x * ab.x + ab.y * ab.y;
    let t = if len_sq < 1e-6 {
        0.0
    } else {
        (ap.x * ab.x + ap.y * ab.y) / len_sq
    };
    let t = t.clamp(0.0, 1.0);
    let proj = a + ab * t;
    p.distance(proj)
}

#[derive(Clone, Copy, PartialEq)]
enum ModelPreset {
    Yolo11n,
    Yolo11s,
}

impl ModelPreset {
    fn filename(self) -> &'static str {
        match self {
            ModelPreset::Yolo11n => "yolo11n.pt",
            ModelPreset::Yolo11s => "yolo11s.pt",
        }
    }

    /// Ultralytics assets v8.3.0
    fn asset_download_url(self) -> &'static str {
        match self {
            ModelPreset::Yolo11n => {
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
            }
            ModelPreset::Yolo11s => {
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
            }
        }
    }
}

#[derive(Clone)]
struct Bbox {
    /// Pixel coords, image space (x right, y down), inclusive-ish min/max
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    class_id: usize,
}

impl Bbox {
    fn normalize(&mut self, w: u32, h: u32) {
        let ww = w as f32;
        let hh = h as f32;
        self.min_x = self.min_x.clamp(0.0, ww);
        self.max_x = self.max_x.clamp(0.0, ww);
        self.min_y = self.min_y.clamp(0.0, hh);
        self.max_y = self.max_y.clamp(0.0, hh);
        if self.min_x > self.max_x {
            std::mem::swap(&mut self.min_x, &mut self.max_x);
        }
        if self.min_y > self.max_y {
            std::mem::swap(&mut self.min_y, &mut self.max_y);
        }
    }

    fn to_yolo_line(&self, w: u32, h: u32) -> String {
        let ww = w as f32;
        let hh = h as f32;
        let cx = ((self.min_x + self.max_x) * 0.5) / ww;
        let cy = ((self.min_y + self.max_y) * 0.5) / hh;
        let bw = (self.max_x - self.min_x) / ww;
        let bh = (self.max_y - self.min_y) / hh;
        format!(
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, cx, cy, bw, bh
        )
    }

    fn from_yolo_line(line: &str, w: u32, h: u32) -> Option<Bbox> {
        let mut it = line.split_whitespace();
        let cid: usize = it.next()?.parse().ok()?;
        let cx: f32 = it.next()?.parse().ok()?;
        let cy: f32 = it.next()?.parse().ok()?;
        let bw: f32 = it.next()?.parse().ok()?;
        let bh: f32 = it.next()?.parse().ok()?;
        let ww = w as f32;
        let hh = h as f32;
        let px_w = bw * ww;
        let px_h = bh * hh;
        let cx_px = cx * ww;
        let cy_px = cy * hh;
        Some(Bbox {
            min_x: cx_px - px_w * 0.5,
            min_y: cy_px - px_h * 0.5,
            max_x: cx_px + px_w * 0.5,
            max_y: cy_px + px_h * 0.5,
            class_id: cid,
        })
    }
}

/// 与已有标注 IoU 不低于此值时，采纳辅助框会跳过并保留原框（轴对齐框标准 IoU）。
const ASSIST_ADOPT_DUP_IOU: f32 = 0.9;

#[inline]
fn bbox_iou(a: &Bbox, b: &Bbox) -> f32 {
    let ix1 = a.min_x.max(b.min_x);
    let iy1 = a.min_y.max(b.min_y);
    let ix2 = a.max_x.min(b.max_x);
    let iy2 = a.max_y.min(b.max_y);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let aw = (a.max_x - a.min_x).max(0.0);
    let ah = (a.max_y - a.min_y).max(0.0);
    let bw = (b.max_x - b.min_x).max(0.0);
    let bh = (b.max_y - b.min_y).max(0.0);
    let area_a = aw * ah;
    let area_b = bw * bh;
    let union = area_a + area_b - inter;
    if union <= 1e-6 {
        return 0.0;
    }
    inter / union
}

#[inline]
fn assist_class_mask_allows(mask: &[bool], model_class_id: usize) -> bool {
    mask.get(model_class_id).copied().unwrap_or(true)
}

fn build_adopt_candidates(
    preds: &[AssistPred],
    class_mask: &[bool],
    assist_class_names_len: usize,
    w: u32,
    h: u32,
) -> Vec<Bbox> {
    let mut candidates = Vec::new();
    for pr in preds {
        let allowed = if assist_class_names_len == 0 {
            true
        } else {
            assist_class_mask_allows(class_mask, pr.model_class_id)
        };
        if !allowed {
            continue;
        }
        let mut b = Bbox {
            min_x: pr.min_x,
            min_y: pr.min_y,
            max_x: pr.max_x,
            max_y: pr.max_y,
            class_id: pr.model_class_id,
        };
        b.normalize(w, h);
        if b.max_x - b.min_x < 1.0 || b.max_y - b.min_y < 1.0 {
            continue;
        }
        candidates.push(b);
    }
    candidates
}

fn adopt_merge_candidates(
    mut merged: Vec<Bbox>,
    candidates: Vec<Bbox>,
) -> (Vec<Bbox>, usize, usize) {
    let mut added = 0usize;
    let mut skipped_iou = 0usize;
    for cand in candidates {
        let duplicate = merged
            .iter()
            .any(|ex| bbox_iou(&cand, ex) >= ASSIST_ADOPT_DUP_IOU);
        if duplicate {
            skipped_iou += 1;
            continue;
        }
        merged.push(cand);
        added += 1;
    }
    (merged, added, skipped_iou)
}

#[derive(Debug)]
struct GlobalAdoptSummary {
    images_scanned: usize,
    images_open_failed: usize,
    infer_failed: usize,
    total_added: usize,
    total_skipped_iou: usize,
    max_class_id: usize,
    had_any_box: bool,
}

#[derive(Clone, Default)]
enum DrawPhase {
    #[default]
    Idle,
    AwaitingSecondClick {
        ax: f32,
        ay: f32,
    },
}

/// E：整笔轨迹轴对齐最小外接矩形；F：连续柔性外接（笔画自相交形成视觉闭合即可，不必回到起点；嵌套只保留最大圈）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScribbleKind {
    Circumscribed,
    ContinuousCircumscribed,
}

/// 射线法：闭合折线 `poly`（首尾与首点相连）内部为 true。
#[inline]
fn point_in_polygon(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut c = false;
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        if (y0 > y) != (y1 > y) {
            let dy = y1 - y0;
            if dy.abs() > 1e-8 {
                let x_int = x0 + (y - y0) * (x1 - x0) / dy;
                if x < x_int {
                    c = !c;
                }
            }
        }
    }
    c
}

/// `inner` 的顶点是否全部落在 `outer` 所围区域内（用于嵌套闭合：只保留最大圈）。
fn polygon_contains_polygon(outer: &[(f32, f32)], inner: &[(f32, f32)]) -> bool {
    if outer.len() < 3 || inner.len() < 3 {
        return false;
    }
    inner
        .iter()
        .all(|&(x, y)| point_in_polygon(x, y, outer))
}

/// 线段 ab 与 cd 的交点（含端点落在对边上的情况）；近似平行返回 None。
fn segment_segment_intersection(
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    d: (f32, f32),
) -> Option<(f32, f32)> {
    const EPS: f32 = 1e-5;
    let rx = b.0 - a.0;
    let ry = b.1 - a.1;
    let sx = d.0 - c.0;
    let sy = d.1 - c.1;
    let den = rx * sy - ry * sx;
    if den.abs() < EPS {
        return None;
    }
    let qpx = c.0 - a.0;
    let qpy = c.1 - a.1;
    let t = (qpx * sy - qpy * sx) / den;
    let u = (qpx * ry - qpy * rx) / den;
    if t >= -EPS && t <= 1.0 + EPS && u >= -EPS && u <= 1.0 + EPS {
        Some((a.0 + t * rx, a.1 + t * ry))
    } else {
        None
    }
}

/// 轨迹点的轴对齐最小外接矩形（与 E 模式一致），含最小边长修正。
fn circumscribed_aabb_for_scribble(points: &[(f32, f32)], wf: f32, hf: f32) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &(x, y) in points {
        let x = x.clamp(0.0, wf);
        let y = y.clamp(0.0, hf);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    const MIN_EDGE: f32 = 2.0;
    if max_x - min_x < MIN_EDGE {
        let c = (min_x + max_x) * 0.5;
        min_x = (c - MIN_EDGE * 0.5).clamp(0.0, wf);
        max_x = (c + MIN_EDGE * 0.5).clamp(0.0, wf);
        if max_x - min_x < MIN_EDGE {
            max_x = (min_x + MIN_EDGE).min(wf);
        }
    }
    if max_y - min_y < MIN_EDGE {
        let c = (min_y + max_y) * 0.5;
        min_y = (c - MIN_EDGE * 0.5).clamp(0.0, hf);
        max_y = (c + MIN_EDGE * 0.5).clamp(0.0, hf);
        if max_y - min_y < MIN_EDGE {
            max_y = (min_y + MIN_EDGE).min(hf);
        }
    }
    (min_x, min_y, max_x, max_y)
}

#[derive(Clone, Copy, PartialEq)]
enum DragKind {
    Move,
    /// 角点 0..4：tl, tr, br, bl
    Resize(usize),
    /// 边 0=上 1=右 2=下 3=左（屏幕空间，与轴对齐框一致）
    ResizeEdge(u8),
}

#[derive(Clone)]
struct PendingBox {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

/// F 模式已接受的闭合块：保留多边形以便去掉「大圈里的子圈」。
#[derive(Clone)]
struct ScribbleClosedBlock {
    poly: Vec<(f32, f32)>,
    aabb: PendingBox,
}

#[inline]
fn pending_box_area(pb: &PendingBox) -> f32 {
    (pb.max_x - pb.min_x).max(0.0) * (pb.max_y - pb.min_y).max(0.0)
}

/// 两轴对齐框是否有正的面积交叠（仅边贴边不算，避免误删相邻框）。
fn pending_boxes_overlap_positive(a: &PendingBox, b: &PendingBox) -> bool {
    const EPS: f32 = 1e-3;
    let x0 = a.min_x.max(b.min_x);
    let y0 = a.min_y.max(b.min_y);
    let x1 = a.max_x.min(b.max_x);
    let y1 = a.max_y.min(b.max_y);
    x1 - x0 > EPS && y1 - y0 > EPS
}

/// 不重合目标检测：两框 AABB 若有面积交叠则删较小者（不合并成新框）；直到两两无交叠。
fn prune_scribble_closed_blocks_overlap_keep_larger(blocks: &mut Vec<ScribbleClosedBlock>) {
    const AREA_EPS: f32 = 1e-4;
    loop {
        let mut hit: Option<(usize, usize)> = None;
        'outer: for i in 0..blocks.len() {
            for j in i + 1..blocks.len() {
                if pending_boxes_overlap_positive(&blocks[i].aabb, &blocks[j].aabb) {
                    hit = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = hit else {
            break;
        };
        let ai = pending_box_area(&blocks[i].aabb);
        let aj = pending_box_area(&blocks[j].aabb);
        let drop = if ai < aj - AREA_EPS {
            i
        } else if aj < ai - AREA_EPS {
            j
        } else {
            j
        };
        blocks.remove(drop);
    }
}

#[derive(Clone, Copy)]
enum UndoScope {
    /// 当前图与类别列表等内存状态（不备份其它图片的 .txt）。
    Local,
    /// 同时备份数据集中全部标签文件（用于类别交换 / 合并）。
    DatasetLabels,
}

#[derive(Clone)]
struct UndoSnapshot {
    annotations: Vec<Bbox>,
    classes: Vec<String>,
    class_colors: Vec<Color32>,
    active_class_idx: usize,
    selected: Option<usize>,
    draw_phase: DrawPhase,
    draw_new_boxes_enabled: bool,
    scribble_kind: Option<ScribbleKind>,
    scribble_open_start: usize,
    scribble_closed_boxes: Vec<ScribbleClosedBlock>,
    pending_boxes_batch: Vec<PendingBox>,
    pending_box: Option<PendingBox>,
    show_label_window: bool,
    label_draft: String,
    /// `show_label_window` 为真且本字段为 `Some` 时表示在改已有框的类别，而非新框。
    label_edit_idx: Option<usize>,
    drag: Option<(DragKind, usize)>,
    /// 恢复全部图片对应 .txt（`None` 表示该路径原先无文件）。
    all_txt: Option<Vec<(PathBuf, Option<String>)>>,
    /// 数据集级撤销时恢复 `class_log.txt`：`None` 不恢复；`Some(None)` 表示操作前无此文件；`Some(Some)` 为当时全文。
    class_log_undo: Option<Option<String>>,
}

const UNDO_STACK_CAP: usize = 80;

#[inline]
fn class_name_edit_id(i: usize) -> Id {
    Id::new("class_name_edit").with(i)
}

enum TrainMsg {
    Line(String),
    ChildStarted { pid: u32 },
    Done(i32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrainingBackend {
    Conda,
    BuiltinCpu,
}

fn find_latest_best_pt_in_bundle(bundle: &Path) -> Option<PathBuf> {
    let runs = bundle.join("runs").join("detect");
    if !runs.is_dir() {
        return None;
    }
    let mut best_path: Option<PathBuf> = None;
    let mut best_mtime = std::time::SystemTime::UNIX_EPOCH;
    let entries = fs::read_dir(runs).ok()?;
    for entry in entries.flatten() {
        let sub = entry.path();
        if !sub.is_dir() {
            continue;
        }
        let cand = sub.join("weights").join("best.pt");
        if !cand.is_file() {
            continue;
        }
        let Ok(meta) = cand.metadata() else {
            continue;
        };
        let Ok(mt) = meta.modified() else {
            continue;
        };
        if mt >= best_mtime {
            best_mtime = mt;
            best_path = Some(cand);
        }
    }
    best_path
}

fn builtin_tool_base_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            dirs.push(dir.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if !dirs.iter().any(|d| d == &cwd) {
            dirs.push(cwd);
        }
    }
    dirs
}

fn find_builtin_tool(rel_path: &str) -> Option<PathBuf> {
    for base in builtin_tool_base_dirs() {
        let p = base.join(rel_path);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn find_builtin_cpu_train_exe() -> Option<PathBuf> {
    find_builtin_tool(r"embedded_tools\cpu_train_runner\cpu_train_runner.exe")
        .or_else(|| find_builtin_tool("cpu_train_runner.exe"))
}

fn find_builtin_onnx_export_exe() -> Option<PathBuf> {
    find_builtin_tool(r"embedded_tools\onnx_export_runner\onnx_export_runner.exe")
        .or_else(|| find_builtin_tool("onnx_export_runner.exe"))
}

fn spawn_logged_child(
    mut c: Command,
    tx: &mpsc::Sender<TrainMsg>,
) -> Result<i32, String> {
    let mut child = c.spawn().map_err(|e| format!("启动进程失败: {e}"))?;
    let pid = child.id();
    let _ = tx.send(TrainMsg::ChildStarted { pid });
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_handle = stdout.map(|out| {
        let tx2 = tx.clone();
        thread::spawn(move || {
            let r = std::io::BufReader::new(out);
            for line in r.lines().map_while(Result::ok) {
                let _ = tx2.send(TrainMsg::Line(line));
            }
        })
    });
    let stderr_handle = stderr.map(|err| {
        let tx2 = tx.clone();
        thread::spawn(move || {
            let r = std::io::BufReader::new(err);
            for line in r.lines().map_while(Result::ok) {
                let _ = tx2.send(TrainMsg::Line(line));
            }
        })
    });
    let code = child.wait().map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
    if let Some(h) = stdout_handle {
        let _ = h.join();
    }
    if let Some(h) = stderr_handle {
        let _ = h.join();
    }
    Ok(code)
}

fn default_conda_env_path() -> String {
    if let Ok(profile) = std::env::var("USERPROFILE") {
        format!(r"{profile}\anaconda3\envs\yolo")
    } else {
        r"C:\Users\YourName\anaconda3\envs\yolo".to_string()
    }
}

fn parse_conda_env_list_stdout(stdout: &[u8]) -> Vec<String> {
    let s = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        if let Some(last) = parts.last() {
            let p = last.trim().trim_end_matches('*').trim();
            if p == "*" {
                continue;
            }
            if p.len() > 1 && (p.contains(':') || p.starts_with('/')) {
                out.push(p.to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn detect_conda_env_paths() -> Vec<String> {
    let try_run = |mut cmd: Command| -> Option<Vec<String>> {
        let out = cmd.output().ok()?;
        if !out.status.success() {
            return None;
        }
        let v = parse_conda_env_list_stdout(&out.stdout);
        if v.is_empty() { None } else { Some(v) }
    };

    {
        let mut c = Command::new("conda");
        c.args(["env", "list"]);
        command_hide_console(&mut c);
        if let Some(v) = try_run(c) {
            return v;
        }
    }

    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.args(["/C", "conda env list"]);
        command_hide_console(&mut c);
        if let Some(v) = try_run(c) {
            return v;
        }
    }

    Vec::new()
}

fn try_load_chinese_font(ctx: &Context) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ];
    for path in candidates {
        if let Ok(bytes) = fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "zh_ui".into(),
                egui::FontData::from_owned(bytes).into(),
            );
            if let Some(v) = fonts.families.get_mut(&FontFamily::Proportional) {
                v.insert(0, "zh_ui".into());
            }
            if let Some(v) = fonts.families.get_mut(&FontFamily::Monospace) {
                v.push("zh_ui".into());
            }
            ctx.set_fonts(fonts);
            break;
        }
    }
}

fn is_image_file(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| {
            matches!(
                e.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "webp" | "gif"
            )
        })
}

/// YOLO 数据集常见布局：`.../images/.../a.jpg` 的标签在 `.../labels/.../a.txt`；否则与图片同目录的 `a.txt`。
fn label_txt_path_for_image(image_path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    let mut replaced = false;
    for c in image_path.components() {
        if let std::path::Component::Normal(name) = c {
            if !replaced && name.eq_ignore_ascii_case("images") {
                out.push("labels");
                replaced = true;
                continue;
            }
        }
        out.push(c.as_os_str());
    }
    if !replaced {
        image_path.with_extension("txt")
    } else {
        out.with_extension("txt")
    }
}

/// 训练打包时去掉相对路径首段的 `images` / `labels`，避免生成 `bundle/images/images/...`。
fn strip_first_dir_if(rel: &Path, dir_name: &str) -> PathBuf {
    let mut it = rel.components();
    if let Some(c0) = it.next() {
        if c0.as_os_str().eq_ignore_ascii_case(dir_name) {
            let mut p = PathBuf::new();
            for c in it {
                p.push(c);
            }
            return p;
        }
    }
    rel.to_path_buf()
}

/// 同路径下存在非空标签行（YOLO .txt）则视为已标注。
fn path_has_nonempty_label_file(image_path: &Path) -> bool {
    let lbl = label_txt_path_for_image(image_path);
    let Ok(text) = fs::read_to_string(&lbl) else {
        return false;
    };
    text.lines().any(|line| {
        let t = line.trim();
        !t.is_empty() && !t.starts_with('#')
    })
}

/// 仅所选目录当前层级内的图片文件（不进入子文件夹），与侧栏「选择图片目录」语义一致。
fn gather_images_in_dir_flat(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() && is_image_file(&p) {
            out.push(p);
        }
    }
    out.sort();
    out
}

/// 仅解析 Ultralytics `data.yaml` 顶层的 `path` / `train` / `val` 等 `key: value` 行（忽略 `names:` 块）。
fn parse_data_yaml_top_level_scalars(text: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for raw in text.lines() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Some((k, rest)) = t.split_once(':') else {
            continue;
        };
        let k = k.trim();
        if k.is_empty() || k == "names" {
            continue;
        }
        let v = rest
            .trim()
            .split('#')
            .next()
            .unwrap_or("")
            .trim();
        if v.is_empty() {
            continue;
        }
        let v = if (v.starts_with('"') && v.ends_with('"') && v.len() >= 2)
            || (v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2)
        {
            v[1..v.len() - 1].to_string()
        } else {
            v.to_string()
        };
        m.entry(k.to_string()).or_insert(v);
    }
    m
}

/// 当所选目录根下没有图片时，若存在 `data.yaml`，则按其 `path` 与 `train` / `val` 所列子目录收集图片（各目录仍只扫一层）。
fn gather_images_via_ultralytics_data_yaml(root: &Path) -> Vec<PathBuf> {
    let data_yaml = root.join("data.yaml");
    if !data_yaml.is_file() {
        return Vec::new();
    }
    let Ok(text) = fs::read_to_string(&data_yaml) else {
        return Vec::new();
    };
    let keys = parse_data_yaml_top_level_scalars(&text);
    let base: PathBuf = match keys.get("path").map(|s| PathBuf::from(s.trim())) {
        Some(p) if p.is_absolute() => p,
        Some(p) => root.join(p),
        None => root.to_path_buf(),
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    for key in ["train", "val"] {
        let Some(s) = keys.get(key) else {
            continue;
        };
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let p = base.join(part);
            if p.is_dir() {
                dirs.push(p);
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    let mut paths = Vec::new();
    for d in dirs {
        paths.extend(gather_images_in_dir_flat(&d));
    }
    paths.sort();
    paths.dedup();
    paths
}

/// 所选数据集目录：优先当前文件夹内一层图片；若无则尝试同目录下 Ultralytics `data.yaml` 的 train/val 路径。
fn gather_images_for_dataset_root(root: &Path) -> Vec<PathBuf> {
    let paths = gather_images_in_dir_flat(root);
    if paths.is_empty() {
        gather_images_via_ultralytics_data_yaml(root)
    } else {
        paths
    }
}

fn path_relative_to(full: &Path, root: &Path) -> PathBuf {
    full.strip_prefix(root)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| full.file_name().unwrap_or_default().into())
}

/// 解析为绝对路径后去掉 Windows 冗长前缀 `\\?\` / `\\?\UNC\...`。
/// 作为训练子进程的 `current_dir` 时，保留前缀易导致 Ultralytics 用相对路径下载/写文件触发 `OSError 22`。
fn path_abs_for_ospawn(p: &Path) -> PathBuf {
    let resolved = fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let s = resolved.to_string_lossy();
    let Some(stripped) = s.strip_prefix(r"\\?\") else {
        return resolved;
    };
    if let Some(unc) = stripped.strip_prefix("UNC\\") {
        PathBuf::from(format!(r"\\{}", unc))
    } else {
        PathBuf::from(stripped.to_string())
    }
}

fn copy_file_create_parent(src: &Path, dst: &Path) -> std::io::Result<u64> {
    if let Some(p) = dst.parent() {
        fs::create_dir_all(p)?;
    }
    fs::copy(src, dst)
}

fn write_data_yaml(root: &Path, classes: &[String]) -> std::io::Result<()> {
    let path = root.join("data.yaml");
    let mut f = fs::File::create(&path)?;
    let root_s = root.to_string_lossy().replace('\\', "/");
    writeln!(f, "path: {root_s}")?;
    writeln!(f, "train: images")?;
    writeln!(f, "val: images")?;
    writeln!(f, "nc: {}", classes.len())?;
    writeln!(f, "names:")?;
    for (i, name) in classes.iter().enumerate() {
        let esc = name.replace('\\', "\\\\").replace('\'', "\\'");
        writeln!(f, "  {i}: '{esc}'")?;
    }
    Ok(())
}

fn conda_python_executable(env_root: &Path) -> PathBuf {
    env_root.join("python.exe")
}

fn fit_image_rect(available: Rect, img_w: f32, img_h: f32) -> Rect {
    if img_w <= 0.0 || img_h <= 0.0 {
        return available;
    }
    let ar_img = img_w / img_h;
    let ar_avail = available.width() / available.height().max(1.0);
    let (rw, rh) = if ar_img > ar_avail {
        (available.width(), available.width() / ar_img)
    } else {
        (available.height() * ar_img, available.height())
    };
    let center = available.center();
    Rect::from_center_size(center, Vec2::new(rw, rh))
}

/// 顶栏全图缩略图：卡片、缩略图矩形、图像坐标下的当前视野、红框屏幕矩形。
struct OverviewMinimapLayout {
    card_rect: Rect,
    preview_rect: Rect,
    visible_in_image: Option<Rect>,
    red_on_screen: Option<Rect>,
}

fn compute_overview_minimap_layout(
    card_rect_outer: Rect,
    inner: Rect,
    disp: Rect,
    img_wf: f32,
    img_hf: f32,
) -> OverviewMinimapLayout {
    let card_rect = card_rect_outer.shrink2(Vec2::new(1.0, 2.0));
    let inset = 8.0;
    let preview_available = Rect::from_min_max(
        Pos2::new(card_rect.left() + inset, card_rect.top() + inset),
        Pos2::new(card_rect.right() - inset, card_rect.bottom() - inset),
    );
    let preview_rect = fit_image_rect(preview_available, img_wf, img_hf).shrink(1.0);

    let visible_screen = inner.intersect(disp);
    let visible_in_image = if visible_screen.width() > 1.0 && visible_screen.height() > 1.0 {
        let (x0, y0) = map_screen_to_image_px(visible_screen.min, disp, img_wf, img_hf);
        let (x1, y1) = map_screen_to_image_px(visible_screen.max, disp, img_wf, img_hf);
        Some(Rect::from_min_max(
            Pos2::new(x0.min(x1).clamp(0.0, img_wf), y0.min(y1).clamp(0.0, img_hf)),
            Pos2::new(x0.max(x1).clamp(0.0, img_wf), y0.max(y1).clamp(0.0, img_hf)),
        ))
    } else {
        None
    };

    let red_on_screen = visible_in_image.map(|view_rect| {
        let tl = image_to_screen(view_rect.min.x, view_rect.min.y, preview_rect, img_wf, img_hf);
        let br = image_to_screen(view_rect.max.x, view_rect.max.y, preview_rect, img_wf, img_hf);
        Rect::from_two_pos(tl, br)
    });

    OverviewMinimapLayout {
        card_rect,
        preview_rect,
        visible_in_image,
        red_on_screen,
    }
}

fn screen_to_image(
    p: Pos2,
    disp: Rect,
    img_w: f32,
    img_h: f32,
) -> Option<(f32, f32)> {
    if !disp.contains(p) {
        return None;
    }
    let u = (p.x - disp.min.x) / disp.width();
    let v = (p.y - disp.min.y) / disp.height();
    Some((u * img_w, v * img_h))
}

/// 与 `screen_to_image` 相同映射，但不要求点在矩形内（用于缩放后画布外缘仍可交互）。
fn map_screen_to_image_px(p: Pos2, disp: Rect, img_w: f32, img_h: f32) -> (f32, f32) {
    let u = (p.x - disp.min.x) / disp.width().max(1e-6);
    let v = (p.y - disp.min.y) / disp.height().max(1e-6);
    (u * img_w, v * img_h)
}

fn image_to_screen(x: f32, y: f32, disp: Rect, img_w: f32, img_h: f32) -> Pos2 {
    let u = x / img_w;
    let v = y / img_h;
    Pos2::new(disp.min.x + u * disp.width(), disp.min.y + v * disp.height())
}

fn compute_view_disp_rect(fit_base: Rect, zoom: f32, pan: Vec2) -> Rect {
    let z = zoom.max(0.02);
    Rect::from_center_size(fit_base.center() + pan, fit_base.size() * z)
}

fn adjust_pan_for_zoom_at_cursor(
    fit_base: Rect,
    pan: &mut Vec2,
    zoom_old: f32,
    zoom_new: f32,
    cursor: Pos2,
    img_w: f32,
    img_h: f32,
) {
    let z_old = zoom_old.max(0.02);
    let z_new = zoom_new.max(0.02);
    let w_old = fit_base.width() * z_old;
    let h_old = fit_base.height() * z_old;
    let center_old = fit_base.center() + *pan;
    let disp_old = Rect::from_center_size(center_old, Vec2::new(w_old, h_old));
    let Some((ix, iy)) = screen_to_image(cursor, disp_old, img_w, img_h) else {
        let r = z_new / z_old;
        pan.x *= r;
        pan.y *= r;
        return;
    };
    let w_new = fit_base.width() * z_new;
    let h_new = fit_base.height() * z_new;
    let cx_new = cursor.x + w_new * 0.5 - (ix / img_w) * w_new;
    let cy_new = cursor.y + h_new * 0.5 - (iy / img_h) * h_new;
    pan.x = cx_new - fit_base.center().x;
    pan.y = cy_new - fit_base.center().y;
}

fn palette_color(class_id: usize) -> Color32 {
    const PALETTE: &[Color32] = &[
        Color32::from_rgb(80, 200, 255),
        Color32::from_rgb(120, 255, 160),
        Color32::from_rgb(255, 180, 80),
        Color32::from_rgb(255, 120, 200),
        Color32::from_rgb(200, 160, 255),
        Color32::from_rgb(255, 230, 80),
        Color32::from_rgb(100, 220, 200),
        Color32::from_rgb(255, 100, 100),
        Color32::from_rgb(150, 200, 255),
        Color32::from_rgb(180, 255, 120),
        Color32::from_rgb(255, 140, 160),
        Color32::from_rgb(120, 180, 255),
        Color32::from_rgb(220, 200, 120),
        Color32::from_rgb(160, 255, 220),
        Color32::from_rgb(255, 160, 90),
        Color32::from_rgb(190, 140, 255),
    ];
    PALETTE[class_id % PALETTE.len()]
}

/// 类别颜色：独立弹层，避免与 egui 默认色钮的全局焦点/弹层与「输入标签」冲突；「确定」或双击底部当前颜色条关闭。
fn class_color_pick_button(
    ui: &mut Ui,
    row_idx: usize,
    srgba: &mut Color32,
    release_label_focus: bool,
    size: Vec2,
) {
    let popup_id = ui.id().with("class_color_popup").with(row_idx);
    let (rect, btn_resp) = ui.allocate_exact_size(size, Sense::click());
    btn_resp.widget_info(|| WidgetInfo::new(WidgetType::ColorButton));
    if ui.is_rect_visible(rect) {
        show_color_at(ui.painter(), *srgba, rect);
        ui.painter()
            .rect_stroke(rect, 2.0, ui.visuals().widgets.noninteractive.bg_stroke);
    }

    if btn_resp.clicked() {
        if release_label_focus {
            ui.memory_mut(|m| m.surrender_focus(label_draft_textedit_id()));
        }
        ui.memory_mut(|m| m.toggle_popup(popup_id));
    }

    if ui.memory(|m| m.is_popup_open(popup_id)) {
        let area_response = Area::new(popup_id)
            .kind(UiKind::Picker)
            .order(Order::Foreground)
            .fixed_pos(btn_resp.rect.max)
            .show(ui.ctx(), |ui| {
                ui.spacing_mut().slider_width = 275.0;
                Frame::popup(ui.style()).show(ui, |ui| {
                    color_picker_color32(ui, srgba, Alpha::Opaque);
                    ui.add_space(4.0);
                    let sw = egui::vec2(ui.spacing().slider_width, ui.spacing().interact_size.y);
                    let swatch = show_color(ui, *srgba, sw);
                    if swatch.double_clicked() {
                        ui.memory_mut(|m| m.surrender_focus(label_draft_textedit_id()));
                        ui.memory_mut(|m| m.close_popup());
                    }
                    ui.horizontal(|ui| {
                        if ui.button("确定").clicked() {
                            ui.memory_mut(|m| m.surrender_focus(label_draft_textedit_id()));
                            ui.memory_mut(|m| m.close_popup());
                        }
                    });
                    ui.label(
                        RichText::new("点击「确定」或在下方「当前颜色」预览条上双击即可关闭")
                            .weak()
                            .small(),
                    );
                });
            })
            .response;

        if !btn_resp.clicked()
            && (ui.input(|i| i.key_pressed(Key::Escape)) || area_response.clicked_elsewhere())
        {
            ui.memory_mut(|m| m.close_popup());
        }
    }
}

fn load_annotations(path: &Path, w: u32, h: u32) -> Vec<Bbox> {
    let Ok(file) = fs::File::open(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(b) = Bbox::from_yolo_line(t, w, h) {
            out.push(b);
        }
    }
    out
}

fn save_annotations(path: &Path, boxes: &[Bbox], w: u32, h: u32) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    for b in boxes {
        writeln!(f, "{}", b.to_yolo_line(w, h))?;
    }
    Ok(())
}

fn write_train_script(
    root: &Path,
    model_file: &str,
    epochs: u32,
) -> std::io::Result<PathBuf> {
    let script = root.join("ultralytics_train_run.py");
    let root_s = root.to_string_lossy().replace('\\', r"\\");
    let model_s = model_file.replace('\\', r"\\");
    let body = format!(
        r#"# -*- coding: utf-8 -*-
# Windows 下 DataLoader 会 spawn 子进程，必须把训练入口放在 if __name__ == "__main__" 内，否则会报错。
import os
import shutil
import traceback
from pathlib import Path
from ultralytics import YOLO

root = Path(r"{root_s}")
model_path = root / r"{model_s}"
data_yaml = root / "data.yaml"


def _norm_path(s: str) -> str:
    if s.startswith("\\\\?\\"):
        return s[4:]
    return s


def _find_latest_best_pt():
    runs = Path("runs") / "detect"
    if not runs.is_dir():
        return None
    best_path = None
    best_mtime = -1.0
    for sub in runs.iterdir():
        if not sub.is_dir():
            continue
        cand = sub / "weights" / "best.pt"
        if cand.is_file():
            try:
                mt = cand.stat().st_mtime
            except OSError:
                continue
            if mt > best_mtime:
                best_mtime = mt
                best_path = cand
    return best_path


def main() -> None:
    if not model_path.is_file():
        raise SystemExit(f"权重不存在: {{model_path}}")
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml 不存在: {{data_yaml}}")

    bundle_root = _norm_path(str(root.resolve()))
    os.chdir(bundle_root)

    data_s = _norm_path(str(data_yaml.resolve()))
    model_s = _norm_path(str(model_path.resolve()))
    print(
        "[训练] data=",
        data_s,
        "model=",
        model_s,
        "epochs={epochs}",
        flush=True,
    )

    m = YOLO(model_s)
    try:
        # 只传最通用参数；project/name 省略，输出默认在「当前目录」runs/（GUI 已把 cwd 设为训练包目录）
        m.train(data=data_s, epochs={epochs}, imgsz=640, workers=0)
    except BaseException as e:
        print(
            "[训练] 异常:",
            type(e).__module__ + "." + type(e).__name__ + ":",
            e,
            flush=True,
        )
        traceback.print_exc()
        raise
    print("训练完成.", flush=True)

    best = _find_latest_best_pt()
    if best is None:
        print("[训练] 未找到 runs/detect/*/weights/best.pt", flush=True)
        return
    print(f"[训练] best.pt={{best}}", flush=True)


if __name__ == "__main__":
    main()
"#
    );
    fs::write(&script, body)?;
    Ok(script)
}

fn download_url_to_file(url: &str, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("pt.part");
    let _ = fs::remove_file(&tmp);
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP 请求失败: {e}"))?;
    let status = resp.status();
    if !(200..300).contains(&status) {
        return Err(format!("下载失败: HTTP {status}"));
    }
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(&tmp)
        .map_err(|e| format!("无法写入临时文件 {}: {e}", tmp.display()))?;
    std::io::copy(&mut reader, &mut file).map_err(|e| format!("下载写入中断: {e}"))?;
    drop(file);
    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("无法保存到 {}: {e}", dest.display())
    })?;
    Ok(())
}

/// 解析 `yolo11n.pt` / `yolo11s.pt`：优先启动器(exe)同目录，再当前工作目录，再图片根目录，最后开发工程目录兜底。
/// 训练打包时只复制当前所选的那一个权重；若本地均不存在则仅下载到本次 `train_*` 目录，不会写入图片根目录。
fn weight_search_base_dirs(image_root: &Path) -> Vec<PathBuf> {
    let mut bases = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.to_path_buf();
            if !bases.iter().any(|b| b == &p) {
                bases.push(p);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if !bases.iter().any(|b| b == &cwd) {
            bases.push(cwd);
        }
    }
    let ir = image_root.to_path_buf();
    if !bases.iter().any(|b| b == &ir) {
        bases.push(ir);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !bases.iter().any(|b| b == &manifest) {
        bases.push(manifest);
    }
    bases
}

fn model_weights_path_in_root(image_root: &Path, preset: ModelPreset) -> Option<PathBuf> {
    let name = preset.filename();
    for base in weight_search_base_dirs(image_root) {
        let p = base.join(name);
        if p.is_file() {
            let ok = p.metadata().map(|m| m.len() >= 1024).unwrap_or(false);
            if ok {
                return Some(p);
            }
        }
    }
    None
}

/// 训练包根目录下仅保留名为 `keep_filename` 的 `.pt`，删除其余根级 `.pt`（避免误带入其它权重）。
fn bundle_root_keep_only_one_pt(bundle: &Path, keep_filename: &str) -> std::io::Result<()> {
    let Ok(rd) = fs::read_dir(bundle) else {
        return Ok(());
    };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let is_pt = p
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ex| ex.eq_ignore_ascii_case("pt"));
        if !is_pt {
            continue;
        }
        if p.file_name().and_then(|n| n.to_str()) != Some(keep_filename) {
            let _ = fs::remove_file(&p);
        }
    }
    Ok(())
}

fn prepare_training_bundle_in_dir(
    image_root: &Path,
    image_paths: &[PathBuf],
    classes: &[String],
    model_preset: ModelPreset,
    train_epochs: u32,
    tx: &mpsc::Sender<TrainMsg>,
) -> Result<PathBuf, String> {
    if !image_root.is_dir() {
        return Err(format!(
            "图片目录不存在或不可用: {}",
            image_root.display()
        ));
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let bundle = image_root.join(format!("train_{ts}"));
    fs::create_dir_all(&bundle).map_err(|e| format!("创建训练目录失败: {e}"))?;

    let img_out = bundle.join("images");
    let lbl_out = bundle.join("labels");
    let mut labels_copied = 0usize;

    for src_img in image_paths {
        let rel_raw = path_relative_to(src_img, image_root);
        let rel_img = strip_first_dir_if(&rel_raw, "images");
        let dst_img = img_out.join(&rel_img);
        copy_file_create_parent(src_img, &dst_img).map_err(|e| {
            format!("复制图片失败 {} -> {}: {e}", src_img.display(), dst_img.display())
        })?;

        let src_lbl = label_txt_path_for_image(src_img);
        if src_lbl.is_file() {
            let rel_lbl_raw = path_relative_to(&src_lbl, image_root);
            let rel_lbl = strip_first_dir_if(&rel_lbl_raw, "labels");
            let dst_lbl = lbl_out.join(&rel_lbl);
            copy_file_create_parent(&src_lbl, &dst_lbl).map_err(|e| {
                format!("复制标签失败 {}: {e}", src_lbl.display())
            })?;
            labels_copied += 1;
        }
    }

    if labels_copied == 0 {
        let _ = fs::remove_dir_all(&bundle);
        return Err(
            "训练包中未包含任何标签文件。请在图片旁保存与图片同名的 .txt 后再训练。".to_string(),
        );
    }

    let model_name = model_preset.filename();
    let dst_pt = bundle.join(model_name);
    if let Some(src_pt) = model_weights_path_in_root(image_root, model_preset) {
        copy_file_create_parent(&src_pt, &dst_pt).map_err(|e| format!("复制权重失败: {e}"))?;
        let _ = tx.send(TrainMsg::Line(format!(
            "[权重] 已复制 {} → 训练包（仅此预训练权重会进入 train_* 目录）",
            model_name
        )));
    } else {
        let url = model_preset.asset_download_url();
        let _ = tx.send(TrainMsg::Line(format!(
            "[权重] 未在本地找到 {model_name}，将仅下载到当前训练包（不写入图片根目录）…"
        )));
        let _ = tx.send(TrainMsg::Line(format!("[权重] {url}")));
        let _ = fs::remove_file(&dst_pt);
        download_url_to_file(url, &dst_pt)?;
        let _ = tx.send(TrainMsg::Line(format!(
            "[权重] 已保存至训练包: {}",
            dst_pt.display()
        )));
    }
    bundle_root_keep_only_one_pt(&bundle, model_name)
        .map_err(|e| format!("整理训练包内 .pt 文件失败: {e}"))?;

    write_data_yaml(&bundle, classes).map_err(|e| format!("写入 data.yaml 失败: {e}"))?;

    write_train_script(&bundle, model_name, train_epochs)
        .map_err(|e| format!("写入训练脚本失败: {e}"))?;

    Ok(bundle)
}

struct YoloTrainerApp {
    /// 用户任意选择的图片数据集目录（仅该目录内的图片，不含子文件夹）；训练包亦建在此目录下。
    image_root: PathBuf,
    /// `conda env list` 解析得到的环境根目录（含 python.exe）。
    conda_env_paths: Vec<String>,
    conda_env_idx: usize,
    conda_env_list_bootstrapped: bool,
    use_builtin_cpu_train: bool,
    sidebar_open_section: Option<u8>,
    sidebar_width: f32,
    model_preset: ModelPreset,
    classes: Vec<String>,
    /// 与 classes 一一对应，用于画布与列表显示。
    class_colors: Vec<Color32>,
    label_draft: String,
    show_label_window: bool,
    pending_box: Option<PendingBox>,
    /// 双击已有框时，待确认修改的框索引（与 `pending_box` 互斥）。
    label_edit_idx: Option<usize>,

    image_paths: Vec<PathBuf>,
    current_index: usize,
    rgba: Option<RgbaImage>,
    image_texture: Option<egui::TextureHandle>,
    texture_dirty: bool,
    annotations: Vec<Bbox>,
    draw_phase: DrawPhase,
    selected: Option<usize>,
    drag: Option<(DragKind, usize)>, // kind, bbox index

    /// 与 `selected` 同步；换选中框时清零下列动画状态。
    handles_anim_sel: Option<usize>,
    /// 角点悬停放大 0..1（平滑过渡到较大黄球）。
    corner_hover_radius_anim: [f32; 4],
    /// 各边加粗高亮 0..1（仅选中框描边）。
    edge_hover_anim: [f32; 4],

    train_log: Vec<String>,
    train_log_expanded: bool,
    training: bool,
    train_rx: Option<Receiver<TrainMsg>>,
    /// 训练子进程 PID（Windows 下用于 taskkill /T）。
    training_pid: Option<u32>,
    /// 用户在子进程启动前点了「停止」：收到 PID 后立即结束进程。
    training_stop_pending: bool,

    /// 辅助标注：ONNX 模型路径与 ort 会话（纯 Rust 推理，不依赖 Conda）。
    assist_onnx_path: Option<PathBuf>,
    assist_ort: Option<Arc<Mutex<ort::session::Session>>>,
    /// 模型类别显示名，默认 `unknown_0…`；随推理 / 全局采纳中出现的最大 id 自动扩展。
    assist_class_names: Vec<String>,
    /// 与 `assist_class_names` 一一对应：`true` 表示在画布上显示该模型类别的辅助预测框。
    assist_pred_class_on: Vec<bool>,
    assist_preds: Vec<AssistPred>,
    assist_busy: bool,
    assist_rx: Option<Receiver<Result<Vec<AssistPred>, String>>>,
    /// 对数据集内所有图批量「采纳辅助」时的接收端（与单图 `assist_rx` 互斥）。
    assist_batch_rx: Option<Receiver<Result<GlobalAdoptSummary, String>>>,
    assist_batch_busy: bool,
    /// 总开关：是否在画布上绘制辅助预测框（不影响后台推理与类别筛选状态）。
    assist_overlay_visible: bool,
    /// ONNX 辅助检测：后处理置信度下限（与导出模型内置阈值无关）。
    assist_onnx_conf: f32,

    /// 画布缩放（相对适应窗口的矩形）。
    view_zoom: f32,
    /// 画布平移（屏幕坐标，加到 fit 矩形中心）。
    view_pan: Vec2,
    last_canvas_inner: Option<Rect>,
    last_canvas_disp: Option<Rect>,
    /// 新框默认标签、类别列表中的「选用」索引。
    active_class_idx: usize,
    /// 切换选中框类别时滚轮累积（避免一帧跳过多类）。
    class_wheel_accum: f32,

    /// 按 R 切换：为 true 时才能在画布上拉新框；否则只可移动/缩放已有框。
    draw_new_boxes_enabled: bool,
    /// 按 E / F：柔性外接 / 连续柔性外接（与 R 互斥）。
    scribble_kind: Option<ScribbleKind>,
    scribble_active: bool,
    scribble_points: Vec<(f32, f32)>,
    /// F 模式：保留字段（撤销兼容）；闭合由边相交检测，不再依赖回到起点。
    scribble_open_start: usize,
    /// F 模式：已接受的闭合块（套圈只留外圈；框 AABB 重叠则删小留大、不合并）。
    scribble_closed_boxes: Vec<ScribbleClosedBlock>,
    /// 标签窗：一次确认的多个新框（连续柔性外接结束一笔时）。
    pending_boxes_batch: Vec<PendingBox>,

    /// 训练 epoch 数（在「开始训练」按钮上滚轮调整；普通每次 ±10，Shift 每次 ±50）。
    train_epochs: u32,
    train_epoch_scroll_accum: f32,

    undo_stack: Vec<UndoSnapshot>,
    /// 正在应用撤销时不入栈。
    undo_suspend: bool,

    /// 顶部「已标注」条：需重建时为 true（刷新列表、写入标签后）。
    annotated_strip_dirty: bool,
    /// `image_paths` 下标，对应磁盘上已有非空 .txt 的图。
    annotated_strip_indices: Vec<usize>,

    /// `class_log.txt` 需在下一帧写回磁盘。
    class_log_dirty: bool,
    /// 已做过首次从当前 `image_root` 读取 class_log（避免重复覆盖）。
    class_log_bootstrapped: bool,
}

impl YoloTrainerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        try_load_chinese_font(&cc.egui_ctx);
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = theme::SURFACE;
        visuals.panel_fill = theme::SURFACE;
        visuals.extreme_bg_color = theme::SURFACE_DEEP;
        visuals.widgets.noninteractive.bg_fill = theme::SURFACE_ELEVATED;
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, theme::BORDER_SUBTLE);
        visuals.widgets.inactive.bg_fill = theme::SURFACE_SOFT;
        visuals.widgets.inactive.weak_bg_fill = theme::SURFACE_SOFT;
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, theme::BORDER_SUBTLE);
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(42, 52, 68);
        visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(42, 52, 68);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, theme::ACCENT_DIM);
        visuals.widgets.active.bg_fill = theme::ACCENT_DIM;
        visuals.widgets.active.weak_bg_fill = theme::ACCENT_DIM;
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, theme::ACCENT);
        visuals.selection.bg_fill = color_alpha(theme::ACCENT, 96);
        visuals.hyperlink_color = theme::ACCENT;
        visuals.faint_bg_color = theme::SURFACE_SOFT;
        visuals.window_stroke = Stroke::new(1.0, theme::BORDER);
        cc.egui_ctx.set_visuals(visuals);
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.window_margin = egui::Margin::same(16.0);
        style.spacing.button_padding = Vec2::new(14.0, 9.0);
        style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(10.0);
        style.visuals.widgets.inactive.rounding = egui::Rounding::same(10.0);
        style.visuals.widgets.hovered.rounding = egui::Rounding::same(10.0);
        style.visuals.widgets.active.rounding = egui::Rounding::same(10.0);
        cc.egui_ctx.set_style(style);

        Self {
            image_root: PathBuf::from("."),
            conda_env_paths: Vec::new(),
            conda_env_idx: 0,
            conda_env_list_bootstrapped: false,
            use_builtin_cpu_train: false,
            sidebar_open_section: Some(1),
            sidebar_width: 344.0,
            model_preset: ModelPreset::Yolo11n,
            classes: Vec::new(),
            class_colors: Vec::new(),
            label_draft: String::new(),
            show_label_window: false,
            pending_box: None,
            label_edit_idx: None,
            image_paths: Vec::new(),
            current_index: 0,
            rgba: None,
            image_texture: None,
            texture_dirty: false,
            annotations: Vec::new(),
            draw_phase: DrawPhase::Idle,
            selected: None,
            drag: None,
            handles_anim_sel: None,
            corner_hover_radius_anim: [0.0; 4],
            edge_hover_anim: [0.0; 4],
            train_log: Vec::new(),
            train_log_expanded: false,
            training: false,
            train_rx: None,
            training_pid: None,
            training_stop_pending: false,
            assist_onnx_path: None,
            assist_ort: None,
            assist_class_names: Vec::new(),
            assist_pred_class_on: Vec::new(),
            assist_preds: Vec::new(),
            assist_busy: false,
            assist_rx: None,
            assist_batch_rx: None,
            assist_batch_busy: false,
            assist_overlay_visible: true,
            assist_onnx_conf: onnx_assist::ASSIST_ONNX_CONF,
            view_zoom: 1.0,
            view_pan: Vec2::ZERO,
            last_canvas_inner: None,
            last_canvas_disp: None,
            active_class_idx: 0,
            class_wheel_accum: 0.0,
            draw_new_boxes_enabled: false,
            scribble_kind: None,
            scribble_active: false,
            scribble_points: Vec::new(),
            scribble_open_start: 0,
            scribble_closed_boxes: Vec::new(),
            pending_boxes_batch: Vec::new(),
            train_epochs: 30,
            train_epoch_scroll_accum: 0.0,
            undo_stack: Vec::new(),
            undo_suspend: false,
            annotated_strip_dirty: true,
            annotated_strip_indices: Vec::new(),
            class_log_dirty: false,
            class_log_bootstrapped: false,
        }
    }

    fn class_log_path(&self) -> PathBuf {
        self.image_root.join(CLASS_LOG_FILENAME)
    }

    fn ensure_assist_pred_class_mask(&mut self) {
        let n = self.assist_class_names.len();
        self.assist_pred_class_on.resize(n, true);
    }

    /// 按推理中出现的最大类别 id，扩展 `unknown_0…` 与筛选掩码（与模型输出的类别 id 从 0 起一致）。
    fn ensure_assist_unknown_names(&mut self, min_len: usize) {
        let n = self.assist_class_names.len();
        if min_len <= n {
            return;
        }
        self.assist_class_names.reserve(min_len - n);
        for i in n..min_len {
            self.assist_class_names.push(format!("unknown_{}", i));
        }
        self.ensure_assist_pred_class_mask();
    }

    /// ONNX 模型类别 id 对应的展示名 / 采纳后写入 `class_log` 的名称（与模型 id 对齐，从 `unknown_0` 起）。
    fn assist_slot_class_name(&self, model_class_id: usize) -> String {
        self.assist_class_names
            .get(model_class_id)
            .cloned()
            .unwrap_or_else(|| format!("unknown_{}", model_class_id))
    }

    fn assist_model_class_label(&self, model_class_id: usize) -> String {
        self.assist_slot_class_name(model_class_id)
    }

    /// 当前模型类别 id 的辅助框是否应绘制（未载入类别表时一律显示）。
    fn assist_pred_class_visible(&self, model_class_id: usize) -> bool {
        if self.assist_class_names.is_empty() {
            return true;
        }
        assist_class_mask_allows(&self.assist_pred_class_on, model_class_id)
    }

    /// 若存在 `class_log.txt` 则读入并覆盖当前 `classes`（调用前通常已清空列表）。
    fn load_class_log_from_disk(&mut self) {
        let path = self.class_log_path();
        if !path.is_file() {
            return;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            return;
        };
        self.classes.clear();
        self.class_colors.clear();
        for line in text.lines() {
            let t = line.trim_end_matches('\r').trim();
            if !t.is_empty() && !t.starts_with('#') {
                self.classes.push(t.to_string());
            }
        }
        for i in 0..self.classes.len() {
            self.class_colors.push(palette_color(i));
        }
        if !self.classes.is_empty() {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        } else {
            self.active_class_idx = 0;
        }
    }

    fn save_class_log_to_disk(&mut self) {
        if !self.image_root.is_dir() {
            return;
        }
        let path = self.class_log_path();
        let mut out = String::new();
        out.push_str(class_log_file_header());
        for name in &self.classes {
            out.push_str(name.as_str());
            out.push('\n');
        }
        let _ = fs::write(path, out);
    }

    fn mark_class_log_dirty(&mut self) {
        self.class_log_dirty = true;
    }

    fn flush_class_log_if_dirty(&mut self) {
        if self.class_log_dirty {
            self.class_log_dirty = false;
            self.save_class_log_to_disk();
        }
    }

    fn refresh_conda_env_list(&mut self) {
        let preserve = self.conda_env_paths.get(self.conda_env_idx).cloned();
        self.conda_env_paths = detect_conda_env_paths();
        if self.conda_env_paths.is_empty() {
            self.conda_env_idx = 0;
            return;
        }
        fn norm_path(s: &str) -> String {
            s.replace('/', "\\").to_lowercase()
        }
        if let Some(prev) = preserve {
            let np = norm_path(&prev);
            if let Some(i) = self
                .conda_env_paths
                .iter()
                .position(|p| norm_path(p) == np || p == &prev)
            {
                self.conda_env_idx = i;
                return;
            }
        }
        let def = default_conda_env_path();
        let nd = norm_path(&def);
        if let Some(i) = self
            .conda_env_paths
            .iter()
            .position(|p| norm_path(p) == nd)
        {
            self.conda_env_idx = i;
        } else {
            self.conda_env_idx = 0;
        }
        self.conda_env_idx = self
            .conda_env_idx
            .min(self.conda_env_paths.len().saturating_sub(1));
    }

    fn selected_conda_root(&self) -> &str {
        self.conda_env_paths
            .get(self.conda_env_idx)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    fn display_color_for_class(&self, class_id: usize) -> Color32 {
        self.class_colors
            .get(class_id)
            .copied()
            .unwrap_or_else(|| palette_color(class_id))
    }

    fn cancel_assist_tasks(&mut self) {
        self.assist_preds.clear();
        let _ = self.assist_rx.take();
        self.assist_busy = false;
        let _ = self.assist_batch_rx.take();
        self.assist_batch_busy = false;
    }

    fn reset_canvas_runtime_cache(&mut self) {
        self.image_texture = None;
        self.texture_dirty = false;
        self.last_canvas_inner = None;
        self.last_canvas_disp = None;
    }

    fn switch_image_root(&mut self, new_root: PathBuf) {
        if self.image_root == new_root {
            self.refresh_image_list();
            return;
        }
        let _ = self.save_current_labels();
        self.cancel_assist_tasks();
        self.reset_canvas_runtime_cache();
        self.annotated_strip_indices.clear();
        self.current_index = 0;
        self.image_root = new_root;
        self.refresh_image_list();
    }

    fn refresh_image_list(&mut self) {
        self.cancel_assist_tasks();
        self.reset_canvas_runtime_cache();
        self.mark_annotated_strip_dirty();
        self.annotated_strip_indices.clear();
        self.classes.clear();
        self.class_colors.clear();
        self.active_class_idx = 0;
        self.load_class_log_from_disk();
        self.image_paths = if self.image_root.is_dir() {
            gather_images_for_dataset_root(&self.image_root)
        } else {
            Vec::new()
        };
        if self.current_index >= self.image_paths.len() {
            self.current_index = self.image_paths.len().saturating_sub(1);
        }
        self.load_current_image();
        // 在数据集根目录生成或更新 class_log.txt（含索引说明；尚无类别时也会落盘说明头，便于手写增类）
        if self.image_root.is_dir() {
            self.mark_class_log_dirty();
        }
    }

    fn mark_annotated_strip_dirty(&mut self) {
        self.annotated_strip_dirty = true;
    }

    fn rebuild_annotated_strip_if_dirty(&mut self) {
        if !self.annotated_strip_dirty {
            return;
        }
        self.annotated_strip_dirty = false;
        self.annotated_strip_indices = self
            .image_paths
            .iter()
            .enumerate()
            .filter(|(_, p)| path_has_nonempty_label_file(p))
            .map(|(i, _)| i)
            .collect();
    }

    fn go_to_image_index(&mut self, idx: usize) {
        if idx >= self.image_paths.len() || idx == self.current_index {
            return;
        }
        let _ = self.save_current_labels();
        self.current_index = idx;
        self.load_current_image();
    }

    fn go_prev_image(&mut self) {
        if self.current_index == 0 {
            return;
        }
        let _ = self.save_current_labels();
        self.current_index = self.current_index.saturating_sub(1);
        self.load_current_image();
    }

    fn go_next_image(&mut self) {
        if self.image_paths.is_empty() || self.current_index + 1 >= self.image_paths.len() {
            return;
        }
        let _ = self.save_current_labels();
        self.current_index += 1;
        self.load_current_image();
    }

    fn load_current_image(&mut self) {
        self.assist_preds.clear();
        let _ = self.assist_rx.take();
        self.assist_busy = false;
        self.undo_stack.clear();
        self.rgba = None;
        self.annotations.clear();
        self.draw_phase = DrawPhase::Idle;
        self.selected = None;
        self.drag = None;
        self.scribble_active = false;
        self.scribble_points.clear();
        self.scribble_open_start = 0;
        self.scribble_closed_boxes.clear();
        self.pending_boxes_batch.clear();
        self.pending_box = None;
        self.label_edit_idx = None;
        self.show_label_window = false;
        self.view_zoom = 1.0;
        self.view_pan = Vec2::ZERO;
        self.class_wheel_accum = 0.0;
        self.handles_anim_sel = None;
        self.corner_hover_radius_anim = [0.0; 4];
        self.edge_hover_anim = [0.0; 4];

        let Some(path) = self.image_paths.get(self.current_index) else {
            return;
        };
        let Ok(img) = image::open(path) else {
            return;
        };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        self.rgba = Some(rgba);

        let lbl = label_txt_path_for_image(path);
        self.annotations = load_annotations(&lbl, w, h);
        self.texture_dirty = true;

        let classes_len_before = self.classes.len();
        let max_ann_id = self.annotations.iter().map(|b| b.class_id).max().unwrap_or(0);
        while self.classes.len() <= max_ann_id {
            let k = self.classes.len();
            // 与 ONNX 槽位一致：缺省用 unknown_0 起，不改动 assist_class_names（尚未推理时）
            self.classes.push(format!("unknown_{}", k));
            self.class_colors.push(palette_color(k));
        }
        if !self.classes.is_empty() {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        }
        if self.classes.len() != classes_len_before {
            self.mark_class_log_dirty();
        }
        self.schedule_assist_infer();
    }

    fn schedule_assist_infer(&mut self) {
        if self.assist_batch_busy {
            return;
        }
        let Some(arc) = self.assist_ort.clone() else {
            return;
        };
        let Some(img_path) = self.image_paths.get(self.current_index).cloned() else {
            return;
        };
        let _ = self.assist_rx.take();
        let (tx, rx) = mpsc::channel();
        self.assist_rx = Some(rx);
        self.assist_busy = true;
        let conf_min = self.assist_onnx_conf.clamp(0.0, 1.0);
        thread::spawn(move || {
            let r = {
                let mut ses = match arc.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        let _ = tx.send(Err("ONNX 会话锁失败".to_string()));
                        return;
                    }
                };
                onnx_assist::predict_with_session(&mut *ses, &img_path, conf_min)
            };
            let mapped = r.map(|v| {
                v.into_iter()
                    .map(|d| AssistPred {
                        min_x: d.min_x,
                        min_y: d.min_y,
                        max_x: d.max_x,
                        max_y: d.max_y,
                        model_class_id: d.model_class_id,
                        conf: d.conf,
                    })
                    .collect::<Vec<_>>()
            });
            let _ = tx.send(mapped);
        });
    }

    fn poll_assist_infer(&mut self, ctx: &Context) {
        let Some(rx) = self.assist_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(r) => {
                self.assist_rx = None;
                self.assist_busy = false;
                match r {
                    Ok(v) => {
                        if let Some(m) = v.iter().map(|d| d.model_class_id).max() {
                            self.ensure_assist_unknown_names(m.saturating_add(1));
                        }
                        self.assist_preds = v;
                    }
                    Err(e) => self.train_log.push(format!("[辅助] {e}")),
                }
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.assist_rx = None;
                self.assist_busy = false;
            }
        }
    }

    /// 仅抑制显示：与当前正式标注重合度达 [`ASSIST_ADOPT_DUP_IOU`] 的辅助预测不绘制，
    /// 但仍保留在 `assist_preds` 中。这样当用户拖动/缩小/删除正式框导致 IoU 降低后，
    /// 被抑制的虚线框会自动重新显示出来。
    fn assist_pred_suppressed_by_annotations(&self, pr: &AssistPred) -> bool {
        let Some(img) = &self.rgba else {
            return false;
        };
        if self.assist_preds.is_empty() || self.annotations.is_empty() {
            return false;
        }
        let (w, h) = img.dimensions();
        let mut cand = Bbox {
            min_x: pr.min_x,
            min_y: pr.min_y,
            max_x: pr.max_x,
            max_y: pr.max_y,
            class_id: pr.model_class_id,
        };
        cand.normalize(w, h);
        if cand.max_x - cand.min_x < 1.0 || cand.max_y - cand.min_y < 1.0 {
            return false;
        }
        self.annotations
            .iter()
            .any(|ex| bbox_iou(&cand, ex) >= ASSIST_ADOPT_DUP_IOU)
    }

    fn reload_current_labels_from_disk(&mut self) {
        let Some(path) = self.image_paths.get(self.current_index) else {
            return;
        };
        let Some(img) = &self.rgba else {
            return;
        };
        let (w, h) = img.dimensions();
        let lbl = label_txt_path_for_image(path);
        self.annotations = load_annotations(&lbl, w, h);
    }

    /// 将当前 ONNX 辅助框写入正式标注：若与任一已有框 IoU≥[`ASSIST_ADOPT_DUP_IOU`] 则跳过（保留原框）；否则追加。类别 id 与模型一致。
    fn adopt_onnx_assist_to_annotations(&mut self) {
        let Some(img) = &self.rgba else {
            self.train_log
                .push("[辅助] 当前无图像，无法采纳辅助框".to_string());
            return;
        };
        if self.assist_ort.is_none() {
            return;
        }
        let (w, h) = img.dimensions();
        let names_len = self.assist_class_names.len();
        let candidates = build_adopt_candidates(
            &self.assist_preds,
            &self.assist_pred_class_on,
            names_len,
            w,
            h,
        );
        if candidates.is_empty() {
            self.train_log.push(
                "[辅助] 没有可采纳的辅助框（调高置信度、勾选预测类别，或等待推理完成）".to_string(),
            );
            return;
        }

        self.push_undo(UndoScope::Local);
        let adopted_ids: HashSet<usize> = candidates.iter().map(|c| c.class_id).collect();
        let (merged, added, skipped_iou) =
            adopt_merge_candidates(self.annotations.clone(), candidates);
        self.annotations = merged;

        let classes_len_before = self.classes.len();
        let max_cid = self
            .annotations
            .iter()
            .map(|b| b.class_id)
            .max()
            .unwrap_or(0);
        self.ensure_assist_unknown_names(max_cid.saturating_add(1));
        // 只「新追加」的类别槽写名：采纳的模型 id 用 unknown_*；其余新槽用 class_k，避免覆盖已有 class_log。
        while self.classes.len() <= max_cid {
            let k = self.classes.len();
            let name = if adopted_ids.contains(&k) {
                self.assist_slot_class_name(k)
            } else {
                format!("class_{k}")
            };
            self.classes.push(name);
            self.class_colors.push(palette_color(k));
        }
        if !self.classes.is_empty() {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        }
        if self.classes.len() != classes_len_before {
            self.mark_class_log_dirty();
        }

        let _ = self.save_current_labels();
        self.mark_annotated_strip_dirty();
        let pct = (ASSIST_ADOPT_DUP_IOU * 100.0).round() as i32;
        self.train_log.push(format!(
            "[辅助] 已采纳 {added} 个框，跳过 {skipped_iou} 个（与已有框 IoU≥{pct}% 视为同一目标）"
        ));
    }

    fn schedule_assist_global_adopt(&mut self) {
        if self.assist_ort.is_none() {
            return;
        }
        if self.image_paths.is_empty() {
            self.train_log
                .push("[辅助] 数据集为空，无法全局采纳".to_string());
            return;
        }
        if self.assist_batch_busy || self.assist_busy {
            return;
        }
        let Some(arc) = self.assist_ort.clone() else {
            return;
        };
        self.push_undo(UndoScope::DatasetLabels);
        let paths = self.image_paths.clone();
        let conf = self.assist_onnx_conf.clamp(0.0, 1.0);
        let mask = self.assist_pred_class_on.clone();
        let names_len = self.assist_class_names.len();
        let (tx, rx) = mpsc::channel();
        self.assist_batch_rx = Some(rx);
        self.assist_batch_busy = true;
        let _ = self.assist_rx.take();
        self.assist_busy = false;

        thread::spawn(move || {
            let mut images_scanned = 0usize;
            let mut images_open_failed = 0usize;
            let mut infer_failed = 0usize;
            let mut total_added = 0usize;
            let mut total_skipped_iou = 0usize;
            let mut max_class_id = 0usize;
            let mut had_any_box = false;

            for path in paths {
                images_scanned += 1;
                let Ok(img) = image::open(&path) else {
                    images_open_failed += 1;
                    continue;
                };
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let lbl = label_txt_path_for_image(&path);
                let existing = load_annotations(&lbl, w, h);

                let infer_result = {
                    let mut ses = match arc.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            let _ = tx.send(Err("ONNX 会话锁失败".to_string()));
                            return;
                        }
                    };
                    onnx_assist::predict_with_session(&mut *ses, &path, conf)
                };
                let Ok(detections) = infer_result else {
                    infer_failed += 1;
                    continue;
                };
                let preds: Vec<AssistPred> = detections
                    .into_iter()
                    .map(|d| AssistPred {
                        min_x: d.min_x,
                        min_y: d.min_y,
                        max_x: d.max_x,
                        max_y: d.max_y,
                        model_class_id: d.model_class_id,
                        conf: d.conf,
                    })
                    .collect();
                let candidates = build_adopt_candidates(&preds, &mask, names_len, w, h);
                let (merged, added, skipped) = adopt_merge_candidates(existing, candidates);
                total_added += added;
                total_skipped_iou += skipped;
                if !merged.is_empty() {
                    had_any_box = true;
                }
                for b in &merged {
                    max_class_id = max_class_id.max(b.class_id);
                }
                if let Err(e) = save_annotations(&lbl, &merged, w, h) {
                    let _ = tx.send(Err(format!("写入 {}: {e}", lbl.display())));
                    return;
                }
            }

            let _ = tx.send(Ok(GlobalAdoptSummary {
                images_scanned,
                images_open_failed,
                infer_failed,
                total_added,
                total_skipped_iou,
                max_class_id,
                had_any_box,
            }));
        });
    }

    fn poll_assist_batch(&mut self, ctx: &Context) {
        let Some(rx) = self.assist_batch_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(sum)) => {
                self.assist_batch_rx = None;
                self.assist_batch_busy = false;
                if sum.had_any_box {
                    let cb = self.classes.len();
                    self.ensure_assist_unknown_names(sum.max_class_id.saturating_add(1));
                    while self.classes.len() <= sum.max_class_id {
                        let k = self.classes.len();
                        self.classes.push(self.assist_slot_class_name(k));
                        self.class_colors.push(palette_color(k));
                    }
                    if self.classes.len() != cb {
                        self.mark_class_log_dirty();
                    }
                }
                if !self.classes.is_empty() {
                    self.active_class_idx = self
                        .active_class_idx
                        .min(self.classes.len().saturating_sub(1));
                }
                self.assist_preds.clear();
                self.reload_current_labels_from_disk();
                self.schedule_assist_infer();
                self.mark_annotated_strip_dirty();
                let pct = (ASSIST_ADOPT_DUP_IOU * 100.0).round() as i32;
                self.train_log.push(format!(
                    "[辅助] 全局采纳完成：处理 {} 张，打开失败 {}，推理失败 {}，新增框 {}，跳过(IoU≥{pct}%) {}。",
                    sum.images_scanned,
                    sum.images_open_failed,
                    sum.infer_failed,
                    sum.total_added,
                    sum.total_skipped_iou
                ));
                ctx.request_repaint();
            }
            Ok(Err(e)) => {
                self.assist_batch_rx = None;
                self.assist_batch_busy = false;
                self.train_log.push(format!("[辅助] 全局采纳中止：{e}"));
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.assist_batch_rx = None;
                self.assist_batch_busy = false;
            }
        }
    }

    fn current_label_path(&self) -> Option<PathBuf> {
        let path = self.image_paths.get(self.current_index)?;
        Some(label_txt_path_for_image(path))
    }

    fn active_class_label_draft(&self) -> String {
        self.classes
            .get(self.active_class_idx.min(self.classes.len().saturating_sub(1)))
            .cloned()
            .unwrap_or_default()
    }

    fn save_current_labels(&mut self) -> std::io::Result<()> {
        let Some(img) = &self.rgba else {
            return Ok(());
        };
        let (w, h) = img.dimensions();
        let Some(lbl) = self.current_label_path() else {
            return Ok(());
        };
        save_annotations(&lbl, &self.annotations, w, h)?;
        self.mark_annotated_strip_dirty();
        Ok(())
    }

    fn collect_all_label_snapshots(&self) -> Vec<(PathBuf, Option<String>)> {
        self.image_paths
            .iter()
            .map(|p| {
                let lbl = label_txt_path_for_image(p);
                let contents = fs::read_to_string(&lbl).ok();
                (lbl, contents)
            })
            .collect()
    }

    fn restore_all_label_snapshots(files: &[(PathBuf, Option<String>)]) {
        for (path, contents) in files {
            match contents {
                Some(s) => {
                    let _ = fs::write(path, s);
                }
                None => {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }

    fn push_undo(&mut self, scope: UndoScope) {
        if self.undo_suspend {
            return;
        }
        if self.undo_stack.len() >= UNDO_STACK_CAP {
            self.undo_stack.remove(0);
        }
        let (all_txt, class_log_undo) = match scope {
            UndoScope::Local => (None, None),
            UndoScope::DatasetLabels => {
                let cl_path = self.class_log_path();
                let class_log_undo = Some(if cl_path.is_file() {
                    Some(fs::read_to_string(&cl_path).unwrap_or_default())
                } else {
                    None
                });
                (Some(self.collect_all_label_snapshots()), class_log_undo)
            }
        };
        self.undo_stack.push(UndoSnapshot {
            annotations: self.annotations.clone(),
            classes: self.classes.clone(),
            class_colors: self.class_colors.clone(),
            active_class_idx: self.active_class_idx,
            selected: self.selected,
            draw_phase: self.draw_phase.clone(),
            draw_new_boxes_enabled: self.draw_new_boxes_enabled,
            scribble_kind: self.scribble_kind,
            scribble_open_start: self.scribble_open_start,
            scribble_closed_boxes: self.scribble_closed_boxes.clone(),
            pending_boxes_batch: self.pending_boxes_batch.clone(),
            pending_box: self.pending_box.clone(),
            show_label_window: self.show_label_window,
            label_draft: self.label_draft.clone(),
            label_edit_idx: self.label_edit_idx,
            drag: self.drag,
            all_txt,
            class_log_undo,
        });
    }

    fn apply_undo(&mut self) {
        let Some(snap) = self.undo_stack.pop() else {
            return;
        };
        self.undo_suspend = true;
        self.annotations = snap.annotations;
        self.classes = snap.classes;
        self.class_colors = snap.class_colors;
        self.active_class_idx = snap.active_class_idx;
        self.selected = snap.selected;
        self.draw_phase = snap.draw_phase;
        self.draw_new_boxes_enabled = snap.draw_new_boxes_enabled;
        self.scribble_kind = snap.scribble_kind;
        self.scribble_open_start = snap.scribble_open_start;
        self.scribble_closed_boxes = snap.scribble_closed_boxes;
        self.pending_boxes_batch = snap.pending_boxes_batch;
        self.scribble_active = false;
        self.scribble_points.clear();
        self.pending_box = snap.pending_box;
        self.show_label_window = snap.show_label_window;
        self.label_draft = snap.label_draft;
        self.label_edit_idx = snap.label_edit_idx;
        self.drag = snap.drag;
        if let Some(ref files) = snap.all_txt {
            Self::restore_all_label_snapshots(files);
        }
        if let Some(ref prev_class_log) = snap.class_log_undo {
            let p = self.class_log_path();
            match prev_class_log {
                None => {
                    let _ = fs::remove_file(&p);
                }
                Some(content) => {
                    let _ = fs::write(&p, content);
                }
            }
        }
        self.undo_suspend = false;
        if !self.classes.is_empty() {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        } else {
            self.active_class_idx = 0;
        }
        let _ = self.save_current_labels();
        self.mark_class_log_dirty();
        self.mark_annotated_strip_dirty();
    }

    fn any_duplicate_trimmed_class_names(&self) -> bool {
        let n = self.classes.len();
        for i in 0..n {
            let ti = self.classes[i].trim();
            for j in (i + 1)..n {
                if self.classes[j].trim() == ti {
                    return true;
                }
            }
        }
        false
    }

    /// 将 `remove` 合并进 `into`（`into` < `remove`）：标签 id 重映射并删除多余类别行。
    fn merge_class_remove_into(&mut self, into: usize, remove: usize) {
        if into >= self.classes.len() || remove >= self.classes.len() || into >= remove {
            return;
        }
        while self.class_colors.len() < self.classes.len() {
            let k = self.class_colors.len();
            self.class_colors.push(palette_color(k));
        }
        for img_path in self.image_paths.clone() {
            let Ok(img) = image::open(&img_path) else {
                continue;
            };
            let (w, h) = img.dimensions();
            let lbl = label_txt_path_for_image(&img_path);
            let mut ann = load_annotations(&lbl, w, h);
            for b in ann.iter_mut() {
                if b.class_id == remove {
                    b.class_id = into;
                } else if b.class_id > remove {
                    b.class_id -= 1;
                }
            }
            let _ = save_annotations(&lbl, &ann, w, h);
        }
        for b in self.annotations.iter_mut() {
            if b.class_id == remove {
                b.class_id = into;
            } else if b.class_id > remove {
                b.class_id -= 1;
            }
        }
        self.classes.remove(remove);
        if remove < self.class_colors.len() {
            self.class_colors.remove(remove);
        }
        self.class_colors.truncate(self.classes.len());
        match self.active_class_idx.cmp(&remove) {
            Ordering::Equal => self.active_class_idx = into,
            Ordering::Greater => self.active_class_idx -= 1,
            Ordering::Less => {}
        }
        if !self.classes.is_empty() {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        } else {
            self.active_class_idx = 0;
        }
    }

    /// 删除指定类别行，并移除全数据集中该类 id 的所有标注框；更大 id 减一。支持 Ctrl+Z 撤销。
    fn delete_class_and_boxes(&mut self, remove: usize) {
        if remove >= self.classes.len() {
            return;
        }
        self.push_undo(UndoScope::DatasetLabels);
        while self.class_colors.len() < self.classes.len() {
            let k = self.class_colors.len();
            self.class_colors.push(palette_color(k));
        }
        for img_path in self.image_paths.clone() {
            let Ok(img) = image::open(&img_path) else {
                continue;
            };
            let (w, h) = img.dimensions();
            let lbl = label_txt_path_for_image(&img_path);
            let ann = load_annotations(&lbl, w, h);
            let mut new_ann: Vec<Bbox> = Vec::new();
            for mut b in ann {
                if b.class_id == remove {
                    continue;
                }
                if b.class_id > remove {
                    b.class_id -= 1;
                }
                new_ann.push(b);
            }
            let _ = save_annotations(&lbl, &new_ann, w, h);
        }

        let mut new_cur: Vec<Bbox> = Vec::new();
        let mut new_sel: Option<usize> = None;
        for (idx, b) in self.annotations.iter().enumerate() {
            if b.class_id == remove {
                continue;
            }
            let mut b2 = b.clone();
            if b2.class_id > remove {
                b2.class_id -= 1;
            }
            let ni = new_cur.len();
            if self.selected == Some(idx) {
                new_sel = Some(ni);
            }
            new_cur.push(b2);
        }
        self.annotations = new_cur;
        self.selected = new_sel;

        self.classes.remove(remove);
        if remove < self.class_colors.len() {
            self.class_colors.remove(remove);
        }
        self.class_colors.truncate(self.classes.len());

        match self.active_class_idx.cmp(&remove) {
            Ordering::Equal => {
                self.active_class_idx = remove.saturating_sub(1);
            }
            Ordering::Greater => self.active_class_idx -= 1,
            Ordering::Less => {}
        }
        if self.classes.is_empty() {
            self.active_class_idx = 0;
        } else {
            self.active_class_idx = self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        }

        let _ = self.save_current_labels();
        self.mark_class_log_dirty();
        self.mark_annotated_strip_dirty();
    }

    /// 若存在去掉首尾空格后同名的类别，合并到最小索引；在类别名输入框有焦点时不调用，避免输入过程中误合并。
    fn merge_duplicate_class_names(&mut self) {
        if !self.any_duplicate_trimmed_class_names() {
            return;
        }
        self.push_undo(UndoScope::DatasetLabels);
        while self.any_duplicate_trimmed_class_names() {
            let n = self.classes.len();
            let mut pair: Option<(usize, usize)> = None;
            'outer: for i in 0..n {
                let ti = self.classes[i].trim();
                for j in (i + 1)..n {
                    if self.classes[j].trim() == ti {
                        pair = Some((i, j));
                        break 'outer;
                    }
                }
            }
            let Some((keep, remove)) = pair else {
                break;
            };
            self.merge_class_remove_into(keep, remove);
        }
        let _ = self.save_current_labels();
        self.mark_class_log_dirty();
        self.mark_annotated_strip_dirty();
    }

    /// 交换两个类别索引：名称对调，且全数据集中所有框的这两个 id 对调。
    fn swap_class_indices(&mut self, i: usize, j: usize) {
        if i >= self.classes.len() || j >= self.classes.len() || i == j {
            return;
        }
        self.push_undo(UndoScope::DatasetLabels);
        while self.class_colors.len() < self.classes.len() {
            let k = self.class_colors.len();
            self.class_colors.push(palette_color(k));
        }
        self.classes.swap(i, j);
        self.class_colors.swap(i, j);
        for img_path in self.image_paths.clone() {
            let Ok(img) = image::open(&img_path) else {
                continue;
            };
            let (w, h) = img.dimensions();
            let lbl = label_txt_path_for_image(&img_path);
            let mut ann = load_annotations(&lbl, w, h);
            for b in ann.iter_mut() {
                if b.class_id == i {
                    b.class_id = CLASS_SWAP_SENTINEL;
                } else if b.class_id == j {
                    b.class_id = i;
                }
            }
            for b in ann.iter_mut() {
                if b.class_id == CLASS_SWAP_SENTINEL {
                    b.class_id = j;
                }
            }
            let _ = save_annotations(&lbl, &ann, w, h);
        }
        for b in self.annotations.iter_mut() {
            if b.class_id == i {
                b.class_id = CLASS_SWAP_SENTINEL;
            } else if b.class_id == j {
                b.class_id = i;
            }
        }
        for b in self.annotations.iter_mut() {
            if b.class_id == CLASS_SWAP_SENTINEL {
                b.class_id = j;
            }
        }
        if self.active_class_idx == i {
            self.active_class_idx = j;
        } else if self.active_class_idx == j {
            self.active_class_idx = i;
        }
        let _ = self.save_current_labels();
        self.mark_class_log_dirty();
    }

    fn class_id_for_name(&mut self, name: &str) -> usize {
        let name = name.trim();
        if name.is_empty() {
            if self.classes.is_empty() {
                return 0;
            }
            return self
                .active_class_idx
                .min(self.classes.len().saturating_sub(1));
        }
        if let Some(i) = self.classes.iter().position(|c| c == name) {
            return i;
        }
        let k = self.classes.len();
        self.classes.push(name.to_string());
        self.class_colors.push(palette_color(k));
        self.mark_class_log_dirty();
        k
    }

    fn finalize_pending_with_label(&mut self) {
        let Some(img) = &self.rgba else {
            return;
        };
        let (w, h) = img.dimensions();
        let draft = self.label_draft.clone();

        if let Some(idx) = self.label_edit_idx.take() {
            if idx < self.annotations.len() {
                self.push_undo(UndoScope::Local);
                let cid = self.class_id_for_name(&draft);
                self.annotations[idx].class_id = cid;
                let _ = self.save_current_labels();
            }
            self.label_draft.clear();
            self.show_label_window = false;
            self.draw_phase = DrawPhase::Idle;
            return;
        }

        if !self.pending_boxes_batch.is_empty() {
            self.push_undo(UndoScope::Local);
            let cid = self.class_id_for_name(&draft);
            let mut any = false;
            for pb in self.pending_boxes_batch.drain(..) {
                let mut b = Bbox {
                    min_x: pb.min_x,
                    min_y: pb.min_y,
                    max_x: pb.max_x,
                    max_y: pb.max_y,
                    class_id: cid,
                };
                b.normalize(w, h);
                if (b.max_x - b.min_x) >= 2.0 && (b.max_y - b.min_y) >= 2.0 {
                    self.annotations.push(b);
                    any = true;
                }
            }
            if any {
                let _ = self.save_current_labels();
            }
            self.label_draft.clear();
            self.show_label_window = false;
            self.draw_phase = DrawPhase::Idle;
            return;
        }

        if self.pending_box.is_none() {
            return;
        }
        self.push_undo(UndoScope::Local);
        let Some(pb) = self.pending_box.take() else {
            return;
        };
        let cid = self.class_id_for_name(&draft);
        let mut b = Bbox {
            min_x: pb.min_x,
            min_y: pb.min_y,
            max_x: pb.max_x,
            max_y: pb.max_y,
            class_id: cid,
        };
        b.normalize(w, h);
        if (b.max_x - b.min_x) < 2.0 || (b.max_y - b.min_y) < 2.0 {
            self.show_label_window = false;
            return;
        }
        self.annotations.push(b);
        let _ = self.save_current_labels();
        self.label_draft.clear();
        self.show_label_window = false;
        self.draw_phase = DrawPhase::Idle;
    }

    fn corner_points(b: &Bbox) -> [Pos2; 4] {
        [
            Pos2::new(b.min_x, b.min_y),
            Pos2::new(b.max_x, b.min_y),
            Pos2::new(b.max_x, b.max_y),
            Pos2::new(b.min_x, b.max_y),
        ]
    }

    fn hit_corner_screen(
        p_screen: Pos2,
        b: &Bbox,
        disp: Rect,
        img_w: f32,
        img_h: f32,
        px_radius: f32,
    ) -> Option<usize> {
        for (i, c) in Self::corner_points(b).iter().enumerate() {
            let sp = image_to_screen(c.x, c.y, disp, img_w, img_h);
            if sp.distance(p_screen) <= px_radius {
                return Some(i);
            }
        }
        None
    }

    /// 距边最近且在 `edge_px` 内则返回该边索引；角附近已留白。
    fn hit_edge_screen(
        p_screen: Pos2,
        b: &Bbox,
        disp: Rect,
        img_w: f32,
        img_h: f32,
        edge_px: f32,
        corner_skip: f32,
    ) -> Option<u8> {
        let tl = image_to_screen(b.min_x, b.min_y, disp, img_w, img_h);
        let tr = image_to_screen(b.max_x, b.min_y, disp, img_w, img_h);
        let br = image_to_screen(b.max_x, b.max_y, disp, img_w, img_h);
        let bl = image_to_screen(b.min_x, b.max_y, disp, img_w, img_h);
        let s = corner_skip;
        let segs = [
            (tl + Vec2::new(s, 0.0), tr + Vec2::new(-s, 0.0)),
            (tr + Vec2::new(0.0, s), br + Vec2::new(0.0, -s)),
            (br + Vec2::new(-s, 0.0), bl + Vec2::new(s, 0.0)),
            (bl + Vec2::new(0.0, -s), tl + Vec2::new(0.0, s)),
        ];
        let mut best: Option<(u8, f32)> = None;
        for (i, &(va, vb)) in segs.iter().enumerate() {
            let d = dist_point_segment_2d(p_screen, va, vb);
            if d <= edge_px {
                let better = best.map_or(true, |(_, bd)| d < bd - 1e-3);
                if better {
                    best = Some((i as u8, d));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    fn hit_inside(p_img: (f32, f32), b: &Bbox) -> bool {
        let (x, y) = p_img;
        x >= b.min_x && x <= b.max_x && y >= b.min_y && y <= b.max_y
    }

    /// F 模式：最新边与更早某条非相邻边相交则形成视觉闭合块；嵌套时只保留最大圈（见 `polygon_contains_polygon`）。
    fn try_close_scribble_loop(&mut self, img_w: u32, img_h: u32) {
        if !matches!(
            self.scribble_kind,
            Some(ScribbleKind::ContinuousCircumscribed)
        ) {
            return;
        }
        let n = self.scribble_points.len();
        // 至少 4 个点：新边 (P[n-2],P[n-1]) 与最早可测边 (P[0],P[1]) 无公共点。
        if n < 4 {
            return;
        }
        let wf = img_w as f32;
        let hf = img_h as f32;
        let a = self.scribble_points[n - 2];
        let b = self.scribble_points[n - 1];
        if (b.0 - a.0).abs() + (b.1 - a.1).abs() < 1e-6 {
            return;
        }
        let mut found: Option<(usize, (f32, f32))> = None;
        // 从靠近笔尖的一侧找交点：取最大的 j，使新边与 (P[j],P[j+1]) 相交，对应最小的新成环。
        for j in (0..=n.saturating_sub(4)).rev() {
            let c = self.scribble_points[j];
            let d = self.scribble_points[j + 1];
            if (d.0 - c.0).abs() + (d.1 - c.1).abs() < 1e-6 {
                continue;
            }
            if let Some(ipt) = segment_segment_intersection(a, b, c, d) {
                found = Some((j, ipt));
                break;
            }
        }
        let Some((j, ipt)) = found else {
            return;
        };
        let mut new_poly: Vec<(f32, f32)> = self.scribble_points[j + 1..n].to_vec();
        new_poly.push(ipt);
        if new_poly.len() < 3 {
            return;
        }
        let (min_x, min_y, max_x, max_y) = circumscribed_aabb_for_scribble(&new_poly, wf, hf);
        let mut pb = PendingBox {
            min_x,
            min_y,
            max_x,
            max_y,
        };
        if pb.min_x > pb.max_x {
            std::mem::swap(&mut pb.min_x, &mut pb.max_x);
        }
        if pb.min_y > pb.max_y {
            std::mem::swap(&mut pb.min_y, &mut pb.max_y);
        }
        let candidate = ScribbleClosedBlock {
            poly: new_poly,
            aabb: pb,
        };
        // 新圈为外圈：删掉已被完全包在里面的旧块。
        self.scribble_closed_boxes.retain(|b| {
            !polygon_contains_polygon(&candidate.poly, &b.poly)
        });
        prune_scribble_closed_blocks_overlap_keep_larger(&mut self.scribble_closed_boxes);
        // 新圈为内圈（子集）：不加入。
        if self.scribble_closed_boxes.iter().any(|b| {
            polygon_contains_polygon(&b.poly, &candidate.poly)
        }) {
            self.scribble_open_start = n - 1;
            return;
        }
        self.scribble_closed_boxes.push(candidate);
        prune_scribble_closed_blocks_overlap_keep_larger(&mut self.scribble_closed_boxes);
        self.scribble_open_start = n - 1;
    }

    fn open_label_for_new_boxes(&mut self) {
        self.label_edit_idx = None;
        self.show_label_window = true;
        self.label_draft = self.active_class_label_draft();
    }

    /// 柔性标注：E 为整笔外接；F 为连续外接（仅闭合块入批）。
    fn finalize_scribble_to_pending(&mut self, img_w: u32, img_h: u32) {
        if matches!(
            self.scribble_kind,
            Some(ScribbleKind::ContinuousCircumscribed)
        ) {
            self.scribble_points.clear();
            self.scribble_open_start = 0;
            self.scribble_active = false;
            if self.scribble_closed_boxes.is_empty() {
                self.train_log.push(
                    "[提示] 连续柔性外接：本笔未检测到自相交闭合（新画线段需与更早线段交叉成环），未生成框。"
                        .to_string(),
                );
                return;
            }
            let mut blocks = std::mem::take(&mut self.scribble_closed_boxes);
            prune_scribble_closed_blocks_overlap_keep_larger(&mut blocks);
            self.pending_boxes_batch = blocks.into_iter().map(|b| b.aabb).collect();
            self.pending_box = None;
            self.open_label_for_new_boxes();
            return;
        }

        if self.scribble_points.is_empty() {
            return;
        }
        let wf = img_w as f32;
        let hf = img_h as f32;
        let pts: Vec<(f32, f32)> = self
            .scribble_points
            .iter()
            .map(|&(x, y)| (x.clamp(0.0, wf), y.clamp(0.0, hf)))
            .collect();
        self.scribble_points.clear();

        let (min_x, min_y, max_x, max_y) = circumscribed_aabb_for_scribble(&pts, wf, hf);

        let mut pb = PendingBox {
            min_x,
            min_y,
            max_x,
            max_y,
        };
        if pb.min_x > pb.max_x {
            std::mem::swap(&mut pb.min_x, &mut pb.max_x);
        }
        if pb.min_y > pb.max_y {
            std::mem::swap(&mut pb.min_y, &mut pb.max_y);
        }
        self.pending_box = Some(pb);
        self.open_label_for_new_boxes();
    }

    fn start_training(&mut self) {
        if self.training {
            return;
        }
        if self.classes.is_empty() {
            self.train_log.push(
                "[错误] 当前没有任何类别。请先完成至少一次标注并保存，或检查已有 txt 中的类别编号。"
                    .to_string(),
            );
            return;
        }
        if self.image_paths.is_empty() {
            self.train_log.push("[错误] 没有可训练的图片，请先选择图片目录并刷新。".to_string());
            return;
        }
        let backend = if self.use_builtin_cpu_train {
            TrainingBackend::BuiltinCpu
        } else {
            TrainingBackend::Conda
        };
        let mut conda_py: Option<PathBuf> = None;
        let mut builtin_train_exe: Option<PathBuf> = None;
        let Some(export_exe) = find_builtin_onnx_export_exe() else {
            self.train_log.push(
                "[错误] 未找到自带 ONNX 转码器。请先在程序目录下准备 embedded_tools\\onnx_export_runner\\onnx_export_runner.exe".to_string(),
            );
            return;
        };
        match backend {
            TrainingBackend::Conda => {
                let root = self.selected_conda_root();
                if root.is_empty() {
                    self.train_log.push(
                        "[错误] 请先在「训练环境设定」中选择 Conda 环境（或点击刷新环境列表）。"
                            .to_string(),
                    );
                    return;
                }
                let py = conda_python_executable(Path::new(root));
                if !py.is_file() {
                    self.train_log.push(format!(
                        "[错误] 未找到 Python: {} （请确认所选为 Conda 环境根目录，内含 python.exe）",
                        py.display()
                    ));
                    return;
                }
                conda_py = Some(py);
            }
            TrainingBackend::BuiltinCpu => {
                let Some(train_exe) = find_builtin_cpu_train_exe() else {
                    self.train_log.push(
                        "[错误] 未找到自带 CPU 训练器。请先在程序目录下准备 embedded_tools\\cpu_train_runner\\cpu_train_runner.exe".to_string(),
                    );
                    return;
                };
                builtin_train_exe = Some(train_exe);
            }
        }

        self.training = true;
        self.training_stop_pending = false;
        self.training_pid = None;
        self.train_log.push(match backend {
            TrainingBackend::Conda => "[准备] 正在打包 train_<时间戳>/（图片、标签、所选单个 .pt；缺失权重仅下载到该目录，不污染图片根目录）…".to_string(),
            TrainingBackend::BuiltinCpu => "[准备] 正在打包 train_<时间戳>/，随后使用自带 CPU 训练器并在完成后自动转码 ONNX…".to_string(),
        });

        let (tx_msg, rx_msg) = mpsc::channel::<TrainMsg>();
        let image_root = self.image_root.clone();
        let image_paths = self.image_paths.clone();
        let classes = self.classes.clone();
        let model_preset = self.model_preset;
        let train_epochs = self.train_epochs;
        let model_name = model_preset.filename().to_string();
        let export_exe = export_exe;

        thread::spawn(move || {
            let tx = tx_msg;
            let bundle = match prepare_training_bundle_in_dir(
                &image_root,
                &image_paths,
                &classes,
                model_preset,
                train_epochs,
                &tx,
            ) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(TrainMsg::Line(format!("[错误] {e}")));
                    let _ = tx.send(TrainMsg::Done(-1));
                    return;
                }
            };
            let _ = tx.send(TrainMsg::Line(format!(
                "已在图片目录下创建训练包: {}",
                bundle.display()
            )));
            let bundle_cwd = path_abs_for_ospawn(&bundle);
            let train_code = match backend {
                TrainingBackend::Conda => {
                    let script = bundle.join("ultralytics_train_run.py");
                    let script_abs = path_abs_for_ospawn(&script);
                    let py_exe = path_abs_for_ospawn(conda_py.as_ref().expect("missing conda py"));
                    let _ = tx.send(TrainMsg::Line(format!("启动训练: {}", script.display())));
                    let mut c = Command::new(&py_exe);
                    c.current_dir(&bundle_cwd)
                        .env("PYTHONUNBUFFERED", "1")
                        .arg(&script_abs)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped());
                    command_hide_console(&mut c);
                    match spawn_logged_child(c, &tx) {
                        Ok(code) => code,
                        Err(e) => {
                            let _ = tx.send(TrainMsg::Line(format!("[错误] {e}")));
                            -1
                        }
                    }
                }
                TrainingBackend::BuiltinCpu => {
                    let train_exe = path_abs_for_ospawn(
                        builtin_train_exe
                            .as_ref()
                            .expect("missing builtin cpu train exe"),
                    );
                    let _ = tx.send(TrainMsg::Line(format!(
                        "[训练] 使用自带 CPU 训练器: {}",
                        train_exe.display()
                    )));
                    let mut c = Command::new(&train_exe);
                    c.current_dir(&bundle_cwd)
                        .env("PYTHONUNBUFFERED", "1")
                        .args(["--bundle", bundle_cwd.to_string_lossy().as_ref()])
                        .args(["--model", &model_name])
                        .args(["--epochs", &train_epochs.to_string()])
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped());
                    command_hide_console(&mut c);
                    match spawn_logged_child(c, &tx) {
                        Ok(code) => code,
                        Err(e) => {
                            let _ = tx.send(TrainMsg::Line(format!("[错误] {e}")));
                            -1
                        }
                    }
                }
            };
            let final_code = if train_code != 0 {
                train_code
            } else {
                match find_latest_best_pt_in_bundle(&bundle) {
                    Some(best_pt) => {
                        let export_exe = path_abs_for_ospawn(&export_exe);
                        let dest_onnx = bundle.join("best.onnx");
                        let best_abs = path_abs_for_ospawn(&best_pt);
                        let dest_abs = path_abs_for_ospawn(&dest_onnx);
                        let _ = tx.send(TrainMsg::Line(format!(
                            "[导出] 使用自带 ONNX 转码器: {}",
                            export_exe.display()
                        )));
                        let mut export_cmd = Command::new(&export_exe);
                        export_cmd
                            .current_dir(&bundle_cwd)
                            .env("PYTHONUNBUFFERED", "1")
                            .args(["--weights", best_abs.to_string_lossy().as_ref()])
                            .args(["--dest", dest_abs.to_string_lossy().as_ref()])
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped());
                        command_hide_console(&mut export_cmd);
                        match spawn_logged_child(export_cmd, &tx) {
                            Ok(code) => code,
                            Err(e) => {
                                let _ = tx.send(TrainMsg::Line(format!("[错误] {e}")));
                                -3
                            }
                        }
                    }
                    None => {
                        let _ = tx.send(TrainMsg::Line(
                            "[导出] 未找到 runs/detect/*/weights/best.pt，自带 ONNX 转码已跳过".to_string(),
                        ));
                        -2
                    }
                }
            };
            let _ = tx.send(TrainMsg::Done(final_code));
        });

        self.train_rx = Some(rx_msg);
    }

    fn stop_training(&mut self) {
        self.training_stop_pending = true;
        if let Some(pid) = self.training_pid.take() {
            kill_training_process_tree(pid);
        }
        self.train_log.push("[训练] 已请求停止。".to_string());
    }

    fn poll_training(&mut self, ctx: &Context) {
        let Some(rx) = self.train_rx.as_ref() else {
            return;
        };
        let mut drained = false;
        let mut exit_code: Option<i32> = None;
        while let Ok(msg) = rx.try_recv() {
            drained = true;
            match msg {
                TrainMsg::Line(s) => self.train_log.push(s),
                TrainMsg::ChildStarted { pid } => {
                    if pid != 0 {
                        self.training_pid = Some(pid);
                        if self.training_stop_pending {
                            kill_training_process_tree(pid);
                            self.training_pid = None;
                        }
                    }
                }
                TrainMsg::Done(code) => exit_code = Some(code),
            }
        }
        if let Some(code) = exit_code {
            self.training = false;
            self.train_rx = None;
            self.training_pid = None;
            self.training_stop_pending = false;
            self.train_log.push(format!("进程结束，代码: {code}"));
        }
        if drained || self.training {
            ctx.request_repaint();
        }
    }

    /// 已写入标签的图片列表（相对根目录路径，点击切换当前图）。
    fn ui_annotated_strip(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("已标注图片 {}", self.annotated_strip_indices.len()))
                    .small()
                    .strong()
                    .color(theme::TEXT),
            );
            ui.label(
                RichText::new("点击文件名可快速切换，✓ 表示当前图片")
                    .small()
                    .color(theme::TEXT_MUTED),
            );
        });
        ui.add_space(6.0);
        app_card(theme::SURFACE_SOFT).show(ui, |ui| {
            if self.annotated_strip_indices.is_empty() {
                ui.label(
                    RichText::new("暂无已保存标签的图片，保存 .txt 后会在这里出现。")
                        .small()
                        .color(theme::TEXT_MUTED),
                );
            } else {
                let strip_idxs = self.annotated_strip_indices.clone();
                let root = self.image_root.clone();
                egui::ScrollArea::vertical()
                    .id_salt("annotated_strip_scroll")
                    .max_height(200.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for strip_idx in strip_idxs {
                            let Some(path) = self.image_paths.get(strip_idx).cloned() else {
                                continue;
                            };
                            let is_cur = strip_idx == self.current_index;
                            let rel = path_relative_to(&path, &root)
                                .to_string_lossy()
                                .to_string();
                            let full_path = path.display().to_string();
                            let fill = if is_cur {
                                color_alpha(theme::ACCENT, 32)
                            } else {
                                Color32::TRANSPARENT
                            };
                            Frame::default()
                                .fill(fill)
                                .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                                .rounding(egui::Rounding::same(10.0))
                                .stroke(Stroke::new(
                                    1.0,
                                    if is_cur {
                                        color_alpha(theme::ACCENT, 110)
                                    } else {
                                        Color32::TRANSPARENT
                                    },
                                ))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.add_sized(
                                            Vec2::new(20.0, ui.spacing().interact_size.y),
                                            Label::new(if is_cur {
                                                RichText::new("✓").strong().color(theme::OK)
                                            } else {
                                                RichText::new("·").color(theme::TEXT_MUTED)
                                            }),
                                        );
                                        let sel = ui.selectable_label(
                                            is_cur,
                                            RichText::new(&rel)
                                                .small()
                                                .monospace()
                                                .color(if is_cur { theme::TEXT } else { theme::TEXT_MUTED }),
                                        );
                                        if sel.clicked() {
                                            self.go_to_image_index(strip_idx);
                                        }
                                        sel.on_hover_text(&full_path);
                                    });
                                });
                            ui.add_space(4.0);
                        }
                    });
            }
        });
    }

    fn ui_sidebar(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            self.sidebar_width = ui.available_width();
            let footer_h = 236.0_f32;
            // 「图像导航」在侧栏顶部时占位高于原标题行
            let header_h = 148.0_f32;
            let ah = ui.available_height();
            let scroll_h = if ah.is_finite() && ah > footer_h + header_h + 80.0 {
                ah - footer_h - header_h
            } else {
                332.0_f32
            };
            let mut open_section = self.sidebar_open_section;

            let total_images = self.image_paths.len();
            let current_idx = if total_images == 0 {
                0
            } else {
                self.current_index + 1
            };
            let (cur_disp_name, cur_hover_path) = self
                .image_paths
                .get(self.current_index)
                .map(|p| {
                    let name = p
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| p.to_string_lossy().into_owned());
                    (name, p.display().to_string())
                })
                .unwrap_or_else(|| ("（未加载）".to_string(), String::new()));

            app_card(theme::SURFACE_ELEVATED).show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("图像导航")
                                .small()
                                .strong()
                                .color(theme::TEXT),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if self.training {
                                status_chip(ui, "训练中", theme::WARN);
                            } else {
                                status_chip(ui, "待机", theme::OK);
                            }
                        });
                    });
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .add_enabled(self.current_index > 0, egui::Button::new("上一张"))
                            .clicked()
                        {
                            self.go_prev_image();
                        }
                        if ui
                            .add_enabled(
                                !self.image_paths.is_empty()
                                    && self.current_index + 1 < self.image_paths.len(),
                                egui::Button::new("下一张"),
                            )
                            .clicked()
                        {
                            self.go_next_image();
                        }
                        if ui
                            .button("刷新列表")
                            .on_hover_text("重新扫描目录，并刷新左侧已标注列表")
                            .clicked()
                        {
                            self.refresh_image_list();
                        }
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(&cur_disp_name)
                                .small()
                                .monospace()
                                .color(theme::TEXT),
                        )
                        .on_hover_text(if cur_hover_path.is_empty() {
                            "当前无图片".to_string()
                        } else {
                            cur_hover_path.clone()
                        });
                    });
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        status_chip(ui, &format!("进度 {current_idx}/{total_images}"), theme::ACCENT);
                        status_chip(ui, &format!("本图 {} 框", self.annotations.len()), theme::OK);
                        status_chip(
                            ui,
                            &format!("已标注 {}", self.annotated_strip_indices.len()),
                            theme::WARN,
                        );
                        status_chip(ui, &format!("类别 {}", self.classes.len()), theme::ACCENT);
                    });
                });
            });
            ui.add_space(10.0);

            egui::ScrollArea::vertical()
                .id_salt("yolo_left_sidebar_scroll")
                .max_height(scroll_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let _ = section_accordion(
                        ui,
                        1,
                        &mut open_section,
                        "01",
                        "数据集",
                        "先选工作目录，再浏览图片与已标注结果。",
                        Color32::from_rgb(24, 34, 44),
                        theme::ACCENT,
                        |ui| {
                            ui.label(
                                RichText::new("载入所选目录内一层图片；若该层无图但存在 data.yaml，则按其中 train/val 子目录加载一层图片。标签支持同目录 .txt 或 YOLO 的 images↔labels 目录。")
                                    .small()
                                    .color(theme::TEXT_MUTED),
                            );
                            ui.add_space(8.0);
                            if ui
                                .add_sized(
                                    Vec2::new(ui.available_width(), 40.0),
                                    egui::Button::new(
                                        RichText::new("选择图片目录…")
                                            .strong()
                                            .color(theme::TEXT),
                                    )
                                    .fill(Color32::from_rgb(68, 118, 88))
                                    .stroke(Stroke::new(
                                        1.0,
                                        Color32::from_rgb(108, 178, 132),
                                    )),
                                )
                                .clicked()
                            {
                                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                    self.switch_image_root(p);
                                }
                            }

                            ui.add_space(10.0);
                            ui.columns(3, |columns| {
                                metric_card(
                                    &mut columns[0],
                                    "图片",
                                    self.image_paths.len().to_string(),
                                    "当前工作集",
                                    theme::ACCENT,
                                );
                                metric_card(
                                    &mut columns[1],
                                    "已标注",
                                    self.annotated_strip_indices.len().to_string(),
                                    "可快速跳转",
                                    theme::OK,
                                );
                                metric_card(
                                    &mut columns[2],
                                    "类别",
                                    self.classes.len().to_string(),
                                    "class_log 同步",
                                    theme::WARN,
                                );
                            });

                            ui.add_space(10.0);
                            ui.label(
                                RichText::new("当前根目录")
                                    .small()
                                    .strong()
                                    .color(theme::TEXT),
                            );
                            ui.add_space(4.0);
                            {
                                let root_s = self.image_root.to_string_lossy().to_string();
                                let pw = ui.available_width();
                                let row_h = ui.fonts(|f| f.row_height(&egui::FontId::monospace(11.0))) + 10.0;
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    ui.set_width(pw);
                                    egui::ScrollArea::horizontal()
                                        .id_salt("sidebar_image_root_path")
                                        .max_height(row_h)
                                        .show(ui, |ui| {
                                            ui.label(
                                                RichText::new(root_s)
                                                    .monospace()
                                                    .size(11.0)
                                                    .color(theme::TEXT_MUTED),
                                            );
                                        });
                                });
                            }
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new(format!(
                                    "类别顺序保存在根目录「{}」中，切换目录后会自动读取并同步。",
                                    CLASS_LOG_FILENAME
                                ))
                                .small()
                                .color(theme::TEXT_MUTED),
                            );
                            ui.add_space(12.0);
                            self.ui_annotated_strip(ui);
                        },
                    );

                    ui.add_space(12.0);

                    let _ = section_accordion(
                        ui,
                        2,
                        &mut open_section,
                        "02",
                        "类别管理",
                        "在这里维护默认类别、排序、颜色和清理操作。",
                        Color32::from_rgb(30, 38, 32),
                        theme::WARN,
                        |ui| {
                            ui.horizontal_wrapped(|ui| {
                                status_chip(ui, "数字单选 = 新框默认类", theme::ACCENT);
                                status_chip(ui, "↑↓ 调整索引顺序", theme::WARN);
                                status_chip(ui, "× 删除整类并可撤销", theme::DANGER);
                            });
                            ui.add_space(8.0);

                            if self.classes.is_empty() {
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "暂无类别。可以在新建标注框时命名，或直接编辑根目录「{}」新增非注释行。",
                                            CLASS_LOG_FILENAME
                                        ))
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                    );
                                });
                            } else {
                                let active_name = self
                                    .classes
                                    .get(self.active_class_idx.min(self.classes.len().saturating_sub(1)))
                                    .cloned()
                                    .unwrap_or_else(|| "未选择".to_string());
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new("默认类别")
                                                .small()
                                                .strong()
                                                .color(theme::TEXT),
                                        );
                                        ui.label(
                                            RichText::new(active_name)
                                                .small()
                                                .color(theme::ACCENT),
                                        );
                                    });
                                });
                                ui.add_space(8.0);

                                while self.class_colors.len() < self.classes.len() {
                                    let k = self.class_colors.len();
                                    self.class_colors.push(palette_color(k));
                                }
                                self.class_colors.truncate(self.classes.len());
                                let release_lbl = self.show_label_window;

                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    let row_h = ui.spacing().interact_size.y + 18.0;
                                    let row_gap = 4.0_f32;
                                    let visible_rows = 3.0_f32;
                                    egui::ScrollArea::vertical()
                                        .id_salt("class_editor_scroll")
                                        .max_height(
                                            row_h * visible_rows
                                                + row_gap * (visible_rows - 1.0)
                                                + 6.0,
                                        )
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            let mut row = 0usize;
                                            while row < self.classes.len() {
                                                let n_cls = self.classes.len();
                                                let deleted = Frame::default()
                                                    .fill(color_alpha(theme::SURFACE, 180))
                                                    .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                                                    .rounding(egui::Rounding::same(10.0))
                                                    .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
                                                    .show(ui, |ui| {
                                                        ui.horizontal(|ui| {
                                                            class_color_pick_button(
                                                                ui,
                                                                row,
                                                                &mut self.class_colors[row],
                                                                release_lbl,
                                                                ui.spacing().interact_size,
                                                            );
                                                            let radio_resp = ui.radio_value(
                                                                &mut self.active_class_idx,
                                                                row,
                                                                RichText::new(format!("{row}")).monospace(),
                                                            );
                                                            if release_lbl && radio_resp.clicked() {
                                                                ui.memory_mut(|m| {
                                                                    m.surrender_focus(label_draft_textedit_id())
                                                                });
                                                            }
                                                            let te = egui::TextEdit::singleline(&mut self.classes[row])
                                                                .desired_width(100.0)
                                                                .id(class_name_edit_id(row));
                                                            if ui.add(te).changed() {
                                                                self.mark_class_log_dirty();
                                                            }
                                                            if ui
                                                                .add_enabled(row > 0, egui::Button::new("↑"))
                                                                .on_hover_text("与上一项交换索引")
                                                                .clicked()
                                                            {
                                                                if release_lbl {
                                                                    ui.memory_mut(|m| {
                                                                        m.surrender_focus(label_draft_textedit_id())
                                                                    });
                                                                }
                                                                self.swap_class_indices(row, row - 1);
                                                            }
                                                            if ui
                                                                .add_enabled(row + 1 < n_cls, egui::Button::new("↓"))
                                                                .on_hover_text("与下一项交换索引")
                                                                .clicked()
                                                            {
                                                                if release_lbl {
                                                                    ui.memory_mut(|m| {
                                                                        m.surrender_focus(label_draft_textedit_id())
                                                                    });
                                                                }
                                                                self.swap_class_indices(row, row + 1);
                                                            }
                                                            if ui
                                                                .add(egui::Button::new("×").small())
                                                                .on_hover_text(
                                                                    "删除此类别，并移除整个数据集中该类的所有标注框（可 Ctrl+Z 撤销）",
                                                                )
                                                                .clicked()
                                                            {
                                                                if release_lbl {
                                                                    ui.memory_mut(|m| {
                                                                        m.surrender_focus(label_draft_textedit_id())
                                                                    });
                                                                }
                                                                self.delete_class_and_boxes(row);
                                                                true
                                                            } else {
                                                                false
                                                            }
                                                        })
                                                        .inner
                                                    })
                                                    .inner;
                                                if !deleted {
                                                    row += 1;
                                                }
                                                ui.add_space(row_gap);
                                            }
                                        });
                                });

                                let ctx = ui.ctx();
                                let any_name_focused = (0..self.classes.len())
                                    .any(|i| ctx.memory(|m| m.has_focus(class_name_edit_id(i))));
                                if !any_name_focused {
                                    self.merge_duplicate_class_names();
                                }
                            }
                        },
                    );

                    ui.add_space(12.0);

                    let _ = section_accordion(
                        ui,
                        3,
                        &mut open_section,
                        "03",
                        "训练配置",
                        "选择运行环境和基础权重，训练时会自动生成 train_* 包。",
                        Color32::from_rgb(42, 30, 36),
                        theme::DANGER,
                        |ui| {
                            app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                ui.checkbox(
                                    &mut self.use_builtin_cpu_train,
                                    "使用自带 CPU 训练（训练完成后自动转 ONNX）",
                                );
                                ui.label(
                                    RichText::new(
                                        "勾选后会直接调用程序目录里的 PyInstaller 封装训练器与 ONNX 转码器，不再依赖下方 Conda 环境。",
                                    )
                                    .small()
                                    .color(theme::TEXT_MUTED),
                                );
                            });

                            ui.add_space(10.0);
                            if self.use_builtin_cpu_train {
                                let train_ready = find_builtin_cpu_train_exe().is_some();
                                let export_ready = find_builtin_onnx_export_exe().is_some();
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        status_chip(
                                            ui,
                                            if train_ready {
                                                "CPU 训练器已就绪"
                                            } else {
                                                "CPU 训练器缺失"
                                            },
                                            if train_ready { theme::OK } else { theme::DANGER },
                                        );
                                        status_chip(
                                            ui,
                                            if export_ready {
                                                "ONNX 转码器已就绪"
                                            } else {
                                                "ONNX 转码器缺失"
                                            },
                                            if export_ready { theme::OK } else { theme::DANGER },
                                        );
                                    });
                                    ui.add_space(6.0);
                                    ui.label(
                                        RichText::new(
                                            "期望位置：embedded_tools\\cpu_train_runner\\cpu_train_runner.exe 与 embedded_tools\\onnx_export_runner\\onnx_export_runner.exe",
                                        )
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                    );
                                });
                            } else {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new("Conda 环境")
                                            .small()
                                            .strong()
                                            .color(theme::TEXT),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("刷新环境列表").clicked() {
                                                self.refresh_conda_env_list();
                                            }
                                        },
                                    );
                                });

                                if self.conda_env_paths.is_empty() {
                                    app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                        ui.label(
                                            RichText::new(
                                                "未检测到 Conda。请安装 Anaconda/Miniconda，并确保 conda 已加入 PATH 后再刷新。",
                                            )
                                            .small()
                                            .color(theme::TEXT_MUTED),
                                        );
                                    });
                                } else {
                                    self.conda_env_idx = self
                                        .conda_env_idx
                                        .min(self.conda_env_paths.len().saturating_sub(1));
                                    let n = self.conda_env_paths.len();
                                    ComboBox::from_id_salt("conda_env_pick")
                                        .width(ui.available_width())
                                        .selected_text(self.conda_env_paths[self.conda_env_idx].as_str())
                                        .show_index(ui, &mut self.conda_env_idx, n, |i| {
                                            self.conda_env_paths[i].as_str()
                                        });
                                }
                            }

                            ui.add_space(12.0);
                            ui.label(
                                RichText::new("预训练权重")
                                    .small()
                                    .strong()
                                    .color(theme::TEXT),
                            );
                            ui.add_space(6.0);
                            app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                ui.radio_value(
                                    &mut self.model_preset,
                                    ModelPreset::Yolo11n,
                                    "YOLO11n（更轻，更快启动）",
                                );
                                ui.radio_value(
                                    &mut self.model_preset,
                                    ModelPreset::Yolo11s,
                                    "YOLO11s（更强一些，训练更重）",
                                );
                            });

                            ui.add_space(8.0);
                            ui.label(
                                RichText::new(
                                    "查找顺序：程序目录 → exe 同目录 → 图片根目录。若本地没有，则只下载到本次训练包文件夹。",
                                )
                                .small()
                                .color(theme::TEXT_MUTED),
                            );
                            ui.label(
                                RichText::new("开始训练按钮支持滚轮调节 epochs；按住 Shift 可更快增减。")
                                    .small()
                                    .color(theme::TEXT_MUTED),
                            );
                        },
                    );
                });
            self.sidebar_open_section = open_section;

            ui.add_space(12.0);
            app_card(Color32::from_rgb(35, 35, 42)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("训练启动")
                            .strong()
                            .size(17.0)
                            .color(theme::TEXT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.training {
                            status_chip(ui, "训练中", theme::WARN);
                        } else {
                            status_chip(ui, "待启动", theme::OK);
                        }
                    });
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("epochs").small().strong().color(theme::TEXT));
                    ui.add(
                        egui::DragValue::new(&mut self.train_epochs)
                            .range(1..=100_000)
                            .speed(1.0),
                    );
                    ui.label(RichText::new("Shift + 滚轮可快调").small().color(theme::TEXT_MUTED));
                });
                ui.add_space(8.0);

                if self.training {
                    let stop = egui::Button::new(RichText::new("停止训练").strong().size(15.0))
                        .fill(Color32::from_rgb(65, 97, 136))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(112, 152, 201)))
                        .min_size(Vec2::new(ui.available_width(), 44.0));
                    if ui.add(stop).clicked() {
                        self.stop_training();
                    }
                } else {
                    let btn_text = format!("开始训练 · {} epochs", self.train_epochs);
                    let train_resp = ui.add(
                        egui::Button::new(RichText::new(btn_text).strong().size(15.0))
                            .fill(Color32::from_rgb(126, 66, 76))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(179, 102, 115)))
                            .min_size(Vec2::new(ui.available_width(), 44.0)),
                    );
                    if train_resp.hovered() {
                        let sy = ui.ctx().input(|i| {
                            let r = i.raw_scroll_delta.y;
                            if r.abs() > 0.5 {
                                r
                            } else {
                                i.smooth_scroll_delta.y
                            }
                        });
                        if sy.abs() > f32::EPSILON {
                            self.train_epoch_scroll_accum += sy;
                            const NOTCH: f32 = 120.0;
                            let step = if ui.input(|i| i.modifiers.shift) {
                                50u32
                            } else {
                                10u32
                            };
                            while self.train_epoch_scroll_accum >= NOTCH {
                                self.train_epochs = self.train_epochs.saturating_add(step).min(100_000);
                                self.train_epoch_scroll_accum -= NOTCH;
                            }
                            while self.train_epoch_scroll_accum <= -NOTCH {
                                self.train_epochs = (self.train_epochs.saturating_sub(step)).max(1);
                                self.train_epoch_scroll_accum += NOTCH;
                            }
                        }
                    } else {
                        self.train_epoch_scroll_accum = 0.0;
                    }
                    if train_resp.clicked() {
                        self.start_training();
                    }
                }

                if self.training {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new("训练进程正在运行，日志会持续写到下方面板。").color(theme::ACCENT));
                    });
                }
            });
        });
    }

    fn ui_canvas(&mut self, ui: &mut Ui, ctx: &Context) {
        let assist_batch = self.assist_batch_busy;
        let assist_infer = self.assist_busy && !assist_batch;
        let assist_row2_blocked = assist_batch || assist_infer;

        Frame::default()
            .fill(theme::SURFACE_ELEVATED)
            .inner_margin(egui::Margin::symmetric(10.0, 7.0))
            .rounding(egui::Rounding::same(8.0))
            .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(8.0, 4.0);
                    // 第一行：模型与状态
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new("辅助标注")
                                .small()
                                .strong()
                                .color(theme::TEXT_MUTED),
                        );
                        ui.separator();

                        if ui
                            .button("载入 ONNX…")
                            .on_hover_text(
                                "选择 Ultralytics 导出的 .onnx（推荐 nms=False, batch=1；也兼容 nms=True）；类别显示为 unknown_0…，推理后按出现的 id 展开筛选",
                            )
                            .clicked()
                        {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("ONNX", &["onnx"])
                                .pick_file()
                            {
                                let onnx_abs = path_abs_for_ospawn(&p);
                                match onnx_assist::load_session(&onnx_abs) {
                                    Ok(session) => {
                                        self.assist_onnx_path = Some(p.clone());
                                        self.assist_ort = Some(Arc::new(Mutex::new(session)));
                                        self.assist_class_names.clear();
                                        self.assist_pred_class_on.clear();
                                        self.assist_overlay_visible = true;
                                        self.train_log.push(format!(
                                            "[辅助] 已载入 ONNX {}（类别名 unknown_0…，首次推理后按 id 展开）",
                                            p.display()
                                        ));
                                        self.schedule_assist_infer();
                                        ctx.request_repaint();
                                    }
                                    Err(e) => self.train_log.push(format!("[辅助] {e}")),
                                }
                            }
                        }

                        if self.assist_ort.is_some() {
                            if ui
                                .small_button("移除模型")
                                .on_hover_text("卸载 ONNX、清空预测与进行中的任务")
                                .clicked()
                            {
                                self.assist_onnx_path = None;
                                self.assist_ort = None;
                                self.assist_class_names.clear();
                                self.assist_pred_class_on.clear();
                                self.assist_preds.clear();
                                let _ = self.assist_rx.take();
                                self.assist_busy = false;
                                let _ = self.assist_batch_rx.take();
                                self.assist_batch_busy = false;
                                self.assist_overlay_visible = true;
                            }
                        }

                        if let Some(ref p) = self.assist_onnx_path {
                            let short = p
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("?");
                            ui.label(RichText::new(short).small().weak());
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if assist_batch {
                                ui.spinner();
                                ui.label(RichText::new("全局采纳中…").small().weak());
                            } else if assist_infer {
                                ui.spinner();
                                ui.label(RichText::new("辅助推理中…").small().weak());
                            }
                        });
                    });

                    // 第二行：阈值、显示、类别、采纳（仅已载入模型时）
                    if self.assist_ort.is_some() {
                        ui.horizontal_wrapped(|ui| {
                            ui.add_enabled_ui(!assist_row2_blocked, |ui| {
                                ui.label(RichText::new("阈值").small().weak());
                                let conf_slider = ui.add_sized(
                                    [160.0, ui.spacing().interact_size.y],
                                    egui::Slider::new(&mut self.assist_onnx_conf, 0.0..=1.0)
                                        .fixed_decimals(2)
                                        .text("置信度"),
                                );
                                if conf_slider.changed() {
                                    self.assist_onnx_conf =
                                        self.assist_onnx_conf.clamp(0.0, 1.0);
                                    self.schedule_assist_infer();
                                }
                            });

                            ui.separator();

                            ui.add_enabled_ui(!assist_batch, |ui| {
                                let mut vis = self.assist_overlay_visible;
                                let r = ui
                                    .checkbox(&mut vis, "显示虚线预测")
                                    .on_hover_text(
                                        "是否在画布上绘制辅助框；关闭后仍可推理与采纳",
                                    );
                                if r.changed() {
                                    self.assist_overlay_visible = vis;
                                }
                            });

                            ui.separator();

                            if !self.assist_class_names.is_empty() {
                                self.ensure_assist_pred_class_mask();
                                let n = self.assist_class_names.len();
                                let n_on = self
                                    .assist_pred_class_on
                                    .iter()
                                    .filter(|&&x| x)
                                    .count();
                                let menu_label = if n_on == n {
                                    format!("类别筛选 · 全部 ({n})")
                                } else if n_on == 0 {
                                    "类别筛选 · 未选 ▾".to_string()
                                } else {
                                    format!("类别筛选 · {n_on}/{n} ▾")
                                };
                                ui.add_enabled_ui(!assist_batch, |ui| {
                                    ui.menu_button(RichText::new(menu_label).small(), |ui| {
                                        ui.label(
                                            RichText::new(
                                                "勾选参与预览与采纳的类别；未勾选则忽略该类预测",
                                            )
                                            .weak()
                                            .small(),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui.small_button("全选").clicked() {
                                                for v in &mut self.assist_pred_class_on {
                                                    *v = true;
                                                }
                                            }
                                            if ui.small_button("全不选").clicked() {
                                                for v in &mut self.assist_pred_class_on {
                                                    *v = false;
                                                }
                                            }
                                        });
                                        ui.separator();
                                        egui::ScrollArea::vertical()
                                            .max_height(220.0)
                                            .id_salt("assist_pred_class_scroll")
                                            .show(ui, |ui| {
                                                for i in 0..self.assist_class_names.len() {
                                                    let name = self.assist_class_names[i].as_str();
                                                    let mut on = self.assist_pred_class_on[i];
                                                    if ui.checkbox(&mut on, name).changed() {
                                                        self.assist_pred_class_on[i] = on;
                                                    }
                                                }
                                            });
                                    });
                                });
                            }

                            ui.separator();

                            let pct = (ASSIST_ADOPT_DUP_IOU * 100.0).round() as i32;
                            ui.add_enabled_ui(
                                !assist_row2_blocked
                                    && self.rgba.is_some()
                                    && !self.assist_preds.is_empty(),
                                |ui| {
                                    if ui
                                        .small_button("采纳当前图")
                                        .on_hover_text(format!(
                                            "将虚线框写入当前图 .txt；与已有框 IoU≥{pct}% 则不新增"
                                        ))
                                        .clicked()
                                    {
                                        self.adopt_onnx_assist_to_annotations();
                                        ctx.request_repaint();
                                    }
                                },
                            );

                            ui.add_enabled_ui(
                                !assist_row2_blocked && !self.image_paths.is_empty(),
                                |ui| {
                                    if ui
                                        .small_button("全局采纳")
                                        .on_hover_text(format!(
                                            "对左侧数据集每张图推理并写入标签；规则同「采纳当前图」；可用 Ctrl+Z 整集撤销（IoU≥{pct}% 保留原框）"
                                        ))
                                        .clicked()
                                    {
                                        self.schedule_assist_global_adopt();
                                        ctx.request_repaint();
                                    }
                                },
                            );
                        });
                    }
                });
            });
        ui.add_space(4.0);

        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("标注画布")
                    .strong()
                    .size(18.0)
                    .color(theme::TEXT),
            );
            status_chip(ui, "R 矩形", theme::ACCENT);
            status_chip(ui, "E 柔性外接", theme::WARN);
            status_chip(ui, "F 连续柔性", theme::WARN);
            status_chip(ui, "Ctrl+滚轮 缩放", theme::ACCENT);
            status_chip(ui, "空格+拖动 平移", theme::OK);
            CollapsingHeader::new(RichText::new("查看完整快捷键").small().color(theme::TEXT_MUTED))
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(
                            "R：矩形两点框 · E：柔性外接（整笔一框）· F：连续柔性外接（自相交成环；多边形套圈只保留外圈；两检测框 AABB 有面积重叠则删小留大、不合并；绿色笔画与预览；结束一笔后统一命名）· 点击开始 → 移动画线 → 再点击结束 · 右键退出当前模式 · Ctrl+滚轮缩放 · 空格+左键平移 · Del/Q 删除 · Esc 取消 · Ctrl+Z 撤销 · A/D 切图",
                        )
                        .small()
                        .weak(),
                    );
                });
        });
        ui.add_space(4.0);

        let Some(rgba) = &self.rgba else {
            self.image_texture = None;
            Frame::default()
                .fill(theme::SURFACE_ELEVATED)
                .inner_margin(egui::Margin::same(32.0))
                .rounding(egui::Rounding::same(12.0))
                .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
                .show(ui, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(8.0, 12.0);
                            ui.label(
                                RichText::new("尚未加载图片")
                                    .size(20.0)
                                    .strong()
                                    .color(theme::TEXT_MUTED),
                            );
                            ui.label(
                                RichText::new("在左侧「数据集」中选择图片目录，将列出该目录内的图片与同名标签（不含子文件夹）。")
                                    .small()
                                    .weak(),
                            );
                        });
                    });
                });
            return;
        };

        if self.texture_dirty {
            let packed = self.rgba.as_ref().map(|im| {
                let w = im.width() as usize;
                let h = im.height() as usize;
                (w, h, im.as_raw().to_vec())
            });
            if let Some((w, h, buf)) = packed {
                let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
                self.image_texture = Some(ctx.load_texture(
                    "yolo_canvas_image",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
            self.texture_dirty = false;
        }

        let Some(tex_id) = self.image_texture.as_ref().map(|t| t.id()) else {
            return;
        };

        let (img_w, img_h) = rgba.dimensions();
        let img_wf = img_w as f32;
        let img_hf = img_h as f32;

        let pad = 4.0;
        let top_limit = ui.min_rect().bottom() + pad;
        let full = ui.max_rect();
        let bottom_limit = (full.bottom() - pad).max(top_limit + 32.0);
        let inner = Rect::from_min_max(
            Pos2::new(full.left() + pad, top_limit),
            Pos2::new(full.right() - pad, bottom_limit),
        );
        let fit_base = fit_image_rect(inner, img_wf, img_hf);
        let disp = compute_view_disp_rect(fit_base, self.view_zoom, self.view_pan);
        self.last_canvas_inner = Some(inner);
        self.last_canvas_disp = Some(disp);

        let sense = Sense::click_and_drag().union(Sense::hover());
        let response = ui.allocate_rect(inner, sense);

        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));

        if response.hovered() && space_down {
            ctx.set_cursor_icon(if ctx.input(|i| i.pointer.primary_down()) {
                CursorIcon::Grabbing
            } else {
                CursorIcon::Grab
            });
        } else if response.hovered() && self.scribble_kind.is_some() {
            ctx.set_cursor_icon(CursorIcon::Crosshair);
        }

        let wants_ctrl_zoom =
            ctx.input(|i| i.modifiers.ctrl || i.modifiers.command || i.modifiers.mac_cmd);

        if response.hovered() {
            // 仅在有 Ctrl/Cmd 时用 zoom_delta 缩放；否则 zoom_delta 在触控板等环境下可能非 1，会吞掉普通滚轮导致无法换类
            if wants_ctrl_zoom {
                let zd = ctx.input(|i| i.zoom_delta());
                if (zd - 1.0).abs() > 1e-4 {
                    let z_old = self.view_zoom;
                    let z_new = (z_old * zd).clamp(0.05, 64.0);
                    if let Some(cursor) = ctx.input(|i| i.pointer.hover_pos()) {
                        adjust_pan_for_zoom_at_cursor(
                            fit_base,
                            &mut self.view_pan,
                            z_old,
                            z_new,
                            cursor,
                            img_wf,
                            img_hf,
                        );
                    }
                    self.view_zoom = z_new;
                }
                self.class_wheel_accum = 0.0;
            } else if !space_down && self.selected.is_some() && !self.classes.is_empty() {
                // 未按 Ctrl：用滚轮切换选中框类别（raw 优先，避免 ScrollArea 吃掉 smooth 后无响应）
                let sy = ctx.input(|i| {
                    let r = i.raw_scroll_delta.y;
                    if r.abs() > 0.5 {
                        r
                    } else {
                        i.smooth_scroll_delta.y
                    }
                });
                if sy.abs() > f32::EPSILON {
                    self.class_wheel_accum += sy;
                    // 累积约一整格再切一类；Line 模式常见 raw≈120
                    const NOTCH: f32 = 120.0;
                    let mut changed = false;
                    let mut pushed_wheel_undo = false;
                    while self.class_wheel_accum >= NOTCH {
                        if let Some(si) = self.selected {
                            if si < self.annotations.len() {
                                if !pushed_wheel_undo {
                                    self.push_undo(UndoScope::Local);
                                    pushed_wheel_undo = true;
                                }
                                let n = self.classes.len();
                                let b = &mut self.annotations[si];
                                b.class_id = (b.class_id + 1) % n;
                                changed = true;
                            }
                        }
                        self.class_wheel_accum -= NOTCH;
                    }
                    while self.class_wheel_accum <= -NOTCH {
                        if let Some(si) = self.selected {
                            if si < self.annotations.len() {
                                if !pushed_wheel_undo {
                                    self.push_undo(UndoScope::Local);
                                    pushed_wheel_undo = true;
                                }
                                let n = self.classes.len();
                                let b = &mut self.annotations[si];
                                b.class_id = (b.class_id + n - 1) % n;
                                changed = true;
                            }
                        }
                        self.class_wheel_accum += NOTCH;
                    }
                    if changed {
                        let _ = self.save_current_labels();
                    }
                }
            } else {
                self.class_wheel_accum = 0.0;
            }
        } else {
            self.class_wheel_accum = 0.0;
        }

        if space_down && response.hovered() && ctx.input(|i| i.pointer.primary_down()) {
            let d = ctx.input(|i| i.pointer.delta());
            if d != Vec2::ZERO {
                self.view_pan += d;
            }
        }

        let painter = ui.painter().with_clip_rect(inner);
        painter.rect_filled(inner, 0.0, Color32::from_rgb(12, 14, 18));
        painter.image(
            tex_id,
            disp,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );

        let pointer = ctx.input(|i| i.pointer.interact_pos());
        let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
        let primary_released = ctx.input(|i| i.pointer.primary_released());
        let dragging = ctx.input(|i| i.pointer.is_decidedly_dragging());

        if self.handles_anim_sel != self.selected {
            self.corner_hover_radius_anim = [0.0; 4];
            self.edge_hover_anim = [0.0; 4];
            self.handles_anim_sel = self.selected;
        }

        let dt = ctx.input(|i| i.stable_dt as f32).min(0.05);
        let hover_canvas = response.hovered();
        let prim_down = ctx.input(|i| i.pointer.primary_down());
        let drag_edge = self.drag.and_then(|(k, _)| match k {
            DragKind::ResizeEdge(e) => Some(e),
            _ => None,
        });
        let drag_corner_idx = self.drag.and_then(|(k, _)| match k {
            DragKind::Resize(ci) => Some(ci),
            _ => None,
        });

        let corner_hover_smooth = 1.0 - (-14.0_f32 * dt).exp();
        let edge_smooth = 1.0 - (-11.0_f32 * dt).exp();
        let edge_smooth_fast = 1.0 - (-32.0_f32 * dt).exp();

        for j in 0..4 {
            let mut target_r = 0.0_f32;
            if hover_canvas {
                if let (Some(si), Some(pp)) = (self.selected, pointer) {
                    if let Some(b) = self.annotations.get(si) {
                        if Self::hit_corner_screen(
                            pp,
                            b,
                            disp,
                            img_wf,
                            img_hf,
                            CORNER_HANDLE_PX * 1.2,
                        ) == Some(j)
                        {
                            target_r = 1.0;
                        }
                    }
                }
            }
            if let Some(dc) = drag_corner_idx {
                if dc == j && prim_down {
                    target_r = 1.0;
                }
            }
            self.corner_hover_radius_anim[j] += (target_r - self.corner_hover_radius_anim[j])
                * corner_hover_smooth;
        }

        for e in 0..4_u8 {
            let mut target = 0.0_f32;
            if hover_canvas {
                if let (Some(si), Some(pp)) = (self.selected, pointer) {
                    if let Some(b) = self.annotations.get(si) {
                        if Self::hit_corner_screen(pp, b, disp, img_wf, img_hf, CORNER_HANDLE_PX)
                            .is_none()
                            && Self::hit_edge_screen(
                                pp,
                                b,
                                disp,
                                img_wf,
                                img_hf,
                                EDGE_HIT_PX,
                                EDGE_CORNER_SKIP_SCREEN,
                            ) == Some(e)
                        {
                            target = 1.0;
                        }
                    }
                }
            }
            if drag_edge == Some(e) && prim_down {
                target = 0.0;
            }
            let s = if drag_edge == Some(e) && prim_down {
                edge_smooth_fast
            } else {
                edge_smooth
            };
            self.edge_hover_anim[e as usize] +=
                (target - self.edge_hover_anim[e as usize]) * s;
        }

        let block_bbox = space_down || self.show_label_window;

        if response.secondary_clicked() && response.hovered() && !space_down {
            if self.scribble_kind.is_some() {
                self.scribble_kind = None;
                self.scribble_active = false;
                self.scribble_points.clear();
                self.scribble_open_start = 0;
                self.scribble_closed_boxes.clear();
            } else if self.draw_new_boxes_enabled {
                self.draw_new_boxes_enabled = false;
                if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                    self.draw_phase = DrawPhase::Idle;
                }
            } else if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                self.draw_phase = DrawPhase::Idle;
            } else if !self.show_label_window {
                if let Some(p) = pointer {
                    let on_any = if let Some((ix, iy)) = screen_to_image(p, disp, img_wf, img_hf) {
                        self.annotations.iter().any(|b| {
                            Self::hit_inside((ix, iy), b)
                                || Self::hit_corner_screen(p, b, disp, img_wf, img_hf, CORNER_HANDLE_PX)
                                    .is_some()
                                || Self::hit_edge_screen(
                                    p,
                                    b,
                                    disp,
                                    img_wf,
                                    img_hf,
                                    EDGE_HIT_PX,
                                    EDGE_CORNER_SKIP_SCREEN,
                                )
                                .is_some()
                        })
                    } else {
                        false
                    };
                    if !on_any {
                        self.selected = None;
                    }
                }
            }
        }

        if let Some(p) = pointer {
            // 拖拽缩放角点时可略出图外，仍用外推坐标；起点/终点画框必须落在当前图像显示矩形 disp 内，
            // 否则侧栏等处点击也会触发 primary_pressed，会误当成第二点并弹出「输入标签」。
            let (ix, iy) = map_screen_to_image_px(p, disp, img_wf, img_hf);

            let mut skip_primary_after_dbl = false;
            if ctx.input(|i| i.pointer.button_double_clicked(PointerButton::Primary))
                && !block_bbox
                && !self.draw_new_boxes_enabled
                && self.scribble_kind.is_none()
                && !space_down
                && response.hovered()
            {
                if let Some((ix, iy)) = screen_to_image(p, disp, img_wf, img_hf) {
                    if let Some(idx) = self
                        .annotations
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, b)| Self::hit_inside((ix, iy), b))
                        .map(|(i, _)| i)
                    {
                        self.drag = None;
                        self.draw_phase = DrawPhase::Idle;
                        self.selected = Some(idx);
                        self.label_edit_idx = Some(idx);
                        self.pending_box = None;
                        self.label_draft = self
                            .classes
                            .get(self.annotations[idx].class_id)
                            .cloned()
                            .unwrap_or_default();
                        self.show_label_window = true;
                        skip_primary_after_dbl = true;
                    }
                }
            }

            if self.scribble_kind.is_some() && !block_bbox && !space_down {
                let (ix, iy) = map_screen_to_image_px(p, disp, img_wf, img_hf);
                let ix = ix.clamp(0.0, img_wf);
                let iy = iy.clamp(0.0, img_hf);
                if primary_pressed
                    && response.hovered()
                    && screen_to_image(p, disp, img_wf, img_hf).is_some()
                {
                    if self.scribble_active {
                        if let Some(&(lx, ly)) = self.scribble_points.last() {
                            let d = ((ix - lx).powi(2) + (iy - ly).powi(2)).sqrt();
                            if d >= 1.0 {
                                self.scribble_points.push((ix, iy));
                            }
                        }
                        if matches!(
                            self.scribble_kind,
                            Some(ScribbleKind::ContinuousCircumscribed)
                        ) {
                            self.try_close_scribble_loop(img_w, img_h);
                        }
                        self.scribble_active = false;
                        self.finalize_scribble_to_pending(img_w, img_h);
                    } else {
                        self.scribble_active = true;
                        self.scribble_points.clear();
                        self.scribble_closed_boxes.clear();
                        self.scribble_open_start = 0;
                        self.scribble_points.push((ix, iy));
                        self.selected = None;
                        self.drag = None;
                        self.draw_phase = DrawPhase::Idle;
                    }
                } else if self.scribble_active && response.hovered() {
                    if let Some(&(lx, ly)) = self.scribble_points.last() {
                        let d = ((ix - lx).powi(2) + (iy - ly).powi(2)).sqrt();
                        if d >= 2.0 {
                            self.scribble_points.push((ix, iy));
                            if matches!(
                                self.scribble_kind,
                                Some(ScribbleKind::ContinuousCircumscribed)
                            ) {
                                self.try_close_scribble_loop(img_w, img_h);
                            }
                        }
                    }
                }
            }

            if primary_pressed && !block_bbox && !skip_primary_after_dbl && self.scribble_kind.is_none()
            {
                if let Some((ix, iy)) = screen_to_image(p, disp, img_wf, img_hf) {
                    if self.draw_new_boxes_enabled {
                        match &self.draw_phase {
                            DrawPhase::Idle => {
                                self.draw_phase = DrawPhase::AwaitingSecondClick { ax: ix, ay: iy };
                                self.selected = None;
                            }
                            DrawPhase::AwaitingSecondClick { ax, ay } => {
                                let mut pb = PendingBox {
                                    min_x: *ax,
                                    min_y: *ay,
                                    max_x: ix,
                                    max_y: iy,
                                };
                                if (pb.max_x - pb.min_x).abs() < 2.0
                                    && (pb.max_y - pb.min_y).abs() < 2.0
                                {
                                    self.draw_phase = DrawPhase::Idle;
                                } else {
                                    if pb.min_x > pb.max_x {
                                        std::mem::swap(&mut pb.min_x, &mut pb.max_x);
                                    }
                                    if pb.min_y > pb.max_y {
                                        std::mem::swap(&mut pb.min_y, &mut pb.max_y);
                                    }
                                    self.pending_box = Some(pb);
                                    self.label_edit_idx = None;
                                    self.show_label_window = true;
                                    self.draw_new_boxes_enabled = false;
                                    self.label_draft = self.active_class_label_draft();
                                    self.draw_phase = DrawPhase::Idle;
                                }
                            }
                        }
                    } else {
                        if let Some(si) = self.selected {
                            if let Some(b) = self.annotations.get(si) {
                                if let Some(ci) =
                                    Self::hit_corner_screen(p, b, disp, img_wf, img_hf, CORNER_HANDLE_PX)
                                {
                                    self.push_undo(UndoScope::Local);
                                    self.drag = Some((DragKind::Resize(ci), si));
                                } else if let Some(ei) = Self::hit_edge_screen(
                                    p,
                                    b,
                                    disp,
                                    img_wf,
                                    img_hf,
                                    EDGE_HIT_PX,
                                    EDGE_CORNER_SKIP_SCREEN,
                                ) {
                                    self.push_undo(UndoScope::Local);
                                    self.drag = Some((DragKind::ResizeEdge(ei), si));
                                } else if Self::hit_inside((ix, iy), b) {
                                    self.push_undo(UndoScope::Local);
                                    self.drag = Some((DragKind::Move, si));
                                }
                            }
                        }

                        if self.drag.is_none() {
                            match &self.draw_phase {
                                DrawPhase::Idle => {
                                    for (idx, b) in self.annotations.iter().enumerate() {
                                        if Self::hit_inside((ix, iy), b) {
                                            self.push_undo(UndoScope::Local);
                                            self.selected = Some(idx);
                                            self.drag = Some((DragKind::Move, idx));
                                            break;
                                        }
                                    }
                                }
                                DrawPhase::AwaitingSecondClick { .. } => {
                                    self.draw_phase = DrawPhase::Idle;
                                }
                            }
                        }
                    }
                }
            }

            if let Some((kind, idx)) = self.drag {
                if dragging || ctx.input(|i| i.pointer.primary_down()) {
                    if let Some(b) = self.annotations.get_mut(idx) {
                        match kind {
                            DragKind::Move => {
                                let d = response.drag_delta();
                                if d.x != 0.0 || d.y != 0.0 {
                                    let dx = d.x / disp.width() * img_wf;
                                    let dy = d.y / disp.height() * img_hf;
                                    b.min_x += dx;
                                    b.max_x += dx;
                                    b.min_y += dy;
                                    b.max_y += dy;
                                    b.normalize(img_w, img_h);
                                }
                            }
                            DragKind::Resize(ci) => {
                                match ci {
                                    0 => {
                                        b.min_x = ix.min(b.max_x - 2.0);
                                        b.min_y = iy.min(b.max_y - 2.0);
                                    }
                                    1 => {
                                        b.max_x = ix.max(b.min_x + 2.0);
                                        b.min_y = iy.min(b.max_y - 2.0);
                                    }
                                    2 => {
                                        b.max_x = ix.max(b.min_x + 2.0);
                                        b.max_y = iy.max(b.min_y + 2.0);
                                    }
                                    _ => {
                                        b.min_x = ix.min(b.max_x - 2.0);
                                        b.max_y = iy.max(b.min_y + 2.0);
                                    }
                                }
                                b.normalize(img_w, img_h);
                            }
                            DragKind::ResizeEdge(e) => {
                                match e {
                                    0 => {
                                        b.min_y = iy.min(b.max_y - 2.0);
                                    }
                                    1 => {
                                        b.max_x = ix.max(b.min_x + 2.0);
                                    }
                                    2 => {
                                        b.max_y = iy.max(b.min_y + 2.0);
                                    }
                                    _ => {
                                        b.min_x = ix.min(b.max_x - 2.0);
                                    }
                                }
                                b.normalize(img_w, img_h);
                            }
                        }
                    }
                }
                if primary_released {
                    if self.drag.is_some() {
                        let _ = self.save_current_labels();
                    }
                    self.drag = None;
                }
            }
        }

        let typing_elsewhere = ctx.wants_keyboard_input();
        ctx.input(|i| {
            let q_remove = i.key_pressed(egui::Key::Q)
                && !self.show_label_window
                && !typing_elsewhere;
            if i.key_pressed(egui::Key::Delete) || q_remove {
                if let Some(si) = self.selected {
                    if si < self.annotations.len() {
                        self.push_undo(UndoScope::Local);
                        self.annotations.remove(si);
                        self.selected = None;
                        let _ = self.save_current_labels();
                    }
                }
            }
        });

        for (i, b) in self.annotations.iter().enumerate() {
            let c_tl = image_to_screen(b.min_x, b.min_y, disp, img_wf, img_hf);
            let c_br = image_to_screen(b.max_x, b.max_y, disp, img_wf, img_hf);
            let r = Rect::from_two_pos(c_tl, c_br);
            let cc = self.display_color_for_class(b.class_id);
            if self.selected == Some(i) {
                let t = ctx.input(|i| i.time);
                let fill_a = selection_fill_blink_alpha(t, 0.5, 0.1, 0.12);
                if fill_a > 0 {
                    painter.rect_filled(
                        r,
                        0.0,
                        Color32::from_rgba_unmultiplied(cc.r(), cc.g(), cc.b(), fill_a),
                    );
                }
                let m = STROKE_ENDPOINT_INSET;
                let tr = image_to_screen(b.max_x, b.min_y, disp, img_wf, img_hf);
                let bl = image_to_screen(b.min_x, b.max_y, disp, img_wf, img_hf);
                let base_w = 3.0_f32;
                let ew: [f32; 4] = [
                    base_w + EDGE_HOVER_THICK_EXTRA * self.edge_hover_anim[0],
                    base_w + EDGE_HOVER_THICK_EXTRA * self.edge_hover_anim[1],
                    base_w + EDGE_HOVER_THICK_EXTRA * self.edge_hover_anim[2],
                    base_w + EDGE_HOVER_THICK_EXTRA * self.edge_hover_anim[3],
                ];
                painter.line_segment(
                    [c_tl + Vec2::new(m, 0.0), tr + Vec2::new(-m, 0.0)],
                    Stroke::new(ew[0], cc),
                );
                painter.line_segment(
                    [tr + Vec2::new(0.0, m), c_br + Vec2::new(0.0, -m)],
                    Stroke::new(ew[1], cc),
                );
                painter.line_segment(
                    [c_br + Vec2::new(-m, 0.0), bl + Vec2::new(m, 0.0)],
                    Stroke::new(ew[2], cc),
                );
                painter.line_segment(
                    [bl + Vec2::new(0.0, -m), c_tl + Vec2::new(0.0, m)],
                    Stroke::new(ew[3], cc),
                );

                for (j, c) in Self::corner_points(b).iter().enumerate() {
                    let base = image_to_screen(c.x, c.y, disp, img_wf, img_hf);
                    let a = self.corner_hover_radius_anim[j].clamp(0.0, 1.0);
                    let rad = CORNER_DRAW_RADIUS
                        * (1.0 + (CORNER_HOVER_RADIUS_SCALE - 1.0) * a);
                    painter.circle_filled(base, rad, Color32::from_rgb(255, 220, 80));
                }
            } else {
                painter.rect_stroke(r, 0.0, Stroke::new(2.0, cc));
            }

            let name = self
                .classes
                .get(b.class_id)
                .map(String::as_str)
                .unwrap_or("?");
            let galley = painter.layout_no_wrap(
                format!("{} [{}]", name, b.class_id),
                egui::FontId::proportional(14.0),
                cc,
            );
            let mut tp = c_tl;
            tp.y -= galley.size().y + 2.0;
            let bg = Rect::from_min_size(tp, galley.size()).expand(3.0);
            painter.rect_filled(bg, 3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 170));
            painter.add(egui::Shape::galley(tp, galley, cc));
        }

        const ASSIST_DASH: f32 = 5.0;
        const ASSIST_GAP: f32 = 4.0;
        if self.assist_overlay_visible {
            for pr in &self.assist_preds {
                if !self.assist_pred_class_visible(pr.model_class_id) {
                    continue;
                }
                if self.assist_pred_suppressed_by_annotations(pr) {
                    continue;
                }
                let c_tl = image_to_screen(pr.min_x, pr.min_y, disp, img_wf, img_hf);
                let c_br = image_to_screen(pr.max_x, pr.max_y, disp, img_wf, img_hf);
                let r = Rect::from_two_pos(c_tl, c_br);
                let cc = palette_color(pr.model_class_id.wrapping_add(17));
                let stroke = Stroke::new(2.0, cc);
                let mn = r.min;
                let mx = r.max;
                let top = [Pos2::new(mn.x, mn.y), Pos2::new(mx.x, mn.y)];
                let right = [Pos2::new(mx.x, mn.y), Pos2::new(mx.x, mx.y)];
                let bottom = [Pos2::new(mx.x, mx.y), Pos2::new(mn.x, mx.y)];
                let left = [Pos2::new(mn.x, mx.y), Pos2::new(mn.x, mn.y)];
                for seg in [&top[..], &right[..], &bottom[..], &left[..]] {
                    for s in Shape::dashed_line(seg, stroke, ASSIST_DASH, ASSIST_GAP) {
                        painter.add(s);
                    }
                }
                let pname = self.assist_model_class_label(pr.model_class_id);
                let pct = (pr.conf * 100.0).round().clamp(0.0, 100.0) as i32;
                let galley = painter.layout_no_wrap(
                    format!("{pname} [{pct}%] · 辅助"),
                    egui::FontId::proportional(13.0),
                    cc,
                );
                let mut tp = c_tl;
                tp.y -= galley.size().y + 2.0;
                let bg = Rect::from_min_size(tp, galley.size()).expand(3.0);
                painter.rect_filled(bg, 3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 140));
                painter.add(egui::Shape::galley(tp, galley, cc));
            }
        }

        if self.draw_new_boxes_enabled {
            if let DrawPhase::AwaitingSecondClick { ax, ay } = &self.draw_phase {
                if let Some(p) = pointer {
                    let (bx, by) = map_screen_to_image_px(p, disp, img_wf, img_hf);
                    let a = image_to_screen(*ax, *ay, disp, img_wf, img_hf);
                    let b = image_to_screen(bx, by, disp, img_wf, img_hf);
                    let r = Rect::from_two_pos(a, b);
                    // 透明度约 0.1 的淡粉填充
                    let pink_fill = Color32::from_rgba_unmultiplied(255, 182, 193, 26);
                    painter.rect_filled(r, 2.0, pink_fill);
                    painter.rect_stroke(
                        r,
                        2.0,
                        Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 140, 170, 220)),
                    );
                }
            }
        }

        const SCRIBBLE_LINE_W: f32 = 2.2;
        let (scribble_col, scribble_col_soft) = match self.scribble_kind {
            Some(ScribbleKind::ContinuousCircumscribed) => (
                Color32::from_rgb(60, 255, 120),
                Color32::from_rgba_unmultiplied(60, 255, 120, 210),
            ),
            _ => (
                Color32::from_rgb(255, 72, 72),
                Color32::from_rgba_unmultiplied(255, 72, 72, 200),
            ),
        };
        if self.scribble_points.len() >= 2 {
            let pts: Vec<Pos2> = self
                .scribble_points
                .iter()
                .map(|(x, y)| image_to_screen(*x, *y, disp, img_wf, img_hf))
                .collect();
            painter.add(egui::Shape::line(
                pts,
                Stroke::new(SCRIBBLE_LINE_W, scribble_col),
            ));
        } else if self.scribble_points.len() == 1 {
            let (x, y) = self.scribble_points[0];
            let c = image_to_screen(x, y, disp, img_wf, img_hf);
            painter.circle_filled(c, 3.0, scribble_col);
        }
        if matches!(
            self.scribble_kind,
            Some(ScribbleKind::ContinuousCircumscribed)
        ) {
            for cb in &self.scribble_closed_boxes {
                let a = &cb.aabb;
                let tl = image_to_screen(a.min_x, a.min_y, disp, img_wf, img_hf);
                let br = image_to_screen(a.max_x, a.max_y, disp, img_wf, img_hf);
                let pr = Rect::from_two_pos(tl, br);
                painter.rect_stroke(pr, 3.0, Stroke::new(3.0, scribble_col_soft));
            }
        } else if self.scribble_active && !self.scribble_points.is_empty() {
            let mut sx = f32::INFINITY;
            let mut sy = f32::INFINITY;
            let mut ex = f32::NEG_INFINITY;
            let mut ey = f32::NEG_INFINITY;
            for &(x, y) in &self.scribble_points {
                let x = x.clamp(0.0, img_wf);
                let y = y.clamp(0.0, img_hf);
                sx = sx.min(x);
                sy = sy.min(y);
                ex = ex.max(x);
                ey = ey.max(y);
            }
            let tl = image_to_screen(sx, sy, disp, img_wf, img_hf);
            let br = image_to_screen(ex, ey, disp, img_wf, img_hf);
            let pr = Rect::from_two_pos(tl, br);
            painter.rect_stroke(
                pr,
                3.0,
                Stroke::new(3.0, scribble_col_soft),
            );
        }

        // 柔性外接：红色十字；连续柔性外接：绿色十字；矩形拉框：淡绿色十字
        if response.hovered() && self.scribble_kind.is_some() {
            let p = pointer.or_else(|| ctx.input(|i| i.pointer.hover_pos()));
            if let Some(p) = p {
                if disp.contains(p) {
                    let dash_col = match self.scribble_kind {
                        Some(ScribbleKind::ContinuousCircumscribed) => {
                            Color32::from_rgba_unmultiplied(60, 255, 120, 200)
                        }
                        _ => Color32::from_rgba_unmultiplied(255, 72, 72, 200),
                    };
                    let stroke = Stroke::new(1.0, dash_col);
                    let h_seg = [Pos2::new(inner.left(), p.y), Pos2::new(inner.right(), p.y)];
                    let v_seg = [Pos2::new(p.x, inner.top()), Pos2::new(p.x, inner.bottom())];
                    for s in Shape::dashed_line(&h_seg, stroke, CROSSHAIR_DASH, CROSSHAIR_GAP) {
                        painter.add(s);
                    }
                    for s in Shape::dashed_line(&v_seg, stroke, CROSSHAIR_DASH, CROSSHAIR_GAP) {
                        painter.add(s);
                    }
                }
            }
        } else if response.hovered() && self.draw_new_boxes_enabled {
            let p = pointer.or_else(|| ctx.input(|i| i.pointer.hover_pos()));
            if let Some(p) = p {
                if disp.contains(p) {
                    let stroke = Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(60, 255, 120, 77),
                    );
                    let h_seg = [Pos2::new(inner.left(), p.y), Pos2::new(inner.right(), p.y)];
                    let v_seg = [Pos2::new(p.x, inner.top()), Pos2::new(p.x, inner.bottom())];
                    for s in Shape::dashed_line(&h_seg, stroke, CROSSHAIR_DASH, CROSSHAIR_GAP) {
                        painter.add(s);
                    }
                    for s in Shape::dashed_line(&v_seg, stroke, CROSSHAIR_DASH, CROSSHAIR_GAP) {
                        painter.add(s);
                    }
                }
            }
        }

        let mode_str: String = match self.scribble_kind {
            Some(ScribbleKind::Circumscribed) => "柔性外接".to_string(),
            Some(ScribbleKind::ContinuousCircumscribed) => {
                let n = self.scribble_closed_boxes.len();
                if n > 0 {
                    format!("连续外接 · 已闭合 {n} 块（自相交成环）")
                } else {
                    "连续外接（与更早边相交成环，不必首尾相接）".to_string()
                }
            }
            None if self.draw_new_boxes_enabled => "拉新框".to_string(),
            None => "调整框".to_string(),
        };
        let hud = format!("缩放 {:.0}% · {}", self.view_zoom * 100.0, mode_str);
        let hud_font = egui::FontId::proportional(13.0);
        let hud_color = Color32::from_rgb(232, 238, 248);
        let hud_pos = inner.right_top() + Vec2::new(-10.0, 8.0);
        let galley_hud = painter.layout_no_wrap(hud, hud_font, hud_color);
        let text_rect = Align2::RIGHT_TOP.anchor_size(hud_pos, galley_hud.size());
        painter.rect_filled(
            text_rect.expand(5.0),
            5.0,
            Color32::from_rgba_unmultiplied(18, 20, 28, 210),
        );
        painter.galley(text_rect.min, galley_hud, hud_color);
    }

    fn ui_top_bar(&mut self, ui: &mut Ui) {
        let top_h = 96.0;
        let gap = 10.0;
        let total_w = ui.available_width();
        // 全图总览卡片最大宽度（靠右对齐；右缘与下方标注画布右缘对齐，画布矩形来自上一帧）
        let overview_panel_max_w = 240.0;
        let left_target = self.sidebar_width.clamp(280.0, 500.0);
        let left_w = left_target.min((total_w - gap - overview_panel_max_w).max(240.0));

        Frame::default()
            .fill(theme::SURFACE)
            .inner_margin(egui::Margin::symmetric(12.0, 6.0))
            .stroke(Stroke::new(1.0, theme::BORDER))
            .show(ui, |ui| {
                let bar_right = ui.max_rect().right();
                let canvas_right = self
                    .last_canvas_inner
                    .map(|r| r.right())
                    .unwrap_or_else(|| bar_right);
                let overview_right = canvas_right.min(bar_right);

                let left_rect = Rect::from_min_size(ui.cursor().min, Vec2::new(left_w, top_h));
                let min_overview_left = left_rect.right() + gap;
                let slot = (overview_right - min_overview_left).max(1.0);
                let overview_w = slot.min(overview_panel_max_w);
                let overview_left = overview_right - overview_w;
                let right_rect = Rect::from_min_size(
                    Pos2::new(overview_left, left_rect.top()),
                    Vec2::new(overview_w, top_h),
                );
                let union_rect = Rect::from_min_max(left_rect.min, right_rect.max);
                ui.allocate_rect(union_rect, Sense::hover());

                let mut left_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(left_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                Frame::none()
                    .inner_margin(egui::Margin::same(14.0))
                    .show(&mut left_ui, |ui| {
                        ui.set_min_height(top_h - 16.0);
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new("YOLO水平框专属标注工具")
                                        .strong()
                                        .size(19.0)
                                        .color(theme::TEXT),
                                );
                                ui.label(
                                    RichText::new("致力于快速拿到预模型和终模型")
                                        .size(15.0)
                                        .color(theme::TEXT_MUTED),
                                );
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if self.training {
                                    status_chip(ui, "训练中", theme::WARN);
                                } else {
                                    status_chip(ui, "待命", theme::OK);
                                }
                            });
                        });
                    });

                let right_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(right_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                let rect = right_ui.max_rect();
                if let (Some(inner), Some(disp), Some(rgba)) =
                    (self.last_canvas_inner, self.last_canvas_disp, self.rgba.as_ref())
                {
                    let img_wf = rgba.width() as f32;
                    let img_hf = rgba.height() as f32;
                    let layout = compute_overview_minimap_layout(rect, inner, disp, img_wf, img_hf);
                    if let Some(red_r) = layout.red_on_screen {
                        let vp_id = ui.id().with("overview_viewport");
                        let resp = ui
                            .interact(
                                red_r,
                                vp_id,
                                Sense::click_and_drag().union(Sense::hover()),
                            )
                            .on_hover_cursor(CursorIcon::Grab);
                        let ctx = ui.ctx().clone();
                        self.handle_overview_viewport_interaction(
                            &ctx,
                            &resp,
                            inner,
                            disp,
                            &layout,
                            img_wf,
                            img_hf,
                        );
                    }
                    self.paint_canvas_overview(&right_ui, &layout, img_wf, img_hf);
                } else {
                    let painter = right_ui.painter();
                    let card_rect = rect.shrink2(Vec2::new(1.0, 2.0));
                    painter.rect_filled(card_rect, 12.0, theme::SURFACE_ELEVATED);
                    painter.rect_stroke(
                        card_rect,
                        12.0,
                        Stroke::new(1.0, color_alpha(theme::ACCENT, 54)),
                    );
                }
            });
    }

    fn ui_train_log_panel(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let toggle_text = if self.train_log_expanded {
                    "收起训练日志"
                } else {
                    "展开训练日志"
                };
                if ui
                    .button(toggle_text)
                    .on_hover_text("点击切换训练日志面板的展开与折叠")
                    .clicked()
                {
                    self.train_log_expanded = !self.train_log_expanded;
                }

                ui.label(RichText::new("训练日志").strong().size(15.0).color(theme::TEXT));
                ui.label(
                    RichText::new(format!("{} 行", self.train_log.len()))
                        .small()
                        .color(theme::TEXT_MUTED),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.train_log_expanded && ui.small_button("清空日志").clicked() {
                        self.train_log.clear();
                    }
                });
            });
            if self.train_log_expanded {
                ui.add_space(8.0);
                app_card(theme::SURFACE_ELEVATED).show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(280.0)
                        .stick_to_bottom(true)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                            if self.train_log.is_empty() {
                                ui.label(
                                    RichText::new("训练启动后，Ultralytics 输出会实时显示在这里。")
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                );
                            } else {
                                for line in &self.train_log {
                                    ui.label(
                                        RichText::new(line.as_str())
                                            .color(Color32::from_rgb(208, 216, 229))
                                            .size(12.0),
                                    );
                                }
                            }
                        });
                });
            } else {
                let preview = self
                    .train_log
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "训练日志默认收起，点击左侧按钮展开查看。".to_string());
                ui.add_space(4.0);
                ui.label(RichText::new(preview).small().color(theme::TEXT_MUTED));
            }
        });
    }

    fn handle_overview_viewport_interaction(
        &mut self,
        ctx: &Context,
        response: &Response,
        inner: Rect,
        disp: Rect,
        layout: &OverviewMinimapLayout,
        img_wf: f32,
        img_hf: f32,
    ) {
        let fit_base = fit_image_rect(inner, img_wf, img_hf);
        let preview_rect = layout.preview_rect;

        if response.dragged_by(PointerButton::Primary) {
            ctx.set_cursor_icon(CursorIcon::Grabbing);
            let d = ctx.input(|i| i.pointer.delta());
            if d != Vec2::ZERO {
                let sx = disp.width() / preview_rect.width().max(1.0);
                let sy = disp.height() / preview_rect.height().max(1.0);
                // 与缩略图上的位移同向：拖红框向右 → 视野在图像中向右移（对应主画布上 disp 左移）
                self.view_pan.x -= d.x * sx;
                self.view_pan.y -= d.y * sy;
                ctx.request_repaint();
            }
        }

        if response.hovered() {
            let mut zd = ctx.input(|i| i.zoom_delta());
            if (zd - 1.0).abs() <= 1e-4 {
                let sy = ctx.input(|i| i.raw_scroll_delta.y + i.smooth_scroll_delta.y);
                if sy.abs() > 0.5 {
                    zd = (1.0 + sy * 0.0012).max(0.02);
                }
            }
            if (zd - 1.0).abs() > 1e-4 {
                let z_old = self.view_zoom;
                let z_new = (z_old * zd).clamp(0.05, 64.0);
                if (z_new - z_old).abs() > f32::EPSILON {
                    let anchor = if let Some(vr) = layout.visible_in_image {
                        let cx = (vr.min.x + vr.max.x) * 0.5;
                        let cy = (vr.min.y + vr.max.y) * 0.5;
                        image_to_screen(cx, cy, disp, img_wf, img_hf)
                    } else {
                        disp.center()
                    };
                    adjust_pan_for_zoom_at_cursor(
                        fit_base,
                        &mut self.view_pan,
                        z_old,
                        z_new,
                        anchor,
                        img_wf,
                        img_hf,
                    );
                    self.view_zoom = z_new;
                    ctx.request_repaint();
                }
            }
        }
    }

    fn paint_canvas_overview(
        &self,
        ui: &Ui,
        layout: &OverviewMinimapLayout,
        img_wf: f32,
        img_hf: f32,
    ) {
        let painter = ui.painter();
        let card_rect = layout.card_rect;
        painter.rect_filled(card_rect, 12.0, theme::SURFACE_ELEVATED);
        painter.rect_stroke(
            card_rect,
            12.0,
            Stroke::new(1.0, color_alpha(theme::ACCENT, 54)),
        );

        let danger = Color32::from_rgb(222, 78, 78);
        let preview_rect = layout.preview_rect;
        painter.rect_filled(preview_rect, 12.0, Color32::from_rgb(251, 251, 248));
        painter.rect_stroke(
            preview_rect,
            12.0,
            Stroke::new(1.0, Color32::from_rgb(214, 218, 224)),
        );

        for (idx, ann) in self.annotations.iter().enumerate() {
            let cx = (ann.min_x + ann.max_x) * 0.5;
            let cy = (ann.min_y + ann.max_y) * 0.5;
            let p = image_to_screen(cx, cy, preview_rect, img_wf, img_hf);
            let color = self.display_color_for_class(ann.class_id);
            let radius = if self.selected == Some(idx) { 5.0 } else { 3.5 };
            painter.circle_filled(p, radius, color);
            painter.circle_stroke(p, radius, Stroke::new(1.2, Color32::WHITE));
        }

        if let Some(view_rect) = layout.visible_in_image {
            let tl = image_to_screen(view_rect.min.x, view_rect.min.y, preview_rect, img_wf, img_hf);
            let br = image_to_screen(view_rect.max.x, view_rect.max.y, preview_rect, img_wf, img_hf);
            let r = Rect::from_two_pos(tl, br);
            painter.rect_filled(r, 8.0, Color32::from_rgba_unmultiplied(225, 70, 70, 26));
            painter.rect_stroke(r, 8.0, Stroke::new(2.0, danger));
        }
    }
}

impl eframe::App for YoloTrainerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.poll_training(ctx);
        self.poll_assist_infer(ctx);
        self.poll_assist_batch(ctx);
        if !self.class_log_bootstrapped {
            self.class_log_bootstrapped = true;
            self.load_class_log_from_disk();
        }
        self.rebuild_annotated_strip_if_dirty();
        if !self.conda_env_list_bootstrapped {
            self.refresh_conda_env_list();
            self.conda_env_list_bootstrapped = true;
        }

        ctx.input(|i| {
            if i.key_pressed(egui::Key::Escape) {
                if self.show_label_window {
                    self.pending_box = None;
                    self.pending_boxes_batch.clear();
                    self.label_edit_idx = None;
                    self.show_label_window = false;
                    self.draw_phase = DrawPhase::Idle;
                } else if self.scribble_active {
                    self.scribble_active = false;
                    self.scribble_points.clear();
                    self.scribble_open_start = 0;
                    self.scribble_closed_boxes.clear();
                } else if self.scribble_kind.is_some() {
                    self.scribble_kind = None;
                    self.scribble_open_start = 0;
                    self.scribble_closed_boxes.clear();
                } else if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                    self.draw_phase = DrawPhase::Idle;
                }
            }
        });

        if !self.show_label_window && !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                self.draw_new_boxes_enabled = !self.draw_new_boxes_enabled;
                if self.draw_new_boxes_enabled {
                    self.scribble_kind = None;
                    self.scribble_active = false;
                    self.scribble_points.clear();
                    self.scribble_open_start = 0;
                    self.scribble_closed_boxes.clear();
                    self.selected = None;
                    self.drag = None;
                } else if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                    self.draw_phase = DrawPhase::Idle;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::E)) {
                match self.scribble_kind {
                    Some(ScribbleKind::Circumscribed) => {
                        self.scribble_kind = None;
                        self.scribble_active = false;
                        self.scribble_points.clear();
                        self.scribble_open_start = 0;
                        self.scribble_closed_boxes.clear();
                    }
                    _ => {
                        self.scribble_kind = Some(ScribbleKind::Circumscribed);
                        self.draw_new_boxes_enabled = false;
                        if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                            self.draw_phase = DrawPhase::Idle;
                        }
                        self.scribble_active = false;
                        self.scribble_points.clear();
                        self.scribble_open_start = 0;
                        self.scribble_closed_boxes.clear();
                        self.selected = None;
                        self.drag = None;
                    }
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::F)) {
                match self.scribble_kind {
                    Some(ScribbleKind::ContinuousCircumscribed) => {
                        self.scribble_kind = None;
                        self.scribble_active = false;
                        self.scribble_points.clear();
                        self.scribble_open_start = 0;
                        self.scribble_closed_boxes.clear();
                    }
                    _ => {
                        self.scribble_kind = Some(ScribbleKind::ContinuousCircumscribed);
                        self.draw_new_boxes_enabled = false;
                        if matches!(self.draw_phase, DrawPhase::AwaitingSecondClick { .. }) {
                            self.draw_phase = DrawPhase::Idle;
                        }
                        self.scribble_active = false;
                        self.scribble_points.clear();
                        self.scribble_open_start = 0;
                        self.scribble_closed_boxes.clear();
                        self.selected = None;
                        self.drag = None;
                    }
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::A)) {
                self.go_prev_image();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::D)) {
                self.go_next_image();
            }
            if ctx.input(|i| {
                i.key_pressed(egui::Key::Z) && (i.modifiers.ctrl || i.modifiers.command)
            }) {
                self.apply_undo();
            }
        }

        egui::TopBottomPanel::top("top")
            .frame(Frame::none().fill(theme::SURFACE_DEEP))
            .show(ctx, |ui| {
                self.ui_top_bar(ui);
            });

        let train_log_panel = egui::TopBottomPanel::bottom("train_log").frame(
            Frame::default()
                .fill(theme::SURFACE)
                .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                .stroke(Stroke::new(1.0, theme::BORDER)),
        );
        let train_log_panel = if self.train_log_expanded {
            train_log_panel
                .resizable(true)
                .default_height(156.0)
                .min_height(96.0)
                .max_height(300.0)
        } else {
            train_log_panel
                .resizable(false)
                .exact_height(46.0)
        };
        train_log_panel.show(ctx, |ui| {
            self.ui_train_log_panel(ui);
        });

        egui::SidePanel::left("left")
            .resizable(true)
            .default_width(324.0)
            .min_width(280.0)
            .max_width(500.0)
            .frame(Frame::none().fill(theme::SURFACE_DEEP))
            .show(ctx, |ui| {
                ui.add_space(10.0);
                Frame::default()
                    .fill(theme::SURFACE)
                    .inner_margin(egui::Margin::same(14.0))
                    .rounding(egui::Rounding::same(16.0))
                    .stroke(Stroke::new(1.0, theme::BORDER))
                    .show(ui, |ui| {
                        self.ui_sidebar(ui);
                    });
                ui.add_space(10.0);
            });

        egui::CentralPanel::default()
            .frame(Frame::none().fill(theme::SURFACE_DEEP))
            .show(ctx, |ui| {
                ui.add_space(4.0);
                Frame::default()
                    .fill(theme::SURFACE)
                    .inner_margin(egui::Margin::same(8.0))
                    .rounding(egui::Rounding::same(16.0))
                    .stroke(Stroke::new(1.0, theme::BORDER))
                    .show(ui, |ui| {
                        self.ui_canvas(ui, ctx);
                    });
                ui.add_space(4.0);
            });

        if self.show_label_window {
            let editing = self.label_edit_idx.is_some();
            let batch_n = self.pending_boxes_batch.len();
            let win_title: String = if editing {
                "修改标签".to_string()
            } else if batch_n > 0 {
                format!("输入标签（{batch_n} 个框 · 同一类别）")
            } else {
                "输入标签".to_string()
            };
            egui::Window::new(win_title)
                .collapsible(false)
                .resizable(false)
                .order(Order::Foreground)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(if editing {
                        "修改当前选中框的类别名称（可改为已有类名或新类名；确定后写入当前图的 .txt）："
                            .to_string()
                    } else if batch_n > 0 {
                        format!(
                            "连续柔性外接：本批共 {batch_n} 个检测框（每个闭合块对应一个外接框）。您在下方填写的类别名将统一赋给这 {batch_n} 个框，并写入当前图的 .txt："
                        )
                    } else {
                        "请输入当前框的类别名称（将写入与图片同目录的 .txt；训练包中的 data.yaml 在点击训练时生成）："
                            .to_string()
                    });
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.label(RichText::new("快速选择").strong());
                            if self.classes.is_empty() {
                                ui.label(
                                    RichText::new("（暂无已记录类别，可直接在下方输入新名称）")
                                        .weak()
                                        .small(),
                                );
                            } else {
                                ui.label(
                                    RichText::new("点击已有类别可快速填入；不影响左侧当前激活类别。")
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                );
                                ui.add_space(4.0);
                                egui::ScrollArea::vertical()
                                    .id_salt("quick_pick_classes")
                                    .max_height(86.0)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.horizontal_wrapped(|ui| {
                                            for (i, class_name) in self.classes.iter().enumerate() {
                                                let is_active_pick = self.label_draft.trim() == class_name.trim();
                                                let btn = egui::Button::new(class_name.as_str())
                                                    .fill(if is_active_pick {
                                                        color_alpha(theme::ACCENT, 42)
                                                    } else {
                                                        theme::SURFACE_SOFT
                                                    })
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        if i == self.active_class_idx {
                                                            theme::ACCENT
                                                        } else {
                                                            theme::BORDER_SUBTLE
                                                        },
                                                    ));
                                                if ui.add(btn).clicked() {
                                                    self.label_draft = class_name.clone();
                                                }
                                            }
                                        });
                                    });
                            }
                        });
                    });
                    let te = ui
                        .group(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("输入或重命名").strong());
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.label_draft)
                                        .desired_width(240.0)
                                        .id(label_draft_textedit_id()),
                                )
                            })
                            .inner
                        })
                        .inner;
                    let te_focused = te.has_focus();
                    let enter = ui.ctx().input(|i| i.key_pressed(Key::Enter));
                    let space = ui.ctx().input(|i| i.key_pressed(Key::Space));
                    let confirm_keys = if te_focused {
                        enter
                    } else {
                        enter || space
                    };
                    ui.horizontal(|ui| {
                        if ui.button("确定").clicked() || confirm_keys {
                            self.finalize_pending_with_label();
                        }
                        if ui.button("取消").clicked() {
                            self.pending_box = None;
                            self.pending_boxes_batch.clear();
                            self.label_edit_idx = None;
                            self.show_label_window = false;
                            self.draw_phase = DrawPhase::Idle;
                        }
                    });
                });
        }

        self.flush_class_log_if_dirty();
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title("YOLO 标注与训练器"),
        persist_window: false,
        persistence_path: None,
        ..Default::default()
    };
    eframe::run_native(
        "YOLO 标注与训练器",
        native_options,
        Box::new(|cc| Ok(Box::new(YoloTrainerApp::new(cc)))),
    )
}
