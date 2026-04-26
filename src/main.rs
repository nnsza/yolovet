#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod onnx_assist;

use egui::{
    widgets::color_picker::{color_picker_color32, show_color, show_color_at, Alpha},
    Align, Align2, Area, Color32, ComboBox, Context, CursorIcon, FontFamily, FontId, Frame, Id,
    Key, Label, Order, PointerButton, Pos2, Rect, Response, RichText, Sense, Shape, Stroke, Ui,
    UiKind, Vec2, WidgetInfo, WidgetType,
};
use image::{DynamicImage, GenericImageView, RgbaImage};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, Read, Write};
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

/// 与 `image_root` 同级、按行记录类别名。图片集固定为 `class_log.txt`；多路视频同目录时用 `class_log_{视频主名}.txt` 区分。第 1 条非注释行对应类别索引 0。
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

fn compact_metric_tile(ui: &mut Ui, width: f32, label: &str, value: &str, tint: Color32) {
    let w = width.max(1.0);
    Frame::default()
        .fill(color_alpha(tint, 20))
        .inner_margin(egui::Margin::symmetric(4.0, 3.0))
        .rounding(egui::Rounding::same(6.0))
        .stroke(Stroke::new(1.0, color_alpha(tint, 90)))
        .show(ui, |ui| {
            // 仅设 min 时内容会把芯片撑出均分格，与下方「打开当前路径」定宽行不对齐
            ui.set_min_width(w);
            ui.set_max_width(w);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.label(RichText::new(label).size(10.0).color(theme::TEXT_MUTED));
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center)
                        .with_main_align(egui::Align::Min)
                        .with_main_justify(true),
                    |ui| {
                        ui.add(
                            egui::Label::new(
                                RichText::new(value)
                                    .size(10.5)
                                    .strong()
                                    .color(tint),
                            )
                            .truncate(),
                        );
                    },
                );
            });
        });
}

fn training_backend_card(
    ui: &Ui,
    rect: Rect,
    response: &Response,
    id: Id,
    title: &str,
    subtitle: &str,
    accent: Color32,
    selected: bool,
) {
    let hover = ui
        .ctx()
        .animate_bool_responsive(id.with("hover"), response.hovered());
    let selected_t = ui
        .ctx()
        .animate_bool_responsive(id.with("selected"), selected);
    let down = ui.ctx().animate_bool_responsive(
        id.with("down"),
        response.is_pointer_button_down_on(),
    );
    if hover > 0.001 || selected_t > 0.001 || down > 0.001 {
        ui.ctx().request_repaint();
    }

    let painter = ui.painter();
    let draw_rect = rect.translate(Vec2::new(0.0, down * 1.0));
    painter.rect_filled(
        draw_rect.expand(1.5 + hover * 1.2 + selected_t * 1.6),
        10.0,
        color_alpha(accent, (8.0 + hover * 12.0 + selected_t * 24.0) as u8),
    );
    painter.rect_filled(
        draw_rect,
        9.0,
        if selected {
            Color32::from_rgb(35, 47, 56)
        } else {
            Color32::from_rgb(28, 34, 43)
        },
    );
    painter.rect_stroke(
        draw_rect,
        9.0,
        Stroke::new(
            1.0 + hover * 0.3 + selected_t * 0.55,
            color_alpha(
                if selected { accent } else { theme::BORDER_SUBTLE },
                (if selected { 170.0 } else { 132.0 } + hover * 24.0) as u8,
            ),
        ),
    );
    let dot = Pos2::new(draw_rect.left() + 13.0, draw_rect.center().y);
    painter.circle_filled(
        dot,
        4.3 + selected_t * 0.8,
        if selected {
            color_alpha(accent, 220)
        } else {
            color_alpha(theme::TEXT_MUTED, 96)
        },
    );
    painter.text(
        Pos2::new(draw_rect.left() + 24.0, draw_rect.top() + 12.0),
        Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(13.5),
        theme::TEXT,
    );
    painter.text(
        Pos2::new(draw_rect.left() + 24.0, draw_rect.bottom() - 12.0),
        Align2::LEFT_CENTER,
        subtitle,
        egui::FontId::proportional(10.5),
        if selected {
            color_alpha(accent, 230)
        } else {
            theme::TEXT_MUTED
        },
    );
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

    // 悬停判定：interact + 卡片响应 + 本层矩形命中（避免子控件挡掉 hover 导致特效不触发）。
    let header_hovered = header_resp.hovered()
        || header_inner.response.hovered()
        || ui
            .ctx()
            .rect_contains_pointer(ui.layer_id(), header_rect);

    // 悬停光效：平滑渐入 + 轻微脉冲；过渡结束后 egui 默认不再 request_repaint，时间动画会停住，故在显式播放时持续请求重绘。
    let hover_strength = ui.ctx().animate_bool_responsive(
        header_id.with("accordion_header_hover_fx"),
        header_hovered,
    );
    if hover_strength > 0.001 {
        let time = ui.input(|i| i.time as f32);
        let pulse = (time * 3.1).sin() * 0.5 + 0.5;
        let g = hover_strength * (0.62 + 0.38 * pulse);
        let painter = ui.painter();
        let rr = 14.0_f32;
        let halo_a = (g * (40.0 + pulse * 85.0)).clamp(0.0, 255.0) as u8;
        painter.rect_stroke(
            header_rect.expand(2.2 + pulse * 2.0),
            rr + 2.0,
            Stroke::new(0.9 + g * 1.1, color_alpha(accent, halo_a)),
        );
        let rim_a = (g * (175.0 + pulse * 65.0)).clamp(0.0, 255.0) as u8;
        painter.rect_stroke(
            header_rect,
            rr,
            Stroke::new(1.15 + g * (0.95 + pulse * 0.45), color_alpha(accent, rim_a)),
        );
        let sweep_w = (header_rect.width() * (0.22 + pulse * 0.18)).clamp(28.0, 120.0);
        let x0 = header_rect.left()
            - sweep_w
            + (time * 38.0).rem_euclid(header_rect.width() + sweep_w * 1.4);
        let sheen = Rect::from_min_max(
            Pos2::new(x0, header_rect.top() + 4.0),
            Pos2::new(x0 + sweep_w * 0.55, header_rect.bottom() - 4.0),
        )
        .intersect(header_rect);
        if sheen.width() > 2.0 && sheen.height() > 2.0 {
            painter.rect_filled(
                sheen,
                6.0,
                color_alpha(Color32::WHITE, (g * (18.0 + pulse * 16.0)) as u8),
            );
        }
        // animate_bool 仅在 0↔1 过渡期内 request_repaint；脉冲/扫光依赖 time，需持续重绘。
        ui.ctx().request_repaint();
    }

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
/// 柔性外接（E/F）与拉新框（R）时，跟随指针的十字虚线线宽（屏幕像素）。
const CROSSHAIR_STROKE: f32 = 2.85;

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
/// 画布缩放 ≤ 此比例（相对适应窗口）时进入环轨相册：已标注图环列、主图外拖动旋转（带惯性）。
const CAROUSEL_RING_ZOOM_MAX: f32 = 0.4;

/// 环轨相册浏览分组：已存在非空标签行的图 / 尚未标注的图。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum CarouselRingPool {
    #[default]
    Annotated,
    Unannotated,
}

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

/// ONNX 采纳：按模型类别 id 得到与辅助显示一致的名称（通常为 `unknown_{id}`），在列表中按**名称**查找；
/// 找不到则在**末尾**新建，不占用已有索引，避免与手动类别混在同一 id 上。
fn dataset_index_for_onnx_adopt_name(
    classes_acc: &mut Vec<String>,
    model_class_id: usize,
    assist_names: &[String],
) -> usize {
    let name = assist_names
        .get(model_class_id)
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("unknown_{}", model_class_id));
    if let Some(i) = classes_acc.iter().position(|s| s == &name) {
        return i;
    }
    let i = classes_acc.len();
    classes_acc.push(name);
    i
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
    /// 全局采纳过程中按 ONNX 名称在末尾追加类别后的完整 `class_log` 顺序（前缀与采纳前一致）。
    classes_after: Vec<String>,
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

/// 加载中文字体与 **Times New Roman**（`TNR_Brand` 族，专用于顶栏 `YoloVet` 品牌名）。
/// 若系统存在 `C:\Windows\Fonts\times.ttf` 等路径则返回 `true`，否则回退为默认无衬线。
fn setup_app_fonts(ctx: &Context) -> bool {
    let mut fonts = egui::FontDefinitions::default();
    let mut tnr_ready = false;
    for path in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\FONTS\times.ttf"] {
        if let Ok(bytes) = fs::read(path) {
            fonts.font_data.insert(
                "app_tnr".into(),
                egui::FontData::from_owned(bytes).into(),
            );
            fonts
                .families
                .entry(FontFamily::Name("TNR_Brand".into()))
                .or_default()
                .push("app_tnr".into());
            tnr_ready = true;
            break;
        }
    }
    for path in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ] {
        if let Ok(bytes) = fs::read(path) {
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
            break;
        }
    }
    ctx.set_fonts(fonts);
    tnr_ready
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

fn is_video_file(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            matches!(
                e.to_lowercase().as_str(),
                "mp4" | "avi" | "mov" | "mkv" | "webm" | "wmv" | "m4v" | "mpeg" | "mpg"
            )
        })
}

/// 单条视频可展开的最大帧数（避免内存中存百万级路径）。
const MAX_VIDEO_FRAMES: u32 = 200_000;

#[derive(Clone)]
struct VideoSession {
    /// 与 `stem_######.jpg` 同前缀的已净化主名，用于同目录多视频时区分 `class_log_{stem}.txt`。
    stem: String,
    frames: Vec<RgbaImage>,
}

#[derive(Clone, Copy)]
struct VideoProbe {
    frame_count: u32,
    fps: f32,
    width: u32,
    height: u32,
}

struct VideoLoadResult {
    video_path: PathBuf,
    out_dir: PathBuf,
    stem: String,
    fps: f32,
    frames: Vec<RgbaImage>,
}

enum VideoLoadMsg {
    Status { text: String, progress: f32 },
    Done(Result<VideoLoadResult, String>),
}

fn ffmpeg_binary_path() -> PathBuf {
    let name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(name);
            if cand.is_file() {
                return cand;
            }
        }
    }
    PathBuf::from(name)
}

fn ffprobe_binary_path() -> PathBuf {
    let name = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(name);
            if cand.is_file() {
                return cand;
            }
        }
    }
    PathBuf::from(name)
}

fn sanitize_file_stem(stem: &str) -> String {
    let mut s = String::new();
    for ch in stem.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ' ');
        s.push(if ok { ch } else { '_' });
    }
    let t = s.trim();
    if t.is_empty() {
        "video_frames".to_string()
    } else {
        t.chars().take(120).collect()
    }
}

fn parse_r_frame_rate(s: &str) -> Option<f32> {
    let t = s.trim();
    if let Some((a, b)) = t.split_once('/') {
        let n: f32 = a.trim().parse().ok()?;
        let d: f32 = b.trim().parse().ok()?;
        if d.abs() > 1e-6 {
            return Some(n / d);
        }
    }
    t.parse::<f32>().ok()
}

/// 用 ffprobe 取宽高、时长、帧率与帧数；没有可靠帧数时按时长 * 帧率估算。
fn probe_video_stream(path: &Path) -> Result<VideoProbe, String> {
    let ffprobe = ffprobe_binary_path();
    let mut cmd = Command::new(&ffprobe);
    command_hide_console(&mut cmd);
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
    ])
    .arg(path);
    let out = cmd
        .output()
        .map_err(|e| format!("无法运行 ffprobe（{ffprobe:?}）：{e}。请安装 FFmpeg 或将 ffprobe 放在 exe 同目录。"))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe 失败：{}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut dur_sec: Option<f32> = None;
    let mut fps: Option<f32> = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut nb_frames: Option<u32> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "width" => width = value.parse::<u32>().ok().filter(|v| *v > 0),
            "height" => height = value.parse::<u32>().ok().filter(|v| *v > 0),
            "r_frame_rate" => {
                if let Some(f) = parse_r_frame_rate(value) {
                    if f.is_finite() && f > 0.01 {
                        fps = Some(f);
                    }
                }
            }
            "duration" => {
                if let Ok(d) = value.parse::<f32>() {
                    if d.is_finite() && d > 0.0 {
                        dur_sec = Some(dur_sec.map_or(d, |a| a.max(d)));
                    }
                }
            }
            "nb_frames" => nb_frames = value.parse::<u32>().ok().filter(|v| *v > 0),
            _ => {}
        }
    }
    let fps = fps.unwrap_or(25.0).max(0.01);
    let width = width.ok_or_else(|| "ffprobe 未返回视频宽度".to_string())?;
    let height = height.ok_or_else(|| "ffprobe 未返回视频高度".to_string())?;
    let mut n = nb_frames.unwrap_or_else(|| {
        let dur = dur_sec.unwrap_or(0.0).max(0.0);
        ((dur * fps).ceil() as u32).max(1)
    });
    if n > MAX_VIDEO_FRAMES {
        n = MAX_VIDEO_FRAMES;
    }
    Ok(VideoProbe {
        frame_count: n,
        fps,
        width,
        height,
    })
}

fn decode_video_frames_rgba(
    src: &Path,
    probe: VideoProbe,
    progress_tx: Option<&mpsc::Sender<VideoLoadMsg>>,
) -> Result<Vec<RgbaImage>, String> {
    let frame_bytes = (probe.width as usize)
        .checked_mul(probe.height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "视频帧尺寸过大，无法分配内存".to_string())?;
    let ffmpeg = ffmpeg_binary_path();
    let mut cmd = Command::new(&ffmpeg);
    command_hide_console(&mut cmd);
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
    ])
    .arg(src)
    .args([
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-frames:v",
        &probe.frame_count.to_string(),
        "-",
    ]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("无法运行 ffmpeg（{ffmpeg:?}）：{e}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "无法读取 ffmpeg 输出".to_string())?;
    let mut frames = Vec::with_capacity(probe.frame_count as usize);
    let progress_step = (probe.frame_count / 100).max(1);
    for frame_idx in 0..probe.frame_count {
        let mut buf = vec![0_u8; frame_bytes];
        match stdout.read_exact(&mut buf) {
            Ok(()) => {
                let img = RgbaImage::from_raw(probe.width, probe.height, buf)
                    .ok_or_else(|| "解析 raw RGBA 帧失败".to_string())?;
                frames.push(img);
                let frame_no = frame_idx + 1;
                if frame_no == probe.frame_count || frame_no % progress_step == 0 {
                    if let Some(tx) = progress_tx {
                        let progress = 0.08 + 0.88 * (frame_no as f32 / probe.frame_count as f32);
                        let _ = tx.send(VideoLoadMsg::Status {
                            text: format!("正在读取完整帧：{frame_no}/{}", probe.frame_count),
                            progress: progress.clamp(0.0, 0.98),
                        });
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("读取视频帧失败：{e}")),
        }
    }
    drop(stdout);
    let out = child
        .wait_with_output()
        .map_err(|e| format!("等待 ffmpeg 结束失败：{e}"))?;
    if !out.status.success() && frames.is_empty() {
        return Err(format!(
            "ffmpeg 解码失败：{}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    if frames.is_empty() {
        return Err("视频没有解码出任何帧".to_string());
    }
    Ok(frames)
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

/// 同路径下 **存在** 与图对应的 `.txt` 文件（含空文件；用于与「无文件=未标」区分）。
fn path_label_file_exists(image_path: &Path) -> bool {
    label_txt_path_for_image(image_path).is_file()
}

/// 同路径下存在**至少一行**非注释、非空内容（不区分是否为有效 YOLO 框行，与旧版「有写东西」一致）。
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

/// 负样本：已有 `.txt` 且**无数值标签行**（全空或仅 # 备注），与「未创建 txt」相区别。
fn path_is_negative_label_only(image_path: &Path) -> bool {
    let lbl = label_txt_path_for_image(image_path);
    if !lbl.is_file() {
        return false;
    }
    !path_has_nonempty_label_file(image_path)
}

/// 仅所选目录当前层级内的图片文件（不进入子文件夹），与顶栏「选择图片目录」语义一致。
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

const SHORTCUTS_HINT_BLUE: Color32 = Color32::from_rgb(110, 185, 255);
const SHORTCUTS_SECTION_MAGENTA: Color32 = Color32::from_rgb(200, 150, 255);
const SHORTCUTS_SECTION_ORANGE: Color32 = Color32::from_rgb(255, 165, 100);

/// 快捷键面板内：等宽键名色块
fn shortcut_kbd_chip(ui: &mut Ui, key_text: &str, key_tint: Color32) {
    Frame::default()
        .fill(color_alpha(key_tint, 32))
        .inner_margin(egui::Margin::symmetric(6.0, 3.0))
        .rounding(5.0)
        .stroke(Stroke::new(1.0, color_alpha(key_tint, 170)))
        .show(ui, |ui| {
            ui.label(
                RichText::new(key_text)
                    .monospace()
                    .strong()
                    .size(12.0)
                    .color(key_tint),
            );
        });
}

/// 分节小标题 + 色条
fn shortcut_section_header(ui: &mut Ui, title: &str, line: Color32) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(8.0, 0.0);
        let (r, _gal) = ui.allocate_exact_size(Vec2::new(3.0, 18.0), Sense::hover());
        ui.painter().rect_filled(r, 2.0, line);
        ui.label(
            RichText::new(title)
                .strong()
                .size(14.5)
                .color(theme::TEXT),
        );
    });
    ui.add_space(4.0);
}

/// 多枚键帽 + 正文（`desc` 支持换行）
fn shortcut_explain_line(ui: &mut Ui, keys: &[(&str, Color32)], desc: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(4.0, 5.0);
        for (k, tint) in keys {
            shortcut_kbd_chip(ui, k, *tint);
        }
        ui.add_space(2.0);
        ui.label(
            RichText::new(desc)
                .size(12.5)
                .line_height(Some(18.5))
                .color(color_alpha(theme::TEXT, 250)),
        );
    });
    ui.add_space(3.0);
}

/// 画布与工具条：完整快捷键下拉说明（深色底、分色分节）
fn canvas_shortcuts_help_popup(ui: &mut Ui) {
    let a = theme::ACCENT;
    let ok = theme::OK;
    let warn = theme::WARN;
    let danger = theme::DANGER;
    let muted = theme::TEXT_MUTED;
    let popup_id = ui.id().with("canvas_shortcuts_help");
    let open = ui.memory(|m| m.is_popup_open(popup_id));
    let chevron = if open { "  ▲" } else { "  ▼" };
    let btn_fill = color_alpha(SHORTCUTS_HINT_BLUE, 20);
    let btn_st = color_alpha(SHORTCUTS_HINT_BLUE, 190);
    let btn_resp = ui
        .add_sized(
            [228.0, 31.0],
            egui::Button::new(
                RichText::new(format!("查看完整快捷键{chevron}"))
                    .size(15.0)
                    .strong()
                    .color(SHORTCUTS_HINT_BLUE),
            )
            .frame(true)
            .fill(btn_fill)
            .stroke(Stroke::new(1.2, btn_st)),
        );
    if btn_resp.clicked() {
        ui.memory_mut(|m| m.toggle_popup(popup_id));
    }

    if !ui.memory(|m| m.is_popup_open(popup_id)) {
        return;
    }

    let area_response = Area::new(popup_id)
        .kind(UiKind::Picker)
        .order(Order::Foreground)
        .fixed_pos(btn_resp.rect.left_bottom() + Vec2::new(0.0, 6.0))
        .show(ui.ctx(), |ui| {
            Frame::default()
                .fill(theme::SURFACE)
                .stroke(Stroke::new(1.5, color_alpha(a, 180)))
                .inner_margin(egui::Margin::same(14.0))
                .rounding(14.0)
                .show(ui, |ui| {
                    // 顶部标题带渐变感：深色条 + 彩色字
                    let header_bar = color_alpha(SHORTCUTS_SECTION_MAGENTA, 16);
                    Frame::default()
                        .fill(header_bar)
                        .inner_margin(egui::Margin::symmetric(10.0, 8.0))
                        .rounding(8.0)
                        .stroke(Stroke::new(1.0, color_alpha(SHORTCUTS_SECTION_MAGENTA, 90)))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("全局快捷键总览")
                                    .strong()
                                    .size(17.0)
                                    .color(SHORTCUTS_SECTION_MAGENTA),
                            );
                            ui.label(
                                RichText::new("与深色界面配色区分：蓝=导航/视口 · 绿=平移/确认 · 黄/橙=模式 · 红/粉=危险/命名 · 下表与当前代码一致")
                                    .size(11.0)
                                    .line_height(Some(16.0))
                                    .color(muted),
                            );
                        });
                    ui.add_space(6.0);

                    egui::ScrollArea::vertical()
                        .max_height(500.0)
                        .auto_shrink([false, true])
                        .id_salt("canvas_shortcuts_help_scroll")
                        .show(ui, |ui| {
                            ui.set_max_width(560.0);
                            ui.spacing_mut().item_spacing = Vec2::new(6.0, 0.0);

                            shortcut_section_header(
                                ui,
                                "顶栏与切图（A / D）",
                                SHORTCUTS_HINT_BLUE,
                            );
                            shortcut_explain_line(
                                ui,
                                &[("A", a), ("D", a)],
                                "上一张 / 下一张图片或视频帧。在「标签窗未打开且其它控件未抢夺键盘」时生效；将鼠标悬停在大图画布上时，即其它处正在输入，也可 A/D 切图。与顶栏「上一张A」「下一张D」等效。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[],
                                "图集模式：底栏/顶栏的帧进度条可点击或拖动跳帧，白条=当前、红点区段=已存标注的帧。",
                            );

                            shortcut_section_header(
                                ui,
                                "视口与缩放",
                                a,
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Ctrl", a),
                                    ("＋ 滚轮", a),
                                ],
                                "在画布上「按住 Control（或 macOS 上的 Cmd）再滚动」：缩放；光标附近保持位置感。无修饰键的滚轮用于下方「有选中框时切类别」。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Space", ok),
                                ],
                                "按住空格 且 鼠标在画布上：平移主图（抓手光标）。若正在输入、标签弹窗、或 E/F 涂鸦，空格可能被占用。",
                            );

                            shortcut_section_header(ui, "模式开关（R / E / F，全局无文本焦点时）", warn);
                            shortcut_explain_line(
                                ui,
                                &[("R", a)],
                                "「矩形两点框」模式启停。开启时左键在图上点第一点、再点第二点完成一框，随后弹出标签命名。再按 R 为关闭。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[("E", warn)],
                                "「柔性外接」：单笔画一笔成一个框。再按 E 关闭。若曾打开 R 会先协调退出矩形二点、清空选中。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[("F", warn)],
                                "「连续柔性外接」（F 套圈可保留外周等；具体合并规则以画布逻辑为准。）再按 F 关。E/F 与 R 互斥。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[("Esc", muted)],
                                "在多种状态下取消：如标签名输入窗可关闭、退出 E/F 的笔画准备、正在等矩形第二点时清空等（未打开标签窗时，Esc 不关闭训练日志等其它面板）。",
                            );

                            shortcut_section_header(
                                ui,
                                "有选中框时 · 视口内",
                                ok,
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("滚轮", ok),
                                ],
                                "未按 Ctrl、未按空格 且 已选中有框 且 侧栏有类别时：滚轮循环切换该框的类别（有撤销入栈）。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("W", a),
                                    ("S", a),
                                ],
                                "在图上某像素有多框重叠、且未在拖拽时，以当前鼠标位置在重叠栈中 W 向上 / S 向下 切换选中的那一框；之後若指针在屏幕上未明显移动，则 双击 会改 W/S 选到的那一框，若已明显移动或仅缩放则双击仍从叠层最上格命中。",
                            );

                            shortcut_section_header(
                                ui,
                                "删除与撤销",
                                danger,
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Delete", danger),
                                ],
                                "删除当前选中的框（若可删）并保存标签。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Q", danger),
                                ],
                                "在未打开标签命名窗 且 其它处未吃键盘 时，删除当前选中框。若打开的是标签窗，见下方「Q = 仅取消本次命名」。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Ctrl+Z", ok),
                                ],
                                "或 macOS：Cmd+Z，按撤销栈回退一步（有焦点限制：标签窗开时 R/E/F/Ctrl+Z 由全局与输入争用，请优先在画布操作）。",
                            );

                            shortcut_section_header(
                                ui,
                                "标签命名小窗",
                                SHORTCUTS_SECTION_ORANGE,
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Enter", ok),
                                ],
                                "在「输入或重命名」文本框有焦点 时 确认；无焦点 时 空格/回车 均可作「确定」等效。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Space", ok),
                                ],
                                "文本框未聚焦时，可与 Enter 同作「确定」。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[
                                    ("Q", danger),
                                    ("Esc", muted),
                                ],
                                "取消此次命名/改名（新框不落地；已有框的修改可取消）。Q 在输入框内会当普通字符，请先失焦。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[],
                                "在图上 双击 某已有框的填充区域：直接打开本窗编辑该类别的文字标签。",
                            );

                            shortcut_section_header(ui, "鼠标 · 与右键", SHORTCUTS_SECTION_ORANGE);
                            shortcut_explain_line(
                                ui,
                                &[],
                                "右键 在空白处 或 部分状态：可退出 E/F/涂鸦/矩形等中间状态、关闭 R 的第一笔、在空白 取消全选 等（不点到框/柄上时）。与 `secondary_click` 分支一致。",
                            );
                            shortcut_explain_line(
                                ui,
                                &[],
                                "左键拖拽：移动框、拉角/边缩放。环轨/视频等界面另有「单击缩略图」「条上拖动」等提示，见主界面。",
                            );

                            shortcut_section_header(
                                ui,
                                "环轨相册（主图缩很多时）",
                                a,
                            );
                            shortcut_explain_line(
                                ui,
                                &[],
                                "在环形缩略外区域 拖动 旋转；点某张 与 A/D 一样切换当前图。主图内仍遵守框编辑逻辑。",
                            );

                            ui.add_space(8.0);
                            Frame::default()
                                .fill(color_alpha(danger, 18))
                                .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                                .rounding(6.0)
                                .stroke(Stroke::new(1.0, color_alpha(danger, 100)))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("关闭本表")
                                            .size(12.0)
                                            .color(color_alpha(theme::TEXT, 220)),
                                    );
                                    ui.horizontal(|ui| {
                                        shortcut_kbd_chip(ui, "Esc", a);
                                        ui.add_space(4.0);
                                        ui.label(
                                            RichText::new("或 点击 空白/其它处")
                                                .size(12.0)
                                                .color(muted),
                                        );
                                    });
                                });
                        });
                });
        })
        .response;

    if !btn_resp.clicked()
        && (ui.input(|i| i.key_pressed(Key::Escape)) || area_response.clicked_elsewhere())
    {
        ui.memory_mut(|m| m.close_popup());
    }
}

/// 从 YOLO 标签文件统计「不同 class 列数」与「有效框行数」（不读图像尺寸）。
fn label_txt_class_box_counts(path: &Path) -> Option<(usize, usize)> {
    let Ok(text) = fs::read_to_string(path) else {
        return None;
    };
    let mut seen = HashSet::new();
    let mut n = 0usize;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let cid: usize = t.split_whitespace().next()?.parse().ok()?;
        seen.insert(cid);
        n += 1;
    }
    Some((seen.len(), n))
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
    let mut skipped_no_training_labels = 0usize;

    for src_img in image_paths {
        let Ok(img) = image::open(src_img) else {
            skipped_no_training_labels += 1;
            continue;
        };
        let (w, h) = img.dimensions();
        let src_lbl = label_txt_path_for_image(src_img);
        if load_annotations(&src_lbl, w, h).is_empty() {
            skipped_no_training_labels += 1;
            continue;
        }

        let rel_raw = path_relative_to(src_img, image_root);
        let rel_img = strip_first_dir_if(&rel_raw, "images");
        let dst_img = img_out.join(&rel_img);
        copy_file_create_parent(src_img, &dst_img).map_err(|e| {
            format!("复制图片失败 {} -> {}: {e}", src_img.display(), dst_img.display())
        })?;

        let rel_lbl_raw = path_relative_to(&src_lbl, image_root);
        let rel_lbl = strip_first_dir_if(&rel_lbl_raw, "labels");
        let dst_lbl = lbl_out.join(&rel_lbl);
        copy_file_create_parent(&src_lbl, &dst_lbl).map_err(|e| {
            format!("复制标签失败 {}: {e}", src_lbl.display())
        })?;
        labels_copied += 1;
    }

    if skipped_no_training_labels > 0 {
        let _ = tx.send(TrainMsg::Line(format!(
            "[准备] 已跳过 {skipped_no_training_labels} 张未进入训练包的图片（无法打开或无有效标注行）。"
        )));
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
    conda_env_custom_root: String,
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
    /// 标签窗内按 Q 取消后，本帧跳过画布上 Q 删除选中框（避免与「取消改名」同键冲突）。
    suppress_bbox_q_delete_once: bool,

    image_paths: Vec<PathBuf>,
    /// 视频逐帧模式：源视频与帧率；仅当用户保存过带框标注时，对应帧才写入与视频同目录的 jpg + txt。
    video_session: Option<VideoSession>,
    current_index: usize,
    rgba: Option<RgbaImage>,
    image_texture: Option<egui::TextureHandle>,
    texture_dirty: bool,
    annotations: Vec<Bbox>,
    draw_phase: DrawPhase,
    selected: Option<usize>,
    drag: Option<(DragKind, usize)>, // kind, bbox index
    /// W/S 在重叠栈中切框后、若指针在屏幕上几乎未动则双击改该 `target`；元组存当时指针的屏幕 `Pos2`，避免仅缩放时误清。
    stack_nav_dblclk_lock: Option<(Pos2, usize, usize)>,

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
    video_load_rx: Option<Receiver<VideoLoadMsg>>,
    video_load_busy: bool,
    video_load_status: String,
    video_load_progress: f32,

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
    /// ONNX 辅助检测：`nms=False` 时在 Rust 内做 NMS 的 IoU 阈值（`nms=True` 的模型不经过此路径）。
    assist_onnx_iou: f32,

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

    /// 训练 epoch 数。
    train_epochs: u32,
    train_epoch_scroll_accum: f32,

    undo_stack: Vec<UndoSnapshot>,
    /// 正在应用撤销时不入栈。
    undo_suspend: bool,

    /// 顶部「已标注」条：需重建时为 true（刷新列表、写入标签后）。
    annotated_strip_dirty: bool,
    /// `image_paths` 下标，对应磁盘上已有非空 .txt 的图。
    annotated_strip_indices: Vec<usize>,
    /// `image_paths` 下标：无有效标签行（无文件或空/仅注释）的图。
    unannotated_strip_indices: Vec<usize>,
    /// 与 `annotated_strip_indices` 对齐：每条 `(不同类别数, 框数)`，用于已标注列表备注。
    annotated_strip_summaries: Vec<(usize, usize)>,
    /// 与 `annotated_strip_indices` 等长：该条为「仅空 txt 负样本」时为 `true`。
    annotated_strip_is_neg: Vec<bool>,
    /// 全库统计：有框的图数、仅负样本的图数（顶栏/进度条说明用）。
    dataset_n_with_boxes: usize,
    dataset_n_neg_only: usize,
    /// 顶栏「上一张A」「下一张D」点击后触发描边/光晕强度 0~1，逐帧衰减。
    image_nav_btn_fx: [f32; 2],
    /// 仅上/下一张步进为 true 时，下一帧在顶栏已标注列表中把当前行 `scroll_to_me` 一次；点列表/其它跳转不设，避免手滚时回弹。
    annotated_strip_step_scroll_pending: bool,
    /// 顶栏帧进度条：桶数（≤ 图数），与 `image_paths` 等长映射；红=有框，蓝=仅负样本。
    image_nav_progress_bucket_n: usize,
    image_nav_progress_bucket_has_box: Vec<bool>,
    image_nav_progress_bucket_has_neg: Vec<bool>,

    /// `class_log.txt` 需在下一帧写回磁盘。
    class_log_dirty: bool,
    /// 已做过首次从当前 `image_root` 读取 class_log（避免重复覆盖）。
    class_log_bootstrapped: bool,

    /// 环轨相册（缩放较小时）：环方位角（rad）。
    carousel_ring_angle: f32,
    /// 角速度（rad/s），松手后指数衰减。
    carousel_ring_vel: f32,
    /// 本次拖动是否从主图矩形外按下（避免与框拖拽冲突）。
    carousel_ring_drag_from_outer: bool,
    carousel_ring_tex: HashMap<usize, egui::TextureHandle>,
    /// 环轨相册当前查看：已标注集 / 未标注集。
    carousel_ring_pool: CarouselRingPool,
    /// 顶栏品牌 logo（启动时从固定路径尝试加载，失败时仅无图组合）。
    top_bar_logo: Option<egui::TextureHandle>,
    /// 已注册 Windows `times.ttf` 为 `TNR_Brand`（供品牌名用）。
    brand_tnr: bool,
}

impl YoloTrainerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let brand_tnr = setup_app_fonts(&cc.egui_ctx);
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

        let top_bar_logo = (|| {
            let p = std::path::Path::new(r"C:\Users\78672\Pictures\lll.png");
            let img = image::open(p).ok()?;
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            if w == 0 || h == 0 {
                return None;
            }
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_raw(),
            );
            Some(cc.egui_ctx.load_texture(
                "top_bar_logo",
                color_image,
                egui::TextureOptions::LINEAR,
            ))
        })();

        Self {
            image_root: PathBuf::from("."),
            conda_env_paths: Vec::new(),
            conda_env_idx: 0,
            conda_env_custom_root: default_conda_env_path(),
            conda_env_list_bootstrapped: false,
            use_builtin_cpu_train: false,
            // 1 = 类别管理，2 = 训练配置（原「数据集」折叠栏已移除）
            sidebar_open_section: Some(1),
            sidebar_width: 344.0,
            model_preset: ModelPreset::Yolo11n,
            classes: Vec::new(),
            class_colors: Vec::new(),
            label_draft: String::new(),
            show_label_window: false,
            pending_box: None,
            label_edit_idx: None,
            suppress_bbox_q_delete_once: false,
            image_paths: Vec::new(),
            video_session: None,
            current_index: 0,
            rgba: None,
            image_texture: None,
            texture_dirty: false,
            annotations: Vec::new(),
            draw_phase: DrawPhase::Idle,
            selected: None,
            drag: None,
            stack_nav_dblclk_lock: None,
            handles_anim_sel: None,
            corner_hover_radius_anim: [0.0; 4],
            edge_hover_anim: [0.0; 4],
            train_log: Vec::new(),
            train_log_expanded: false,
            training: false,
            train_rx: None,
            training_pid: None,
            training_stop_pending: false,
            video_load_rx: None,
            video_load_busy: false,
            video_load_status: String::new(),
            video_load_progress: 0.0,
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
            assist_onnx_iou: onnx_assist::ASSIST_ONNX_IOU,
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
            unannotated_strip_indices: Vec::new(),
            annotated_strip_summaries: Vec::new(),
            annotated_strip_is_neg: Vec::new(),
            dataset_n_with_boxes: 0,
            dataset_n_neg_only: 0,
            image_nav_btn_fx: [0.0; 2],
            annotated_strip_step_scroll_pending: false,
            image_nav_progress_bucket_n: 0,
            image_nav_progress_bucket_has_box: Vec::new(),
            image_nav_progress_bucket_has_neg: Vec::new(),
            class_log_dirty: false,
            class_log_bootstrapped: false,
            carousel_ring_angle: 0.0,
            carousel_ring_vel: 0.0,
            carousel_ring_drag_from_outer: false,
            carousel_ring_tex: HashMap::new(),
            carousel_ring_pool: CarouselRingPool::Annotated,
            top_bar_logo,
            brand_tnr,
        }
    }

    /// 品牌名 YoloVet 使用的字体（Times New Roman 或回退为 UI 无衬线）。
    fn brand_name_font(&self, size: f32) -> FontId {
        if self.brand_tnr {
            FontId::new(size, FontFamily::Name("TNR_Brand".into()))
        } else {
            FontId::proportional(size)
        }
    }

    /// 写入/撤销等使用的类别表路径。图片集：`class_log.txt`；视频集：`class_log_{stem}.txt`（同目录多视频不共用一名）。
    fn class_log_path(&self) -> PathBuf {
        if let Some(vs) = &self.video_session {
            self.image_root
                .join(format!("class_log_{}.txt", &vs.stem))
        } else {
            self.image_root.join(CLASS_LOG_FILENAME)
        }
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

    /// 从磁盘重载 `classes`：切换数据根或视频后应调用；无文件时清空（与选图片目录时一致，不会残留上一次的类别）。
    fn load_class_log_from_disk(&mut self) {
        self.classes.clear();
        self.class_colors.clear();
        // 读路径：图片模式仅 `class_log.txt`；视频模式先 `class_log_{stem}.txt`，若无则回退到同目录 `class_log.txt`（旧版习惯）
        let path = if let Some(vs) = &self.video_session {
            let a = self
                .image_root
                .join(format!("class_log_{}.txt", &vs.stem));
            if a.is_file() {
                a
            } else {
                let b = self.image_root.join(CLASS_LOG_FILENAME);
                if b.is_file() { b } else { self.active_class_idx = 0; return; }
            }
        } else {
            let p = self.image_root.join(CLASS_LOG_FILENAME);
            if !p.is_file() {
                self.active_class_idx = 0;
                return;
            }
            p
        };
        let Ok(text) = fs::read_to_string(&path) else {
            self.active_class_idx = 0;
            return;
        };
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

    fn resolved_conda_root(&self) -> String {
        let custom = self.conda_env_custom_root.trim();
        if !custom.is_empty() {
            return custom.to_string();
        }
        self.conda_env_paths
            .get(self.conda_env_idx)
            .cloned()
            .unwrap_or_default()
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
        let was_video_mode = self.video_session.is_some();
        if self.image_root == new_root {
            if was_video_mode {
                let _ = self.save_current_labels();
                self.cancel_assist_tasks();
                self.reset_canvas_runtime_cache();
                self.annotated_strip_indices.clear();
                self.unannotated_strip_indices.clear();
                self.video_session = None;
                self.current_index = 0;
            }
            self.refresh_image_list();
            return;
        }
        let _ = self.save_current_labels();
        self.cancel_assist_tasks();
        self.reset_canvas_runtime_cache();
        self.annotated_strip_indices.clear();
        self.unannotated_strip_indices.clear();
        self.video_session = None;
        self.current_index = 0;
        self.image_root = new_root;
        self.refresh_image_list();
    }

    /// 载入视频：从第 0 帧起可浏览；仅当保存带框标注时写入与视频同目录的 jpg + txt。
    fn load_video_session(&mut self, video_path: PathBuf) {
        self.cancel_assist_tasks();
        if !video_path.is_file() || !is_video_file(&video_path) {
            self.train_log
                .push("[视频] 请选择一个视频文件（如 mp4、mov、mkv、webm）。".to_string());
            return;
        }
        if self.video_load_busy {
            self.train_log
                .push("[视频] 当前已有视频正在载入，请等待完成。".to_string());
            return;
        }
        let _ = self.save_current_labels();
        let (tx, rx) = mpsc::channel();
        self.video_load_rx = Some(rx);
        self.video_load_busy = true;
        self.video_load_progress = 0.01;
        self.video_load_status = "正在读取视频信息...".to_string();
        self.train_log
            .push(format!("[视频] 开始载入 {}", video_path.display()));

        thread::spawn(move || {
            let result = (|| -> Result<VideoLoadResult, String> {
                let _ = tx.send(VideoLoadMsg::Status {
                    text: "正在读取视频信息...".to_string(),
                    progress: 0.02,
                });
                let probe = probe_video_stream(&video_path)?;
                let _ = tx.send(VideoLoadMsg::Status {
                    text: format!(
                        "准备读取完整帧：{} 帧，{}x{} @ {:.2} fps",
                        probe.frame_count, probe.width, probe.height, probe.fps
                    ),
                    progress: 0.06,
                });
                let frames = decode_video_frames_rgba(&video_path, probe, Some(&tx))?;
                if frames.is_empty() {
                    return Err("视频没有解码出任何帧".to_string());
                }
                let out_dir = video_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
                let stem = video_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(sanitize_file_stem)
                    .unwrap_or_else(|| "video_frames".to_string());
                Ok(VideoLoadResult {
                    video_path,
                    out_dir,
                    stem,
                    fps: probe.fps,
                    frames,
                })
            })();
            let _ = tx.send(VideoLoadMsg::Done(result));
        });
    }

    fn refresh_image_list(&mut self) {
        self.cancel_assist_tasks();
        self.reset_canvas_runtime_cache();
        self.mark_annotated_strip_dirty();
        self.annotated_strip_indices.clear();
        self.unannotated_strip_indices.clear();
        if self.video_session.is_none() {
            // `load_class_log_from_disk` 会先清空再读盘（与换目录/无文件时一致）
            self.load_class_log_from_disk();
            self.image_paths = if self.image_root.is_dir() {
                gather_images_for_dataset_root(&self.image_root)
            } else {
                Vec::new()
            };
        }
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
        // 已标栏：有 `.txt` 即视为已处理（含空 txt 的负样本）；无 txt 为未标
        self.annotated_strip_indices = self
            .image_paths
            .iter()
            .enumerate()
            .filter(|(_, p)| path_label_file_exists(p))
            .map(|(i, _)| i)
            .collect();
        self.unannotated_strip_indices = self
            .image_paths
            .iter()
            .enumerate()
            .filter(|(_, p)| !path_label_file_exists(p))
            .map(|(i, _)| i)
            .collect();
        self.annotated_strip_summaries = self
            .annotated_strip_indices
            .iter()
            .map(|&idx| {
                self.image_paths
                    .get(idx)
                    .and_then(|p| label_txt_class_box_counts(&label_txt_path_for_image(p)))
                    .unwrap_or((0, 0))
            })
            .collect();
        self.annotated_strip_is_neg = self
            .annotated_strip_indices
            .iter()
            .filter_map(|&idx| self.image_paths.get(idx))
            .map(|p| path_is_negative_label_only(p))
            .collect();
        self.dataset_n_with_boxes = self
            .image_paths
            .iter()
            .filter(|p| path_has_nonempty_label_file(p))
            .count();
        self.dataset_n_neg_only = self
            .image_paths
            .iter()
            .filter(|p| path_is_negative_label_only(p))
            .count();
        self.carousel_ring_tex.clear();

        let n = self.image_paths.len();
        const MAX_BUCKETS: usize = 1200;
        if n == 0 {
            self.image_nav_progress_bucket_n = 0;
            self.image_nav_progress_bucket_has_box.clear();
            self.image_nav_progress_bucket_has_neg.clear();
        } else {
            let bucket_n = n.min(MAX_BUCKETS);
            let mut has_box = vec![false; bucket_n];
            let mut has_neg = vec![false; bucket_n];
            for (idx, p) in self.image_paths.iter().enumerate() {
                let bi = (idx * bucket_n / n).min(bucket_n.saturating_sub(1));
                if path_has_nonempty_label_file(p) {
                    has_box[bi] = true;
                } else if path_is_negative_label_only(p) {
                    has_neg[bi] = true;
                }
            }
            self.image_nav_progress_bucket_n = bucket_n;
            self.image_nav_progress_bucket_has_box = has_box;
            self.image_nav_progress_bucket_has_neg = has_neg;
        }
    }

    #[inline]
    fn carousel_ring_indices(&self) -> &[usize] {
        match self.carousel_ring_pool {
            CarouselRingPool::Annotated => &self.annotated_strip_indices,
            CarouselRingPool::Unannotated => &self.unannotated_strip_indices,
        }
    }

    fn set_carousel_ring_pool(&mut self, pool: CarouselRingPool) {
        if self.carousel_ring_pool == pool {
            return;
        }
        self.carousel_ring_pool = pool;
        self.carousel_ring_angle = 0.0;
        self.carousel_ring_vel = 0.0;
        self.carousel_ring_drag_from_outer = false;
        self.carousel_ring_tex.clear();
    }

    #[inline]
    fn carousel_ring_active(&self) -> bool {
        self.view_zoom <= CAROUSEL_RING_ZOOM_MAX
            && !self.show_label_window
            && !self.image_paths.is_empty()
    }

    fn carousel_ring_items(&self, inner: Rect) -> Vec<(f32, usize, Rect)> {
        let indices = self.carousel_ring_indices();
        let n = indices.len();
        if n == 0 {
            return Vec::new();
        }
        let cx = inner.center().x;
        let cy = inner.center().y + inner.height() * 0.05;
        let r_min = inner.width().min(inner.height());
        let scale_ui = (r_min / 520.0).clamp(0.66, 1.48);
        let n_f = n as f32;
        let tau = std::f32::consts::TAU;
        // 大图集：整库 n 张只在「角度」上滚动浏览，环上只放 vis 个槽位，半径按槽位数取疏密度，始终在视口附近可见。
        // 转一整圈 (2π) 对应滚过 n 张图，避免 n=1000 时半径大到整环都在屏外。
        let vis_cap = ((r_min / 24.0).round() as usize).clamp(16, 42);
        let vis = n.min(vis_cap).max(1);
        let vis_f = vis as f32;
        let sin_v = (std::f32::consts::PI / vis_f).max(1e-4).sin().max(1e-3);
        let want_gap = (92.0 + 0.05 * n_f.min(800.0)) * scale_ui;
        let ring_r_base = r_min * 0.415;
        let ring_r_layout = ring_r_base.max(want_gap / (2.0 * sin_v));
        let max_side_est = (30.0 + 188.0) * scale_ui;
        let spread_max = 0.84_f32 + 0.34_f32;
        let pad = 12.0_f32;
        let half_span = inner.width().min(inner.height()) * 0.5 - pad - max_side_est * 0.55;
        let ring_cap = (half_span / spread_max).max(r_min * 0.38);
        let ring_r = ring_r_layout.min(ring_cap);
        let scroll = (-self.carousel_ring_angle / tau) * n_f;
        let mut items: Vec<(f32, usize, Rect)> = Vec::with_capacity(vis);
        for j in 0..vis {
            let ang = self.carousel_ring_angle + tau * (j as f32) / vis_f;
            let (s, c) = ang.sin_cos();
            let depth = (1.0 + c) * 0.5;
            let t = (depth.clamp(0.0, 1.0)).powf(1.2);
            let smooth = t * t * (3.0 - 2.0 * t);
            let spread = 0.84 + 0.34 * smooth;
            let px = cx + ring_r * s * spread;
            let py = cy - ring_r * (0.48 + 0.12 * smooth) * c - r_min * 0.028 * smooth;
            let crowding = (n_f / 120.0).powf(0.12).max(1.0).recip();
            let side = (30.0 + 188.0 * smooth) * scale_ui * crowding;
            let aspect = 0.66 + 0.14 * smooth;
            let rect = Rect::from_center_size(Pos2::new(px, py), Vec2::new(side, side * aspect));
            let idx_f = scroll + (j as f32) * (n_f / vis_f);
            let m = n as f32;
            let u = idx_f.rem_euclid(m);
            let pick = (u.floor() as usize).min(n.saturating_sub(1));
            let strip_idx = indices[pick];
            items.push((depth, strip_idx, rect));
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        items
    }

    fn carousel_ring_step(
        &mut self,
        ctx: &Context,
        response: &Response,
        inner: Rect,
        dt: f32,
        pointer: Option<Pos2>,
        primary_pressed: bool,
        primary_released: bool,
        dragging: bool,
        space_down: bool,
    ) {
        if !self.carousel_ring_active() {
            self.carousel_ring_vel = 0.0;
            self.carousel_ring_drag_from_outer = false;
            return;
        }
        if space_down || self.scribble_kind.is_some() {
            return;
        }
        // 整块画布（含主图区域）均可拖动环轨；缩略图点击切图由 `carousel_ring_try_pick` 处理。
        let ptr_steering = pointer.map(|p| inner.contains(p)).unwrap_or(false);
        if primary_pressed && response.hovered() && ptr_steering {
            self.carousel_ring_drag_from_outer = true;
        }
        if primary_released {
            self.carousel_ring_drag_from_outer = false;
        }
        let sens = 0.011_f32;
        let dt = dt.max(1e-4);
        if dragging && self.carousel_ring_drag_from_outer {
            let d = response.drag_delta();
            let da = -d.x * sens;
            self.carousel_ring_angle += da;
            self.carousel_ring_vel = da / dt;
        } else if !ctx.input(|i| i.pointer.primary_down()) {
            self.carousel_ring_angle += self.carousel_ring_vel * dt;
            // 略慢的指数衰减 + 低速时额外阻尼，惯性更顺、停得更柔。
            let v = self.carousel_ring_vel.abs();
            let decay = (-3.35_f32 * dt).exp() * if v < 0.045 { 0.88 + 0.12 * (v / 0.045) } else { 1.0 };
            self.carousel_ring_vel *= decay;
            if self.carousel_ring_vel.abs() < 0.0019 {
                self.carousel_ring_vel = 0.0;
            }
        }
        // 持续刷新：惯性滑行与正面卡片光晕呼吸。
        ctx.request_repaint();
        if self.carousel_ring_active()
            && ptr_steering
            && response.hovered()
            && !space_down
            && self.scribble_kind.is_none()
        {
            ctx.set_cursor_icon(if dragging && self.carousel_ring_drag_from_outer {
                CursorIcon::Grabbing
            } else {
                CursorIcon::Grab
            });
        }
    }

    fn carousel_ring_try_pick(&mut self, ctx: &Context, inner: Rect, pointer: Option<Pos2>) {
        if !self.carousel_ring_active() {
            return;
        }
        if !ctx.input(|i| i.pointer.button_clicked(PointerButton::Primary)) {
            return;
        }
        let Some(p) = pointer else {
            return;
        };
        if !inner.contains(p) {
            return;
        }
        let items = self.carousel_ring_items(inner);
        for (_, strip_idx, rect) in items.into_iter().rev() {
            if rect.contains(p) {
                if strip_idx != self.current_index {
                    let _ = self.save_current_labels();
                    self.current_index = strip_idx;
                    self.load_current_image();
                }
                ctx.request_repaint();
                break;
            }
        }
    }

    fn carousel_ring_load_thumb(&mut self, ctx: &Context, path_idx: usize) {
        if self.carousel_ring_tex.contains_key(&path_idx) {
            return;
        }
        let Some(path) = self.image_paths.get(path_idx) else {
            return;
        };
        let Ok(img) = image::open(path) else {
            return;
        };
        let rgba = img.thumbnail(320, 320).to_rgba8();
        let (w, h) = rgba.dimensions();
        if w == 0 || h == 0 {
            return;
        }
        let color_image =
            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw());
        let tex = ctx.load_texture(
            format!("carousel_ring_{path_idx}"),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.carousel_ring_tex.insert(path_idx, tex);
    }

    fn paint_carousel_ring(&mut self, ctx: &Context, painter: &egui::Painter, inner: Rect) {
        const HINT_DASH: f32 = 6.0;
        const HINT_GAP: f32 = 5.0;
        let hint_r = inner.shrink(10.0);
        let hint_stroke = Stroke::new(
            1.25,
            color_alpha(theme::ACCENT, 72),
        );
        let mn = hint_r.min;
        let mx = hint_r.max;
        let top = [Pos2::new(mn.x, mn.y), Pos2::new(mx.x, mn.y)];
        let right = [Pos2::new(mx.x, mn.y), Pos2::new(mx.x, mx.y)];
        let bottom = [Pos2::new(mx.x, mx.y), Pos2::new(mn.x, mx.y)];
        let left = [Pos2::new(mn.x, mx.y), Pos2::new(mn.x, mn.y)];
        for seg in [&top[..], &right[..], &bottom[..], &left[..]] {
            for s in Shape::dashed_line(seg, hint_stroke, HINT_DASH, HINT_GAP) {
                painter.add(s);
            }
        }
        let items = self.carousel_ring_items(inner);
        if items.is_empty() {
            let msg = match self.carousel_ring_pool {
                CarouselRingPool::Annotated => "当前分组暂无已标注图片",
                CarouselRingPool::Unannotated => "当前分组暂无未标注图片",
            };
            painter.text(
                inner.center(),
                Align2::CENTER_CENTER,
                msg,
                egui::FontId::proportional(15.0),
                theme::TEXT_MUTED,
            );
            painter.text(
                inner.left_bottom() + Vec2::new(8.0, -48.0),
                Align2::LEFT_BOTTOM,
                "请用底部按钮切换「已标注 / 未标注」· 虚线内可拖动旋转",
                egui::FontId::proportional(12.0),
                theme::TEXT_MUTED,
            );
            return;
        }
        let mut budget = 24usize;
        for &(_, strip_idx, _) in &items {
            if budget == 0 {
                break;
            }
            if !self.carousel_ring_tex.contains_key(&strip_idx) {
                self.carousel_ring_load_thumb(ctx, strip_idx);
                budget -= 1;
            }
        }
        let t_anim = ctx.input(|i| i.time) as f32;
        let breathe = 0.5 + 0.5 * (t_anim * 1.85).sin();
        for (depth, strip_idx, rect) in items {
            let smooth_d = (depth.clamp(0.0, 1.0)).powf(1.15);
            let alpha =
                (58 + (200.0 * ((smooth_d + 0.08) / 1.08).min(1.0)) as i32).clamp(0, 255) as u8;
            let round = 6.0 + 5.0 * smooth_d;
            // 阴影：仅中前层，随深度加重
            if smooth_d > 0.22 {
                let sh_off = Vec2::new(3.2 + smooth_d * 2.0, 5.0 + smooth_d * 3.5);
                let sh_alpha = (18.0 + 55.0 * smooth_d.powf(1.1)) as u8;
                painter.rect_filled(
                    rect.translate(sh_off),
                    round,
                    Color32::from_rgba_unmultiplied(0, 0, 0, sh_alpha),
                );
            }
            let fill = Color32::from_rgba_unmultiplied(16, 20, 30, alpha);
            painter.rect_filled(rect, round, fill);
            let stroke_w = 1.15 + 1.35 * smooth_d;
            painter.rect_stroke(
                rect,
                round,
                Stroke::new(
                    stroke_w,
                    color_alpha(theme::ACCENT, (f32::from(alpha) * (0.5 + 0.22 * smooth_d)) as u8),
                ),
            );
            // 最靠前的一张：呼吸光晕 + 内高光边
            if smooth_d > 0.88 {
                let glow = (34.0 + 38.0 * breathe * smooth_d) as u8;
                painter.rect_stroke(
                    rect.expand(2.5 + 1.2 * breathe),
                    round + 2.0,
                    Stroke::new(1.25 + 0.85 * breathe, color_alpha(theme::ACCENT, glow)),
                );
                let hi = color_alpha(Color32::WHITE, (22.0 + 28.0 * breathe) as u8);
                painter.rect_stroke(
                    rect.shrink(1.5),
                    (round - 1.0).max(2.0),
                    Stroke::new(1.0, hi),
                );
            }
            let luma = (0.36 + 0.64 * smooth_d.powf(0.82)).clamp(0.0, 1.0);
            let tint_g = (255.0 * luma) as u8;
            let img_tint = Color32::from_rgb(tint_g, tint_g, tint_g);
            if let Some(tex) = self.carousel_ring_tex.get(&strip_idx) {
                let pad = 2.5 + 0.8 * (1.0 - smooth_d);
                let img_rect = rect.shrink(pad);
                painter.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    img_tint,
                );
            } else {
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "…",
                    egui::FontId::proportional(14.0),
                    theme::TEXT_MUTED,
                );
            }
        }
        painter.text(
            inner.left_bottom() + Vec2::new(8.0, -48.0),
            Align2::LEFT_BOTTOM,
            "大图集：环上为一段窗口，拖动旋转在整库中滚动（转一圈≈滚过一遍列表）· 底部切换已标/未标 · 点击切图",
            egui::FontId::proportional(12.0),
            theme::TEXT_MUTED,
        );
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
        self.annotated_strip_step_scroll_pending = true;
        self.load_current_image();
    }

    fn go_next_image(&mut self) {
        if self.image_paths.is_empty() || self.current_index + 1 >= self.image_paths.len() {
            return;
        }
        let _ = self.save_current_labels();
        self.current_index += 1;
        self.annotated_strip_step_scroll_pending = true;
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
        let rgba = if let Some(vs) = &self.video_session {
            let Some(frame) = vs.frames.get(self.current_index) else {
                self.train_log.push(format!(
                    "[视频] 内存帧索引越界：{}/{}",
                    self.current_index + 1,
                    vs.frames.len()
                ));
                return;
            };
            frame.clone()
        } else {
            let Ok(img) = image::open(path) else {
                return;
            };
            img.to_rgba8()
        };
        let (w, h) = rgba.dimensions();
        self.rgba = Some(rgba);

        let lbl = label_txt_path_for_image(path);
        self.annotations = load_annotations(&lbl, w, h);
        self.texture_dirty = true;

        let classes_len_before = self.classes.len();
        // 仅当当前图已有标注行时才扩展类别表，避免「尚未标注」就自动出现 unknown_0。
        if let Some(max_ann_id) = self.annotations.iter().map(|b| b.class_id).max() {
            while self.classes.len() <= max_ann_id {
                let k = self.classes.len();
                self.classes.push(format!("unknown_{}", k));
                self.class_colors.push(palette_color(k));
            }
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
        let use_in_memory = self.video_session.is_some();
        let mem_rgba = if use_in_memory {
            self.rgba.clone()
        } else {
            None
        };
        let path_for_disk = if !use_in_memory {
            self.image_paths.get(self.current_index).cloned()
        } else {
            None
        };
        if use_in_memory {
            if mem_rgba.is_none() {
                return;
            }
        } else {
            let Some(p) = path_for_disk.as_ref() else {
                return;
            };
            if !p.is_file() {
                return;
            }
        }
        let _ = self.assist_rx.take();
        let (tx, rx) = mpsc::channel();
        self.assist_rx = Some(rx);
        self.assist_busy = true;
        let conf_min = self.assist_onnx_conf.clamp(0.0, 1.0);
        let iou_nms = self.assist_onnx_iou.clamp(0.0, 1.0);
        thread::spawn(move || {
            let r = {
                let mut ses = match arc.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        let _ = tx.send(Err("ONNX 会话锁失败".to_string()));
                        return;
                    }
                };
                if use_in_memory {
                    let Some(ref img) = mem_rgba else {
                        let _ = tx.send(Err("内部：无内存图像".to_string()));
                        return;
                    };
                    onnx_assist::predict_with_session_rgba(&mut *ses, img, conf_min, iou_nms)
                } else {
                    let Some(p) = path_for_disk else {
                        let _ = tx.send(Err("内部：无路径".to_string()));
                        return;
                    };
                    onnx_assist::predict_with_session(&mut *ses, &p, conf_min, iou_nms)
                }
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

    /// 将当前 ONNX 辅助框写入正式标注：若与任一已有框 IoU≥[`ASSIST_ADOPT_DUP_IOU`] 则跳过（保留原框）；否则追加。
    /// 类别名按辅助栏中的 `unknown_*`（与模型 id 对应）在 `class_log` **末尾**独占新行，不占用已有手动类别索引。
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
        let mut candidates = build_adopt_candidates(
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
        let classes_len_before = self.classes.len();
        let assist_names = self.assist_class_names.clone();
        for c in &mut candidates {
            c.class_id = dataset_index_for_onnx_adopt_name(
                &mut self.classes,
                c.class_id,
                &assist_names,
            );
        }
        let (merged, added, skipped_iou) =
            adopt_merge_candidates(self.annotations.clone(), candidates);
        self.annotations = merged;

        while self.class_colors.len() < self.classes.len() {
            let k = self.class_colors.len();
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
        let iou_nms = self.assist_onnx_iou.clamp(0.0, 1.0);
        let mask = self.assist_pred_class_on.clone();
        let names_len = self.assist_class_names.len();
        let assist_names = self.assist_class_names.clone();
        let initial_classes = self.classes.clone();
        let (tx, rx) = mpsc::channel();
        self.assist_batch_rx = Some(rx);
        self.assist_batch_busy = true;
        let _ = self.assist_rx.take();
        self.assist_busy = false;

        thread::spawn(move || {
            let mut classes_acc = initial_classes;
            let mut images_scanned = 0usize;
            let mut images_open_failed = 0usize;
            let mut infer_failed = 0usize;
            let mut total_added = 0usize;
            let mut total_skipped_iou = 0usize;

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
                    onnx_assist::predict_with_session(&mut *ses, &path, conf, iou_nms)
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
                let mut candidates = build_adopt_candidates(&preds, &mask, names_len, w, h);
                for c in &mut candidates {
                    c.class_id = dataset_index_for_onnx_adopt_name(
                        &mut classes_acc,
                        c.class_id,
                        &assist_names,
                    );
                }
                let (merged, added, skipped) = adopt_merge_candidates(existing, candidates);
                total_added += added;
                total_skipped_iou += skipped;
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
                classes_after: classes_acc,
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
                let old_classes = self.classes.clone();
                let old_colors = self.class_colors.clone();
                self.classes = sum.classes_after;
                self.class_colors.resize(self.classes.len(), Color32::TRANSPARENT);
                for i in 0..self.classes.len() {
                    self.class_colors[i] = if i < old_colors.len() && i < old_classes.len()
                        && old_classes.get(i) == self.classes.get(i)
                    {
                        old_colors[i]
                    } else {
                        palette_color(i)
                    };
                }
                if self.classes != old_classes {
                    self.mark_class_log_dirty();
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

    fn active_class_label_draft(&self) -> String {
        self.classes
            .get(self.active_class_idx.min(self.classes.len().saturating_sub(1)))
            .cloned()
            .unwrap_or_default()
    }

    /// 当前图无框时，写入**空**与图同名的 `.txt`，作为 YOLO 负样本；若已有实框或已是负样本则不起作用。
    fn mark_current_image_as_negative_sample(&mut self) -> std::io::Result<()> {
        if !self.annotations.is_empty() {
            return Ok(());
        }
        let Some(p) = self.image_paths.get(self.current_index) else {
            return Ok(());
        };
        if path_has_nonempty_label_file(p) {
            return Ok(());
        }
        if path_is_negative_label_only(p) {
            return Ok(());
        }
        let txt = label_txt_path_for_image(p);
        if let Some(dir) = txt.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(&txt, "")?;
        self.mark_annotated_strip_dirty();
        Ok(())
    }

    fn save_current_labels(&mut self) -> std::io::Result<()> {
        let Some(img) = &self.rgba else {
            return Ok(());
        };
        let (w, h) = img.dimensions();
        let Some(img_path) = self.image_paths.get(self.current_index) else {
            return Ok(());
        };
        let lbl = label_txt_path_for_image(img_path);
        if self.video_session.is_some() {
            if self.annotations.is_empty() {
                // 负样本：仅空 .txt，不删，保留训练时的「无目标」
                if !path_is_negative_label_only(img_path) {
                    let _ = fs::remove_file(&lbl);
                }
                let _ = fs::remove_file(img_path);
            } else {
                if let Some(parent) = img_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                DynamicImage::ImageRgba8(img.clone())
                    .save(img_path)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                save_annotations(&lbl, &self.annotations, w, h)?;
            }
        } else {
            save_annotations(&lbl, &self.annotations, w, h)?;
        }
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

    /// 关闭标签窗：编辑已有框时仅取消本次改名；新框待命名时直接丢弃未确认的框。
    fn cancel_label_dialog(&mut self) {
        if let Some(idx) = self.label_edit_idx.take() {
            if idx < self.annotations.len() {
                let cid = self.annotations[idx].class_id;
                self.label_draft = self
                    .classes
                    .get(cid)
                    .cloned()
                    .unwrap_or_default();
            }
        } else {
            self.pending_box = None;
            self.pending_boxes_batch.clear();
            self.scribble_closed_boxes.clear();
            self.scribble_active = false;
            self.scribble_points.clear();
            self.scribble_open_start = 0;
            self.label_draft = self.active_class_label_draft();
        }
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

    fn overlap_stack_indices_at(&self, p_img: (f32, f32)) -> Vec<usize> {
        let mut hits: Vec<usize> = self
            .annotations
            .iter()
            .enumerate()
            .filter_map(|(idx, b)| Self::hit_inside(p_img, b).then_some(idx))
            .collect();
        hits.sort_by(|&a, &b| {
            let area_a = (self.annotations[a].max_x - self.annotations[a].min_x)
                * (self.annotations[a].max_y - self.annotations[a].min_y);
            let area_b = (self.annotations[b].max_x - self.annotations[b].min_x)
                * (self.annotations[b].max_y - self.annotations[b].min_y);
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        hits
    }

    fn cycle_overlap_selection_at(&mut self, p_img: (f32, f32), toward_inner: bool) -> bool {
        let stack = self.overlap_stack_indices_at(p_img);
        if stack.is_empty() {
            return false;
        }
        let next_idx = match self
            .selected
            .and_then(|selected| stack.iter().position(|&idx| idx == selected))
        {
            Some(pos) if toward_inner => stack[(pos + 1).min(stack.len() - 1)],
            Some(pos) if pos > 0 => stack[pos - 1],
            Some(_) => stack[0],
            None => stack[0],
        };
        self.selected = Some(next_idx);
        self.drag = None;
        true
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
                let root = self.resolved_conda_root();
                if root.is_empty() {
                    self.train_log.push(
                        "[错误] 请先在「训练环境设定」中选择 Conda 环境（或点击刷新环境列表）。"
                            .to_string(),
                    );
                    return;
                }
                let py = conda_python_executable(Path::new(&root));
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

    fn poll_video_load(&mut self, ctx: &Context) {
        let Some(rx) = self.video_load_rx.as_ref() else {
            return;
        };
        let mut done: Option<Result<VideoLoadResult, String>> = None;
        let mut changed = false;
        while let Ok(msg) = rx.try_recv() {
            changed = true;
            match msg {
                VideoLoadMsg::Status { text, progress } => {
                    self.video_load_status = text;
                    self.video_load_progress = progress.clamp(0.0, 0.99);
                }
                VideoLoadMsg::Done(result) => done = Some(result),
            }
        }

        if let Some(result) = done {
            self.video_load_rx = None;
            self.video_load_busy = false;
            self.video_load_progress = 1.0;
            match result {
                Ok(result) => {
                    let n_frames = result.frames.len();
                    let mut paths = Vec::with_capacity(n_frames);
                    for i in 0..n_frames {
                        paths.push(result.out_dir.join(format!(
                            "{}_{:06}.jpg",
                            result.stem,
                            i + 1
                        )));
                    }
                    self.image_root = result.out_dir.clone();
                    self.video_session = Some(VideoSession {
                        stem: result.stem.clone(),
                        frames: result.frames,
                    });
                    // 同目录可有多路视频，类别表按 `class_log_{stem}.txt` 区分；与选图片目录一样从磁盘恢复
                    self.load_class_log_from_disk();
                    self.image_paths = paths;
                    self.current_index = 0;
                    self.mark_annotated_strip_dirty();
                    self.reset_canvas_runtime_cache();
                    self.load_current_image();
                    self.video_load_status = "载入完成".to_string();
                    self.train_log.push(format!(
                        "[视频] 已载入 {} · 共 {} 帧 @ {:.2} fps。同目录输出 {}_######.jpg + .txt；仅保存过标注的帧会落盘。",
                        result.video_path.display(),
                        n_frames,
                        result.fps,
                        result.stem,
                    ));
                }
                Err(e) => {
                    self.video_load_status = "载入失败".to_string();
                    self.train_log.push(format!("[视频] {e}"));
                }
            }
            changed = true;
        }

        if changed || self.video_load_busy {
            ctx.request_repaint();
        }
    }

    /// 顶栏中部：上一张A / 下一张D / 刷新、进度与统计芯片（当前文件名在左侧已标注标题行）。
    fn ui_top_bar_image_nav(&mut self, ui: &mut Ui) {
        let total_images = self.image_paths.len();
        let current_idx = if total_images == 0 {
            0
        } else {
            self.current_index + 1
        };
        Frame::default()
            .fill(theme::SURFACE_ELEVATED)
            .inner_margin(egui::Margin::same(8.0))
            .rounding(egui::Rounding::same(10.0))
            .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
                // 本卡可用宽；子块用同一 block_w 居中，使上排三键/四统计/下两格+开路径 左缘右缘对齐，且略短于全宽。
                let row_w = ui.available_width().max(1.0);
                let block_w = (row_w * 0.86).min(row_w).max(100.0);
                let h_pad = ((row_w - block_w) * 0.5).max(0.0);
                ui.horizontal(|ui| {
                    if h_pad > 0.0 {
                        ui.add_space(h_pad);
                    }
                    ui.vertical(|ui| {
                        ui.set_width(block_w);
                        let btn_gap = 6.0_f32;
                        let action_w = ((block_w - 2.0 * btn_gap) / 3.0).max(40.0);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(btn_gap, 0.0);
                            for v in &mut self.image_nav_btn_fx {
                                *v *= 0.84;
                                if *v < 0.008 {
                                    *v = 0.0;
                                }
                            }
                            if self.image_nav_btn_fx[0] + self.image_nav_btn_fx[1] > 0.002 {
                                ui.ctx().request_repaint();
                            }
                            let prev = ui
                                .add_enabled(
                                    self.current_index > 0,
                                    egui::Button::new(
                                        RichText::new("上一张A").small().strong(),
                                    )
                                    .min_size(Vec2::new(action_w, 24.0)),
                                )
                                .on_hover_text("切到上一张图片 / 上一帧");
                            if prev.clicked() {
                                self.image_nav_btn_fx[0] = 1.0;
                                self.go_prev_image();
                            }

                            let next = ui
                                .add_enabled(
                                    !self.image_paths.is_empty()
                                        && self.current_index + 1
                                            < self.image_paths.len(),
                                    egui::Button::new(
                                        RichText::new("下一张D").small().strong(),
                                    )
                                    .min_size(Vec2::new(action_w, 24.0)),
                                )
                                .on_hover_text("切到下一张图片 / 下一帧");
                            if next.clicked() {
                                self.image_nav_btn_fx[1] = 1.0;
                                self.go_next_image();
                            }

                            let refresh = ui
                                .add_sized(
                                    Vec2::new(action_w, 24.0),
                                    egui::Button::new(
                                        RichText::new("刷新列表").small().strong(),
                                    ),
                                )
                                .on_hover_text("重新扫描目录，并刷新顶部已标注文件列表");
                            if refresh.clicked() {
                                self.refresh_image_list();
                            }

                            let t0 = self.image_nav_btn_fx[0];
                            if t0 > 0.001 {
                                let g = 1.0 + 2.5 * t0;
                                ui.painter().rect_stroke(
                                    prev.rect.expand(g),
                                    10.0,
                                    Stroke::new(
                                        0.75 + 1.8 * t0,
                                        color_alpha(
                                            theme::ACCENT,
                                            (12.0 + 185.0 * t0) as u8,
                                        ),
                                    ),
                                );
                            }
                            let t1 = self.image_nav_btn_fx[1];
                            if t1 > 0.001 {
                                let g = 1.0 + 2.5 * t1;
                                ui.painter().rect_stroke(
                                    next.rect.expand(g),
                                    10.0,
                                    Stroke::new(
                                        0.75 + 1.8 * t1,
                                        color_alpha(
                                            theme::OK,
                                            (12.0 + 185.0 * t1) as u8,
                                        ),
                                    ),
                                );
                            }
                        });
                        // 与 block_w 右缘对齐：芯片内层 max_w 之外还有 Frame 左右内边距(各 4)，
                        // 必须摊进总宽，否则会整体比下方「图片目录/开路径」多出一截。
                        let stat_gap = 4.0_f32;
                        let frame_hpad = 4.0_f32 * 2.0; // 每枚 `compact_metric_tile` 左+右内边距
                        let stat_w = (block_w
                            - stat_gap * 3.0
                            - 4.0 * frame_hpad)
                            * 0.25;
                        let stat_w = stat_w.max(1.0);
                        ui.scope(|ui| {
                            ui.set_max_width(block_w);
                            ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = stat_gap;
                            compact_metric_tile(
                                ui,
                                stat_w,
                                "进度",
                                &format!("{current_idx}/{total_images}"),
                                theme::ACCENT,
                            );
                            compact_metric_tile(
                                ui,
                                stat_w,
                                "本图",
                                &format!("{} 框", self.annotations.len()),
                                theme::OK,
                            );
                            compact_metric_tile(
                                ui,
                                stat_w,
                                "已标注",
                                &self.annotated_strip_indices.len().to_string(),
                                theme::WARN,
                            );
                            compact_metric_tile(
                                ui,
                                stat_w,
                                "类别",
                                &self.classes.len().to_string(),
                                theme::ACCENT,
                            );
                            });
                        });
                        ui.add_space(2.0);
                        self.ui_source_picker_pair(ui, block_w, 26.0);
                    });
                    if h_pad > 0.0 {
                        ui.add_space(h_pad);
                    }
                });
            });
    }

    fn ui_source_picker_pair(&mut self, ui: &mut Ui, width: f32, height: f32) {
        let width = width.min(ui.available_width().max(1.0)).max(120.0);
        let row_gap = 6.0_f32;
        let col_gap = 6.0_f32.min((width * 0.06).max(4.0));
        let total_h = height * 2.0 + row_gap;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, total_h), Sense::hover());
        let half_w = ((width - col_gap) * 0.5).max(1.0);
        let left_rect = Rect::from_min_size(rect.min, Vec2::new(half_w, height));
        let right_rect = Rect::from_min_size(
            Pos2::new(left_rect.right() + col_gap, rect.top()),
            Vec2::new(half_w, height),
        );
        let open_rect = Rect::from_min_size(
            Pos2::new(rect.left(), rect.top() + height + row_gap),
            Vec2::new(width, height),
        );

        let image_mode_active = self.video_session.is_none() && !self.video_load_busy;
        let video_mode_active = self.video_session.is_some() || self.video_load_busy;
        let current_mode_accent = if video_mode_active {
            Color32::from_rgb(94, 164, 255)
        } else {
            Color32::from_rgb(120, 218, 145)
        };
        let can_open_path = self.image_root.is_dir()
            && (self.video_session.is_some()
                || !self.image_paths.is_empty()
                || self.image_root.to_string_lossy() != ".");

        let time_s = ui.input(|i| i.time) as f32;
        let entry_attention = self.image_paths.is_empty()
            && self.video_session.is_none()
            && !self.video_load_busy;
        // 已选过图/视频/正在载入 后才启用 W/S 式动效；冷启动未加载时两键不播特效
        let has_media = !self.image_paths.is_empty()
            || self.video_session.is_some()
            || self.video_load_busy;
        let left_fx = has_media && (entry_attention || image_mode_active);
        let right_fx = has_media && (entry_attention || video_mode_active);
        if left_fx || right_fx {
            ui.ctx().request_repaint();
        }
        let img_tip = if entry_attention {
            "【由此开始】选文件夹加载图片。\n载入所选目录内一层图片；若该层无图但存在 data.yaml，则按 train/val 子目录加载。"
        } else {
            "载入所选目录内一层图片；若该层无图但存在 data.yaml，则按 train/val 子目录加载。"
        };
        let vid_tip = if entry_attention {
            "【由此开始】选视频做逐帧标注。\n逐帧标注视频：需 FFmpeg。仅保存过标注的帧会写入同目录 jpg + txt。"
        } else {
            "逐帧标注视频：需 FFmpeg。仅保存过标注的帧会写入同目录 jpg + txt。"
        };

        let img_id = ui.id().with("pick_image_dir_segment");
        let vid_id = ui.id().with("pick_video_file_segment");
        let open_id = ui.id().with("open_current_root_segment");
        let img_resp = ui
            .interact(left_rect, img_id, Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text(img_tip);
        let vid_resp = ui
            .interact(right_rect, vid_id, Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text(vid_tip);
        let mut open_resp = ui.interact(
            open_rect,
            open_id,
            if can_open_path {
                Sense::click()
            } else {
                Sense::hover()
            },
        );
        open_resp = if can_open_path {
            open_resp
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(format!("打开当前路径：{}", self.image_root.display()))
        } else {
            open_resp.on_hover_text("先选择图片目录或视频文件")
        };

        // 图片目录 / 视频：紫底/蓝底；文字黑粗；动效见 has_media
        Self::paint_source_picker_button(
            ui,
            left_rect,
            &img_resp,
            img_id,
            "图片目录",
            Color32::from_rgb(88, 48, 120),
            Color32::from_rgb(190, 150, 255),
            image_mode_active,
            time_s,
            left_fx,
            0.0_f32,
        );
        Self::paint_source_picker_button(
            ui,
            right_rect,
            &vid_resp,
            vid_id,
            "视频文件",
            Color32::from_rgb(48, 81, 132),
            Color32::from_rgb(112, 172, 255),
            video_mode_active,
            time_s,
            right_fx,
            1.1_f32,
        );
        Self::paint_source_aux_button(
            ui,
            open_rect,
            &open_resp,
            open_id,
            "打开当前路径",
            current_mode_accent,
            can_open_path,
        );

        if img_resp.clicked() {
            if let Some(p) = rfd::FileDialog::new().pick_folder() {
                self.switch_image_root(p);
            }
        }
        if vid_resp.clicked() {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter(
                    "视频",
                    &["mp4", "avi", "mov", "mkv", "webm", "wmv", "m4v", "mpeg", "mpg"],
                )
                .pick_file()
            {
                self.load_video_session(p);
            }
        }
        if can_open_path && open_resp.clicked() {
            self.open_current_root_in_explorer();
        }
    }

    fn open_current_root_in_explorer(&mut self) {
        let target = if self.image_root.is_dir() {
            self.image_root.clone()
        } else if let Some(parent) = self.image_root.parent() {
            parent.to_path_buf()
        } else {
            return;
        };

        #[cfg(windows)]
        let mut cmd = {
            let mut c = Command::new("explorer");
            c.arg(&target);
            c
        };
        #[cfg(not(windows))]
        let mut cmd = {
            let mut c = Command::new("xdg-open");
            c.arg(&target);
            c
        };
        command_hide_console(&mut cmd);
        if let Err(err) = cmd.spawn() {
            self.train_log.push(format!(
                "[路径] 打开失败：{} ({err})",
                target.display()
            ));
        }
    }

    fn paint_source_picker_button(
        ui: &Ui,
        rect: Rect,
        response: &Response,
        id: Id,
        text: &str,
        fill: Color32,
        accent: Color32,
        selected: bool,
        time_s: f32,
        fx_active: bool,
        entry_phase: f32,
    ) {
        let hover = ui
            .ctx()
            .animate_bool_responsive(id.with("hover"), response.hovered());
        let down = ui.ctx().animate_bool_responsive(
            id.with("down"),
            response.is_pointer_button_down_on(),
        );
        let selected_t = ui
            .ctx()
            .animate_bool_responsive(id.with("selected"), selected);
        if hover > 0.001 || down > 0.001 || selected_t > 0.001 || fx_active {
            ui.ctx().request_repaint();
        }

        let att_breathe = if fx_active {
            0.5 + 0.5 * (time_s * 2.1 + entry_phase).sin()
        } else {
            0.0
        };
        let att_flicker = if fx_active {
            0.5 + 0.5 * (time_s * 3.4 + entry_phase * 1.3).cos()
        } else {
            0.0
        };

        let painter = ui.painter();
        let draw_rect = rect.translate(Vec2::new(0.0, down * 1.0));
        if fx_active {
            // 双层呼吸外晕
            let wob = 3.2 * att_breathe;
            for (k, a_mul) in [(0.0_f32, 1.0), (0.5 * std::f32::consts::PI, 0.6)] {
                let ph = 0.5 + 0.5 * (time_s * 1.9 + entry_phase + k).sin();
                let er = 2.0 + wob + 4.5 * (1.0 - ph) * a_mul;
                let ring = draw_rect.expand2(Vec2::new(er, er * 0.7));
                painter.rect_filled(
                    ring,
                    10.0,
                    color_alpha(
                        accent,
                        ((9.0 + 38.0 * ph * a_mul) * a_mul) as u8,
                    ),
                );
            }
        }
        let glow = draw_rect.expand2(Vec2::new(1.5 + hover * 1.8, 1.0 + hover * 0.8));
        painter.rect_filled(
            glow,
            9.0,
            color_alpha(
                accent,
                (10.0
                    + hover * 18.0
                    + selected_t * 34.0
                    + if fx_active {
                        8.0 + 22.0 * att_breathe
                    } else {
                        0.0
                    }) as u8,
            ),
        );

        let base = if selected {
            Color32::from_rgb(
                fill.r().saturating_add(8),
                fill.g().saturating_add(8),
                fill.b().saturating_add(8),
            )
        } else {
            fill
        };
        painter.rect_filled(draw_rect, 8.0, base);
        if fx_active {
            painter.with_clip_rect(draw_rect).rect_filled(
                draw_rect,
                8.0,
                color_alpha(
                    Color32::WHITE,
                    (6.0 + 9.0 * att_flicker) as u8,
                ),
            );
        }
        painter.rect_filled(
            draw_rect,
            8.0,
            color_alpha(accent, (14.0 + hover * 10.0 + selected_t * 26.0) as u8),
        );
        if fx_active {
            // 扫过高光带
            let s = (time_s * 0.55 + entry_phase * 0.08).rem_euclid(1.0);
            let w = draw_rect.width();
            let sheen_l = draw_rect.left() - 0.25 * w + s * (w * 1.5);
            let sheen = Rect::from_min_max(
                Pos2::new(sheen_l, draw_rect.top() + 1.0),
                Pos2::new(
                    (sheen_l + w * 0.3).min(draw_rect.right() + 4.0),
                    draw_rect.bottom() - 1.0,
                ),
            );
            painter.with_clip_rect(draw_rect).rect_filled(
                sheen,
                7.0,
                color_alpha(Color32::from_rgb(255, 255, 255), (7 + (18.0 * att_breathe) as u8) as u8),
            );
        }
        painter.rect_filled(
            draw_rect,
            8.0,
            color_alpha(Color32::BLACK, (8.0 + down * 24.0) as u8),
        );
        let stroke_w = 1.0
            + hover * 0.35
            + selected_t * 0.75
            + if fx_active {
                0.4 + 0.55 * att_breathe
            } else {
                0.0
            };
        let stroke_bright = 96.0
            + hover * 34.0
            + selected_t * 84.0
            + if fx_active {
                12.0 + 55.0 * att_flicker
            } else {
                0.0
            };
        painter.rect_stroke(
            draw_rect,
            8.0,
            Stroke::new(
                stroke_w,
                color_alpha(accent, stroke_bright as u8),
            ),
        );
        if fx_active {
            // 外缘霓虹
            let neon = draw_rect.expand(1.2 + 1.5 * (0.5 + 0.5 * (time_s * 2.0 + entry_phase).sin()));
            let na = (28.0 + 80.0 * (0.5 + 0.5 * (time_s * 2.8 + entry_phase * 0.5).sin())) as u8;
            painter.rect_stroke(
                neon,
                8.0,
                Stroke::new(1.0, color_alpha(accent, na)),
            );
        }

        // 单选式标识：仅当前模式为「本键」时显示在文字前；未选中的模式不画
        let text_left = if selected {
            let r_center = Pos2::new(draw_rect.left() + 10.0, draw_rect.center().y);
            let r_outer = 4.2 + selected_t * 0.3;
            painter.circle_filled(
                r_center,
                r_outer,
                color_alpha(
                    Color32::from_rgb(8, 10, 16),
                    (200.0 + hover * 20.0) as u8,
                ),
            );
            painter.circle_stroke(
                r_center,
                r_outer,
                Stroke::new(1.1, color_alpha(accent, (140.0 + hover * 30.0 + selected_t * 50.0) as u8)),
            );
            painter.circle_filled(
                r_center,
                1.6 + selected_t * 0.3,
                color_alpha(accent, (230.0 + hover * 15.0) as u8),
            );
            if fx_active {
                let glow_r = 3.0 + 1.0 * (0.5 + 0.5 * (time_s * 2.5 + entry_phase).sin());
                let ga = (45.0 + 55.0 * att_breathe) as u8;
                painter.circle_filled(
                    r_center,
                    glow_r,
                    color_alpha(accent, ga / 2),
                );
            }
            draw_rect.left() + 23.0
        } else {
            draw_rect.left() + 9.0
        };

        // 黑、加粗感：同位置微偏移多次叠字（无单独粗字重字体时等效于粗体）
        let tpos = Pos2::new(text_left, draw_rect.center().y);
        let tfont = egui::FontId::proportional(14.0);
        let tcol = Color32::from_rgb(0, 0, 0);
        for o in [
            Vec2::ZERO,
            Vec2::new(0.45, 0.0),
            Vec2::new(0.0, 0.45),
            Vec2::new(0.45, 0.45),
        ] {
            painter.text(
                tpos + o,
                Align2::LEFT_CENTER,
                text,
                tfont.clone(),
                tcol,
            );
        }
    }

    fn paint_source_aux_button(
        ui: &Ui,
        rect: Rect,
        response: &Response,
        id: Id,
        text: &str,
        accent: Color32,
        enabled: bool,
    ) {
        let hover = ui
            .ctx()
            .animate_bool_responsive(id.with("hover"), response.hovered() && enabled);
        let down = ui.ctx().animate_bool_responsive(
            id.with("down"),
            response.is_pointer_button_down_on() && enabled,
        );
        if hover > 0.001 || down > 0.001 {
            ui.ctx().request_repaint();
        }

        let painter = ui.painter();
        let draw_rect = rect.translate(Vec2::new(0.0, down * 1.0));
        let base = if enabled {
            Color32::from_rgb(29, 36, 46)
        } else {
            Color32::from_rgb(25, 30, 38)
        };
        painter.rect_filled(draw_rect, 8.0, base);
        painter.rect_filled(
            draw_rect,
            8.0,
            color_alpha(accent, (if enabled { 10.0 } else { 4.0 } + hover * 14.0) as u8),
        );
        painter.rect_stroke(
            draw_rect,
            8.0,
            Stroke::new(
                1.0 + hover * 0.35,
                if enabled {
                    color_alpha(accent, (92.0 + hover * 52.0) as u8)
                } else {
                    color_alpha(theme::BORDER_SUBTLE, 220)
                },
            ),
        );

        let badge = Pos2::new(draw_rect.left() + 13.0, draw_rect.center().y);
        painter.circle_filled(
            badge,
            4.0 + hover * 0.6,
            if enabled {
                color_alpha(accent, 180)
            } else {
                color_alpha(theme::TEXT_MUTED, 92)
            },
        );
        painter.circle_stroke(
            Pos2::new(draw_rect.right() - 13.0, draw_rect.center().y),
            5.0 + hover * 0.6,
            Stroke::new(
                1.0,
                if enabled {
                    color_alpha(accent, 170)
                } else {
                    color_alpha(theme::TEXT_MUTED, 72)
                },
            ),
        );
        painter.text(
            Pos2::new(draw_rect.left() + 26.0, draw_rect.center().y),
            Align2::LEFT_CENTER,
            text,
            egui::FontId::proportional(13.0),
            if enabled {
                theme::TEXT
            } else {
                theme::TEXT_MUTED
            },
        );
    }

    /// 顶栏：已标注文件名列表（紧凑行高，点击切换当前图）。
    fn ui_annotated_strip_top_bar(&mut self, ui: &mut Ui, bar_h: f32) {
        let title_h = 18.0_f32;
        let list_h = (bar_h - title_h - 6.0).max(36.0);
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
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(
                RichText::new(format!("已标注 {}", self.annotated_strip_indices.len()))
                    .small()
                    .strong()
                    .color(theme::TEXT),
            );
            ui.label(
                RichText::new("点击切换 · ✓ 当前")
                    .small()
                    .color(theme::TEXT_MUTED),
            );
            ui.label(
                RichText::new(&cur_disp_name)
                    .small()
                    .monospace()
                    .color(theme::ACCENT),
            )
            .on_hover_text(if cur_hover_path.is_empty() {
                "当前无图片".to_string()
            } else {
                cur_hover_path.clone()
            });
        });
        ui.add_space(3.0);
        Frame::default()
            .fill(color_alpha(theme::SURFACE_ELEVATED, 255))
            .inner_margin(egui::Margin::symmetric(6.0, 4.0))
            .rounding(egui::Rounding::same(8.0))
            .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
            .show(ui, |ui| {
                if self.annotated_strip_indices.is_empty() {
                    self.annotated_strip_step_scroll_pending = false;
                    ui.label(
                        RichText::new("保存有框标签或负样本空 .txt 后出现于此")
                            .small()
                            .color(theme::TEXT_MUTED),
                    );
                } else {
                    let strip_idxs = self.annotated_strip_indices.clone();
                    let summaries = self.annotated_strip_summaries.clone();
                    let neg_flags = self.annotated_strip_is_neg.clone();
                    let root = self.image_root.clone();
                    let row_h = 15.0_f32;
                    let font = egui::FontId::monospace(11.0);
                    egui::ScrollArea::vertical()
                        .id_salt("annotated_strip_top_scroll")
                        .max_height(list_h)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = 0.0;
                            for (row_i, strip_idx) in strip_idxs.iter().copied().enumerate() {
                                let Some(path) = self.image_paths.get(strip_idx).cloned() else {
                                    continue;
                                };
                                let is_neg_row = neg_flags.get(row_i).copied().unwrap_or(false);
                                let (n_cls, n_box) = summaries
                                    .get(row_i)
                                    .copied()
                                    .unwrap_or((0, 0));
                                let is_cur = strip_idx == self.current_index;
                                let rel = path_relative_to(&path, &root)
                                    .to_string_lossy()
                                    .to_string();
                                let full_path = path.display().to_string();
                                let fill = if is_cur {
                                    color_alpha(theme::ACCENT, 40)
                                } else {
                                    Color32::TRANSPARENT
                                };
                                Frame::default()
                                    .fill(fill)
                                    .inner_margin(egui::Margin::symmetric(4.0, 0.0))
                                    .rounding(egui::Rounding::same(5.0))
                                    .stroke(Stroke::new(
                                        1.0,
                                        if is_cur {
                                            color_alpha(theme::ACCENT, 120)
                                        } else {
                                            Color32::TRANSPARENT
                                        },
                                    ))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().item_spacing.x = 3.0;
                                            ui.add_sized(
                                                Vec2::new(12.0, row_h),
                                                Label::new(if is_cur {
                                                    RichText::new("✓")
                                                        .strong()
                                                        .color(theme::OK)
                                                        .font(font.clone())
                                                } else {
                                                    RichText::new("·")
                                                        .color(theme::TEXT_MUTED)
                                                        .font(font.clone())
                                                }),
                                            );
                                            let path_resp = ui.selectable_label(
                                                is_cur,
                                                RichText::new(&rel)
                                                    .font(font.clone())
                                                    .color(if is_cur {
                                                        theme::TEXT
                                                    } else {
                                                        theme::TEXT_MUTED
                                                    }),
                                            );
                                            let (sum_s, sum_tip) = if is_neg_row {
                                                (
                                                    "负样本".to_string(),
                                                    format!("{full_path}\n空 .txt，无目标"),
                                                )
                                            } else {
                                                (
                                                    format!("{n_cls}类{n_box}框"),
                                                    format!("{full_path}\n{n_cls} 个不同类别，{n_box} 个框"),
                                                )
                                            };
                                            let sum_resp = ui.label(
                                                RichText::new(sum_s)
                                                    .font(font.clone())
                                                    .color(if is_neg_row {
                                                        theme::ACCENT
                                                    } else {
                                                        theme::WARN
                                                    }),
                                            );
                                            let row_resp = path_resp.union(sum_resp).on_hover_text(sum_tip);
                                            if row_resp.clicked() {
                                                self.go_to_image_index(strip_idx);
                                            }
                                            if is_cur
                                                && self.annotated_strip_step_scroll_pending
                                            {
                                                // 仅上/下一张步进时滚入视区，避免每帧 scroll 导致手滚条无法停留
                                                row_resp.scroll_to_me(Some(Align::Center));
                                                self.annotated_strip_step_scroll_pending = false;
                                            }
                                        });
                                    });
                            }
                            if self.annotated_strip_step_scroll_pending {
                                // 步进后当前图不在本列表中（如正在看未标图）则无需也不应滚
                                self.annotated_strip_step_scroll_pending = false;
                            }
                        });
                }
            });
    }

    fn ui_sidebar(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            self.sidebar_width = ui.available_width();
            let footer_h = 236.0_f32;
            let header_h = 0.0_f32;
            let ah = ui.available_height();
            let scroll_h = if ah.is_finite() && ah > footer_h + header_h + 80.0 {
                ah - footer_h - header_h
            } else {
                332.0_f32
            };
            let mut open_section = self.sidebar_open_section;

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
                        "类别管理",
                        "在这里维护类别、排序、颜色和清理操作。",
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
                                    let cl_path = self.class_log_path();
                                    let cl_name = cl_path
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or(CLASS_LOG_FILENAME);
                                    ui.label(
                                        RichText::new(format!(
                                            "暂无类别。可以在新建标注框时命名，或直接编辑根目录「{}」新增非注释行。",
                                            cl_name
                                        ))
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                    );
                                });
                            } else {
                                while self.class_colors.len() < self.classes.len() {
                                    let k = self.class_colors.len();
                                    self.class_colors.push(palette_color(k));
                                }
                                self.class_colors.truncate(self.classes.len());
                                let release_lbl = self.show_label_window;

                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    let interact_h = ui.spacing().interact_size.y;
                                    // 每行外框：inner_margin 上下各 6 + 一行控件（色块/输入/按钮 ≈ interact_h），原先 +18 偏小，视口只能容纳约一行。
                                    let row_slot_h = interact_h + 30.0;
                                    let row_gap = 6.0_f32;
                                    let visible_rows = 3.0_f32;
                                    let scroll_viewport_h = row_slot_h * visible_rows
                                        + row_gap * (visible_rows - 1.0)
                                        + 8.0;
                                    ui.set_min_height(scroll_viewport_h);
                                    egui::ScrollArea::vertical()
                                        .id_salt("class_editor_scroll")
                                        .max_height(scroll_viewport_h)
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
                        2,
                        &mut open_section,
                        "02",
                        "训练配置",
                        "选择运行环境和基础权重，训练时会自动生成 train_* 包。",
                        Color32::from_rgb(42, 30, 36),
                        theme::DANGER,
                        |ui| {
                            ui.label(
                                RichText::new("训练方式")
                                    .small()
                                    .strong()
                                    .color(theme::TEXT),
                            );
                            ui.add_space(6.0);
                            let mode_w = ui.available_width().max(1.0);
                            let mode_gap = 8.0_f32;
                            let card_w = ((mode_w - mode_gap) * 0.5).max(96.0);
                            let card_h = 54.0_f32;
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = mode_gap;
                                let builtin_id = ui.id().with("training_backend_builtin");
                                let conda_id = ui.id().with("training_backend_conda");
                                let (builtin_rect, builtin_resp) =
                                    ui.allocate_exact_size(Vec2::new(card_w, card_h), Sense::click());
                                let (conda_rect, conda_resp) =
                                    ui.allocate_exact_size(Vec2::new(card_w, card_h), Sense::click());
                                training_backend_card(
                                    ui,
                                    builtin_rect,
                                    &builtin_resp,
                                    builtin_id,
                                    "自带 CPU",
                                    "内置训练器 + 自动转 ONNX",
                                    theme::OK,
                                    self.use_builtin_cpu_train,
                                );
                                training_backend_card(
                                    ui,
                                    conda_rect,
                                    &conda_resp,
                                    conda_id,
                                    "Conda 环境",
                                    "填写环境根目录并调用 python.exe",
                                    theme::ACCENT,
                                    !self.use_builtin_cpu_train,
                                );
                                if builtin_resp.clicked() {
                                    self.use_builtin_cpu_train = true;
                                }
                                if conda_resp.clicked() {
                                    self.use_builtin_cpu_train = false;
                                }
                            });

                            ui.add_space(10.0);
                            if self.use_builtin_cpu_train {
                                let train_ready = find_builtin_cpu_train_exe().is_some();
                                let export_ready = find_builtin_onnx_export_exe().is_some();
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    // 与「训练方式 / 预训练权重」等侧栏定宽区贴合：仅内容时 Frame 会缩窄，需占满可用列宽
                                    let full_w = ui.available_width().max(1.0);
                                    ui.set_min_width(full_w);
                                    ui.label(
                                        RichText::new("内置组件状态")
                                            .small()
                                            .strong()
                                            .color(theme::TEXT),
                                    );
                                    ui.add_space(8.0);
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
                                });
                            } else {
                                app_card(theme::SURFACE_SOFT).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new("Conda 环境根目录")
                                                .small()
                                                .strong()
                                                .color(theme::TEXT),
                                        );
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui.small_button("刷新列表").clicked() {
                                                    self.refresh_conda_env_list();
                                                }
                                            },
                                        );
                                    });
                                    ui.add_space(8.0);
                                    if self.conda_env_paths.is_empty() {
                                        ui.label(
                                            RichText::new(
                                                "未检测到 Conda，下面可直接填写环境根目录；目录内需要有 python.exe。",
                                            )
                                            .small()
                                            .color(theme::TEXT_MUTED),
                                        );
                                    } else {
                                        self.conda_env_idx = self
                                            .conda_env_idx
                                            .min(self.conda_env_paths.len().saturating_sub(1));
                                        let before_idx = self.conda_env_idx;
                                        let n = self.conda_env_paths.len();
                                        ComboBox::from_id_salt("conda_env_pick")
                                            .width(ui.available_width())
                                            .selected_text(self.conda_env_paths[self.conda_env_idx].as_str())
                                            .show_index(ui, &mut self.conda_env_idx, n, |i| {
                                                self.conda_env_paths[i].as_str()
                                            });
                                        if self.conda_env_idx != before_idx
                                            || self.conda_env_custom_root.trim().is_empty()
                                        {
                                            if let Some(path) = self.conda_env_paths.get(self.conda_env_idx) {
                                                self.conda_env_custom_root = path.clone();
                                            }
                                        }
                                        ui.add_space(6.0);
                                        ui.label(
                                            RichText::new("也可以直接填写路径")
                                                .small()
                                                .color(theme::TEXT_MUTED),
                                        );
                                    }
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.conda_env_custom_root)
                                            .desired_width(ui.available_width())
                                            .hint_text(default_conda_env_path()),
                                    );
                                    ui.add_space(6.0);
                                    let resolved_root = self.resolved_conda_root();
                                    let py = conda_python_executable(Path::new(&resolved_root));
                                    ui.horizontal_wrapped(|ui| {
                                        status_chip(
                                            ui,
                                            if py.is_file() {
                                                "python.exe 已找到"
                                            } else {
                                                "等待有效环境路径"
                                            },
                                            if py.is_file() { theme::OK } else { theme::WARN },
                                        );
                                        if !self.conda_env_paths.is_empty() {
                                            status_chip(
                                                ui,
                                                &format!("检测到 {} 个环境", self.conda_env_paths.len()),
                                                theme::ACCENT,
                                            );
                                        }
                                    });
                                    ui.add_space(4.0);
                                    ui.label(
                                        RichText::new(format!(
                                            "当前将使用：{}",
                                            if resolved_root.trim().is_empty() {
                                                "未设置".to_string()
                                            } else {
                                                resolved_root
                                            }
                                        ))
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                    );
                                });
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
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new("查找顺序：程序目录 -> exe 同目录 -> 图片根目录。若本地没有，则只下载到本次训练包文件夹。")
                                        .small()
                                        .color(theme::TEXT_MUTED),
                                );
                            });

                            if false {
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
                                RichText::new(
                                    "epochs：输入框中直接输入，或托住数字横向拖动；在「开始训练」按钮上滚轮为每次 ±10。",
                                )
                                .small()
                                .color(theme::TEXT_MUTED),
                            );
                            }
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
                    ui.label(
                        RichText::new("epochs")
                            .size(15.0)
                            .strong()
                            .color(theme::TEXT),
                    );
                    {
                        use std::ops::Deref;
                        let mut s = ui.style().deref().clone();
                        s.override_font_id = Some(egui::FontId::proportional(16.0));
                        ui.set_style(s);
                        ui.add(
                            egui::DragValue::new(&mut self.train_epochs)
                                .range(1..=100_000)
                                .speed(1.0),
                        );
                        ui.reset_style();
                    }
                });
                ui.add_space(6.0);

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
                            const STEP: u32 = 10;
                            while self.train_epoch_scroll_accum >= NOTCH {
                                self.train_epochs = self.train_epochs.saturating_add(STEP).min(100_000);
                                self.train_epoch_scroll_accum -= NOTCH;
                            }
                            while self.train_epoch_scroll_accum <= -NOTCH {
                                self.train_epochs = (self.train_epochs.saturating_sub(STEP)).max(1);
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
                    // 第一行：模型与状态；未载入 ONNX 时「查看完整快捷键」靠右与载入按钮同一行
                    ui.horizontal(|ui| {
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
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if assist_batch {
                                ui.spinner();
                                ui.label(RichText::new("全局采纳中…").small().weak());
                            } else if assist_infer {
                                ui.spinner();
                                ui.label(RichText::new("辅助推理中…").small().weak());
                            }
                            if self.assist_ort.is_none() {
                                canvas_shortcuts_help_popup(ui);
                            }
                        });
                    });

                    // 第二行：仅载入 ONNX 后显示：后处理与采纳 + 同排右侧「查看完整快捷键」
                    if self.assist_ort.is_some() {
                    let assist_pct = (ASSIST_ADOPT_DUP_IOU * 100.0).round() as i32;
                    let row_h = ui.spacing().interact_size.y;
                    Frame::default()
                            .fill(theme::SURFACE_SOFT)
                            .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                            .rounding(egui::Rounding::same(8.0))
                            .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                egui::ScrollArea::horizontal()
                                    .id_salt("assist_onnx_toolbar_h")
                                    .auto_shrink([true, true])
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().item_spacing = Vec2::new(6.0, 0.0);

                                            // 置信度 / IoU：同一列宽标签（右对齐）+ 等宽滑块，两行严格对齐。
                                            const SLIDER_LABEL_W: f32 = 58.0;
                                            const SLIDER_TRACK_W: f32 = 90.0;
                                            ui.add_enabled_ui(!assist_row2_blocked, |ui| {
                                                ui.vertical(|ui| {
                                                    ui.spacing_mut().item_spacing.y = 5.0;
                                                    ui.horizontal(|ui| {
                                                        ui.spacing_mut().item_spacing.x = 8.0;
                                                        ui.allocate_ui_with_layout(
                                                            Vec2::new(SLIDER_LABEL_W, row_h),
                                                            egui::Layout::right_to_left(egui::Align::Center),
                                                            |ui| {
                                                                ui.label(
                                                                    RichText::new("置信度")
                                                                        .small()
                                                                        .color(theme::TEXT_MUTED),
                                                                );
                                                            },
                                                        );
                                                        let r = ui
                                                            .add_sized(
                                                                [SLIDER_TRACK_W, row_h],
                                                                egui::Slider::new(
                                                                    &mut self.assist_onnx_conf,
                                                                    0.0..=1.0,
                                                                )
                                                                .fixed_decimals(2),
                                                            )
                                                            .on_hover_text(
                                                                "低于此置信度的检测会被丢弃，随后再按 IoU 做 NMS。",
                                                            );
                                                        if r.changed() {
                                                            self.assist_onnx_conf = self
                                                                .assist_onnx_conf
                                                                .clamp(0.0, 1.0);
                                                            self.schedule_assist_infer();
                                                        }
                                                    });
                                                    ui.horizontal(|ui| {
                                                        ui.spacing_mut().item_spacing.x = 8.0;
                                                        ui.allocate_ui_with_layout(
                                                            Vec2::new(SLIDER_LABEL_W, row_h),
                                                            egui::Layout::right_to_left(egui::Align::Center),
                                                            |ui| {
                                                                ui.label(
                                                                    RichText::new("IoU")
                                                                        .small()
                                                                        .color(theme::TEXT_MUTED),
                                                                );
                                                            },
                                                        );
                                                        let r = ui
                                                            .add_sized(
                                                                [SLIDER_TRACK_W, row_h],
                                                                egui::Slider::new(
                                                                    &mut self.assist_onnx_iou,
                                                                    0.0..=1.0,
                                                                )
                                                                .fixed_decimals(2),
                                                            )
                                                            .on_hover_text(
                                                                "同类框 IoU 大于该值时去掉较低分框；nms=True/False 导出均会再应用一轮。\n调高 → 要更高重叠才去重；调低 → 去重更积极。",
                                                            );
                                                        if r.changed() {
                                                            self.assist_onnx_iou = self
                                                                .assist_onnx_iou
                                                                .clamp(0.0, 1.0);
                                                            self.schedule_assist_infer();
                                                        }
                                                    });
                                                });
                                            });

                                            ui.separator();

                                            ui.add_enabled_ui(!assist_batch, |ui| {
                                                let mut vis = self.assist_overlay_visible;
                                                let r = ui
                                                    .checkbox(&mut vis, "虚线")
                                                    .on_hover_text(
                                                        "显示虚线预测框；关闭后仍可推理与采纳",
                                                    );
                                                if r.changed() {
                                                    self.assist_overlay_visible = vis;
                                                }
                                            });

                                            if !self.assist_class_names.is_empty() {
                                                ui.separator();
                                                self.ensure_assist_pred_class_mask();
                                                let n = self.assist_class_names.len();
                                                let n_on = self
                                                    .assist_pred_class_on
                                                    .iter()
                                                    .filter(|&&x| x)
                                                    .count();
                                                let menu_label = if n_on == n {
                                                    format!("类 {n} ▾")
                                                } else if n_on == 0 {
                                                    "类 · 关 ▾".to_string()
                                                } else {
                                                    format!("类 {n_on}/{n} ▾")
                                                };
                                                ui.add_enabled_ui(!assist_batch, |ui| {
                                                    ui.menu_button(
                                                        RichText::new(menu_label)
                                                            .small()
                                                            .color(theme::ACCENT),
                                                        |ui| {
                                                            ui.label(
                                                                RichText::new(
                                                                    "勾选参与预览与采纳的类别",
                                                                )
                                                                .weak()
                                                                .small(),
                                                            );
                                                            ui.horizontal(|ui| {
                                                                if ui.small_button("全选").clicked()
                                                                {
                                                                    for v in &mut self
                                                                        .assist_pred_class_on
                                                                    {
                                                                        *v = true;
                                                                    }
                                                                }
                                                                if ui.small_button("全不选").clicked()
                                                                {
                                                                    for v in &mut self
                                                                        .assist_pred_class_on
                                                                    {
                                                                        *v = false;
                                                                    }
                                                                }
                                                            });
                                                            ui.separator();
                                                            egui::ScrollArea::vertical()
                                                                .max_height(220.0)
                                                                .id_salt(
                                                                    "assist_pred_class_scroll",
                                                                )
                                                                .show(ui, |ui| {
                                                                    for i in 0..self
                                                                        .assist_class_names
                                                                        .len()
                                                                    {
                                                                        let name = self
                                                                            .assist_class_names
                                                                            [i]
                                                                            .as_str();
                                                                        let mut on =
                                                                            self.assist_pred_class_on
                                                                                [i];
                                                                        if ui
                                                                            .checkbox(&mut on, name)
                                                                            .changed()
                                                                        {
                                                                            self.assist_pred_class_on
                                                                                [i] = on;
                                                                        }
                                                                    }
                                                                });
                                                        },
                                                    );
                                                });
                                            }

                                            ui.separator();

                                            ui.add_enabled_ui(
                                                !assist_row2_blocked
                                                    && self.rgba.is_some()
                                                    && !self.assist_preds.is_empty(),
                                                |ui| {
                                                    let b = egui::Button::new(
                                                        RichText::new("采纳当前")
                                                            .small()
                                                            .strong(),
                                                    )
                                                    .fill(color_alpha(theme::ACCENT_DIM, 200))
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        color_alpha(theme::ACCENT, 140),
                                                    ))
                                                    .min_size(Vec2::new(68.0, row_h));
                                                    if ui
                                                        .add(b)
                                                        .on_hover_text(format!(
                                                            "将虚线框写入当前图 .txt；与已有框 IoU≥{assist_pct}% 则不新增"
                                                        ))
                                                        .clicked()
                                                    {
                                                        self.adopt_onnx_assist_to_annotations();
                                                        ctx.request_repaint();
                                                    }
                                                },
                                            );

                                            ui.add_enabled_ui(
                                                !assist_row2_blocked
                                                    && !self.image_paths.is_empty(),
                                                |ui| {
                                                    let b = egui::Button::new(
                                                        RichText::new("全局采纳")
                                                            .small()
                                                            .strong(),
                                                    )
                                                    .fill(color_alpha(theme::WARN, 55))
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        color_alpha(theme::WARN, 160),
                                                    ))
                                                    .min_size(Vec2::new(72.0, row_h));
                                                    if ui
                                                        .add(b)
                                                        .on_hover_text(format!(
                                                            "对数据集每张图推理并写入；规则同采纳当前；Ctrl+Z 可整集撤销（IoU≥{assist_pct}% 保留原框）"
                                                        ))
                                                        .clicked()
                                                    {
                                                        self.schedule_assist_global_adopt();
                                                        ctx.request_repaint();
                                                    }
                                                },
                                            );

                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    canvas_shortcuts_help_popup(ui);
                                                    ui.separator();
                                                },
                                            );
                                        });
                                    });
                            });
                    }
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
                                RichText::new("在顶部进度/统计信息下方点击「选择图片目录…」，将列出该目录内的图片与同名标签（不含子文件夹）。")
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
        let _ = response.context_menu(|ui| {
            let can = !self.image_paths.is_empty()
                && self.annotations.is_empty()
                && !self.show_label_window
                && self.scribble_kind.is_none()
                && !self.scribble_active
                && self.pending_box.is_none()
                && self.pending_boxes_batch.is_empty();
            if !can {
                return;
            }
            let Some(p) = self.image_paths.get(self.current_index) else {
                return;
            };
            if path_has_nonempty_label_file(p) {
                ui.label(
                    RichText::new("当前图已有非空标签行，请清空标签后再设负样本")
                        .small()
                        .color(theme::TEXT_MUTED),
                );
                return;
            }
            if path_is_negative_label_only(p) {
                ui.label(
                    RichText::new("已是负样本（与图同名的空 .txt 已存在）")
                        .small()
                        .color(theme::TEXT_MUTED),
                );
                return;
            }
            if ui
                .button("将此图以负样本加入训练")
                .on_hover_text("写入与图同名的**空** .txt；将出现在已标注列表，视频进度为蓝点。")
                .clicked()
            {
                match self.mark_current_image_as_negative_sample() {
                    Ok(()) => {
                        self.train_log
                            .push("已写入空 .txt 作为负样本，并刷新已标/进度。".to_string());
                    }
                    Err(e) => self
                        .train_log
                        .push(format!("[负样本] 写入失败: {e}")),
                }
            }
        });

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
        if self.carousel_ring_active() && self.drag.is_some() {
            let _ = self.save_current_labels();
            self.drag = None;
        }
        self.carousel_ring_step(
            ctx,
            &response,
            inner,
            dt,
            pointer,
            primary_pressed,
            primary_released,
            dragging,
            space_down,
        );
        self.carousel_ring_try_pick(ctx, inner, pointer);
        let hover_canvas = response.hovered();
        let hover_img_pos = if hover_canvas {
            pointer.map(|p| map_screen_to_image_px(p, disp, img_wf, img_hf))
        } else {
            None
        };
        // 与按 W/S 时相比指针在**屏幕**上移动超过约 3 点则放弃同点双击改层（缩放不改变屏幕位置，不破坏锁定）
        const STACK_DBLCLK_MAX_MOVE_PX2: f32 = 3.0 * 3.0;
        if let Some((at_screen, _tidx, lock_frame)) = self.stack_nav_dblclk_lock.as_ref() {
            let should_clear = *lock_frame != self.current_index
                || pointer.map_or(true, |cur| (cur - *at_screen).length_sq() > STACK_DBLCLK_MAX_MOVE_PX2);
            if should_clear {
                self.stack_nav_dblclk_lock = None;
            }
        }
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

            let mut skip_primary_pending_reopen = false;
            if !self.show_label_window
                && (self.pending_box.is_some() || !self.pending_boxes_batch.is_empty())
                && primary_pressed
                && !block_bbox
                && !space_down
                && self.scribble_kind.is_none()
                && response.hovered()
            {
                if let Some((pix, piy)) = screen_to_image(p, disp, img_wf, img_hf) {
                    let hit_pb =
                        |pb: &PendingBox| pix >= pb.min_x && pix <= pb.max_x && piy >= pb.min_y && piy <= pb.max_y;
                    let hit = self.pending_box.as_ref().is_some_and(hit_pb)
                        || self.pending_boxes_batch.iter().any(hit_pb);
                    if hit {
                        self.show_label_window = true;
                        self.label_edit_idx = None;
                        self.label_draft = self.active_class_label_draft();
                        skip_primary_pending_reopen = true;
                    }
                }
            }

            let mut skip_primary_after_dbl = false;
            if ctx.input(|i| i.pointer.button_double_clicked(PointerButton::Primary))
                && !block_bbox
                && !skip_primary_pending_reopen
                && !self.draw_new_boxes_enabled
                && self.scribble_kind.is_none()
                && !space_down
                && response.hovered()
            {
                if let Some((ix, iy)) = screen_to_image(p, disp, img_wf, img_hf) {
                    const STACK_DBLCLK_MAX_MOVE_PX2: f32 = 3.0 * 3.0;
                    let idx = {
                        let from_stack_lock = if let Some((sp, tidx, at_i)) =
                            self.stack_nav_dblclk_lock
                        {
                            if at_i == self.current_index
                                && tidx < self.annotations.len()
                                && Self::hit_inside((ix, iy), &self.annotations[tidx])
                                && (p - sp).length_sq() <= STACK_DBLCLK_MAX_MOVE_PX2
                            {
                                Some(tidx)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some(i) = from_stack_lock {
                            Some(i)
                        } else {
                            self.annotations
                                .iter()
                                .enumerate()
                                .rev()
                                .find(|(_, b)| Self::hit_inside((ix, iy), b))
                                .map(|(i, _)| i)
                        }
                    };
                    if let Some(idx) = idx {
                        self.stack_nav_dblclk_lock = None;
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

            if self.scribble_kind.is_some() && !block_bbox && !space_down && !skip_primary_pending_reopen {
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

            if primary_pressed
                && !block_bbox
                && !skip_primary_after_dbl
                && !skip_primary_pending_reopen
                && self.scribble_kind.is_none()
                && !self.carousel_ring_active()
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
                                    if let Some(idx) = self
                                        .overlap_stack_indices_at((ix, iy))
                                        .into_iter()
                                        .next()
                                    {
                                        self.push_undo(UndoScope::Local);
                                        self.selected = Some(idx);
                                        self.drag = Some((DragKind::Move, idx));
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

        let label_te_focused_here = ctx.memory(|m| m.has_focus(label_draft_textedit_id()));
        let typing_elsewhere = ctx.wants_keyboard_input();
        ctx.input(|i| {
            let q_remove = i.key_pressed(egui::Key::Q)
                && !self.show_label_window
                && !typing_elsewhere;
            if i.key_pressed(egui::Key::Delete) {
                if let Some(si) = self.selected {
                    if si < self.annotations.len() {
                        self.push_undo(UndoScope::Local);
                        self.annotations.remove(si);
                        self.selected = None;
                        let _ = self.save_current_labels();
                    }
                }
            } else if q_remove {
                if self.suppress_bbox_q_delete_once {
                    self.suppress_bbox_q_delete_once = false;
                } else if let Some(si) = self.selected {
                    if si < self.annotations.len() {
                        self.push_undo(UndoScope::Local);
                        self.annotations.remove(si);
                        self.selected = None;
                        let _ = self.save_current_labels();
                    }
                }
            }
            let allow_ad = !self.show_label_window
                && !label_te_focused_here
                && (!typing_elsewhere || hover_canvas);
            if allow_ad {
                if self.drag.is_none() {
                    if let Some(p_img) = hover_img_pos {
                        if i.key_pressed(egui::Key::W) {
                            if self.cycle_overlap_selection_at(p_img, true) {
                                if let (Some(screen_p), Some(si)) = (pointer, self.selected) {
                                    self.stack_nav_dblclk_lock =
                                        Some((screen_p, si, self.current_index));
                                }
                            }
                        }
                        if i.key_pressed(egui::Key::S) {
                            if self.cycle_overlap_selection_at(p_img, false) {
                                if let (Some(screen_p), Some(si)) = (pointer, self.selected) {
                                    self.stack_nav_dblclk_lock =
                                        Some((screen_p, si, self.current_index));
                                }
                            }
                        }
                    }
                }
                if i.key_pressed(egui::Key::A) {
                    if self.current_index > 0 {
                        self.image_nav_btn_fx[0] = 1.0;
                    }
                    self.go_prev_image();
                }
                if i.key_pressed(egui::Key::D) {
                    if !self.image_paths.is_empty()
                        && self.current_index + 1 < self.image_paths.len()
                    {
                        self.image_nav_btn_fx[1] = 1.0;
                    }
                    self.go_next_image();
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

        const PENDING_LBL_DASH: f32 = 5.0;
        const PENDING_LBL_GAP: f32 = 4.0;
        let pending_lbl_stroke =
            Stroke::new(2.0, Color32::from_rgba_unmultiplied(90, 210, 130, 220));
        for pb in self.pending_box.iter().chain(self.pending_boxes_batch.iter()) {
            let c_tl = image_to_screen(pb.min_x, pb.min_y, disp, img_wf, img_hf);
            let c_br = image_to_screen(pb.max_x, pb.max_y, disp, img_wf, img_hf);
            let r = Rect::from_two_pos(c_tl, c_br);
            let mn = r.min;
            let mx = r.max;
            let top = [Pos2::new(mn.x, mn.y), Pos2::new(mx.x, mn.y)];
            let right = [Pos2::new(mx.x, mn.y), Pos2::new(mx.x, mx.y)];
            let bottom = [Pos2::new(mx.x, mx.y), Pos2::new(mn.x, mx.y)];
            let left = [Pos2::new(mn.x, mx.y), Pos2::new(mn.x, mn.y)];
            for seg in [&top[..], &right[..], &bottom[..], &left[..]] {
                for s in Shape::dashed_line(seg, pending_lbl_stroke, PENDING_LBL_DASH, PENDING_LBL_GAP) {
                    painter.add(s);
                }
            }
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
                theme::ACCENT,
                color_alpha(theme::ACCENT, 210),
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

        // 柔性外接：红色十字；连续柔性外接：蓝色十字；矩形拉框：淡绿色十字
        if response.hovered() && self.scribble_kind.is_some() {
            let p = pointer.or_else(|| ctx.input(|i| i.pointer.hover_pos()));
            if let Some(p) = p {
                if disp.contains(p) {
                    let dash_col = match self.scribble_kind {
                        Some(ScribbleKind::ContinuousCircumscribed) => {
                            color_alpha(theme::ACCENT, 200)
                        }
                        _ => Color32::from_rgba_unmultiplied(255, 72, 72, 200),
                    };
                    let stroke = Stroke::new(CROSSHAIR_STROKE, dash_col);
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
                        CROSSHAIR_STROKE,
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

        let mut mode_str: String = match self.scribble_kind {
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
        if self.carousel_ring_active() {
            mode_str.push_str(" · 环轨相册");
            painter.rect_filled(
                inner,
                6.0,
                Color32::from_rgba_unmultiplied(8, 10, 18, 140),
            );
            self.paint_carousel_ring(ctx, &painter, inner);
        }
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

        if self.carousel_ring_active() {
            let bar_h = 34.0_f32;
            let bar = Rect::from_min_max(
                Pos2::new(inner.left() + 8.0, inner.bottom() - bar_h - 6.0),
                Pos2::new(inner.right() - 8.0, inner.bottom() - 6.0),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(bar), |ui| {
                Frame::default()
                    .fill(Color32::from_rgba_unmultiplied(14, 16, 24, 210))
                    .rounding(egui::Rounding::same(8.0))
                    .stroke(Stroke::new(1.0, theme::BORDER_SUBTLE))
                    .inner_margin(egui::Margin::symmetric(10.0, 6.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 10.0;
                            ui.label(
                                RichText::new("环轨分组")
                                    .small()
                                    .color(theme::TEXT_MUTED),
                            );
                            ui.separator();
                            let n_a = self.annotated_strip_indices.len();
                            let n_u = self.unannotated_strip_indices.len();
                            let a_sel = self.carousel_ring_pool == CarouselRingPool::Annotated;
                            let u_sel = self.carousel_ring_pool == CarouselRingPool::Unannotated;
                            if ui
                                .selectable_label(
                                    a_sel,
                                    RichText::new(format!("已标注 · {n_a}"))
                                        .small()
                                        .strong(),
                                )
                                .on_hover_text("仅浏览已有非空标签的图片")
                                .clicked()
                            {
                                self.set_carousel_ring_pool(CarouselRingPool::Annotated);
                            }
                            if ui
                                .selectable_label(
                                    u_sel,
                                    RichText::new(format!("未标注 · {n_u}"))
                                        .small()
                                        .strong(),
                                )
                                .on_hover_text("浏览尚无有效标注行的图片")
                                .clicked()
                            {
                                self.set_carousel_ring_pool(CarouselRingPool::Unannotated);
                            }
                        });
                    });
            });
        }
    }

    fn ui_top_bar(&mut self, ui: &mut Ui) {
        // 含导航行 + 统计芯片 +「选择图片目录」按钮
        let top_h = 140.0;
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
                let middle_w = (slot - overview_w - gap).max(0.0);
                let middle_rect = Rect::from_min_size(
                    Pos2::new(min_overview_left, left_rect.top()),
                    Vec2::new(middle_w, top_h),
                );
                let right_rect = Rect::from_min_size(
                    Pos2::new(overview_left, left_rect.top()),
                    Vec2::new(overview_w, top_h),
                );
                let union_rect = left_rect.union(middle_rect).union(right_rect);
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
                        const NAME_PX: f32 = 31.0;
                        const LOGO_INSET: f32 = 3.0;
                        let name_yolo = self.brand_name_font(NAME_PX);
                        let name_vet = self.brand_name_font(NAME_PX);
                        ui.with_layout(
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
                                if let Some(tex) = &self.top_bar_logo {
                                    // 左上展示行内，为 YoloVet+分隔+状态芯片保留宽度，徽标为内接**最大**正方形
                                    const RESERVE_NAME_CHIP: f32 = 255.0;
                                    const HAIR_GUT: f32 = 16.0;
                                    const ROW_PAD: f32 = 8.0;
                                    let row = ui.max_rect();
                                    let h_lim = (row.height() * 0.95).max(1.0);
                                    let w_lim = (row.width() - RESERVE_NAME_CHIP - HAIR_GUT - ROW_PAD)
                                        .max(1.0);
                                    // 在「可用行高 × 为 logo 预留的横向带」的矩形中内接的最大正方形
                                    let side = h_lim.min(w_lim);
                                    // 过窄/过矮时仍保下限；过大时限制在单栏内观感
                                    let side = side.clamp(24.0, 96.0);
                                    let logo_outer = side + LOGO_INSET * 2.0;
                                    let corner = (side * 0.26).clamp(6.0, 12.0);
                                    Frame::none()
                                        .fill(theme::SURFACE_DEEP)
                                        .inner_margin(egui::Margin::same(LOGO_INSET))
                                        .rounding(egui::Rounding::same(corner))
                                        .stroke(Stroke::new(
                                            1.0,
                                            color_alpha(theme::ACCENT, 64),
                                        ))
                                        .show(ui, |ui| {
                                            let (r, _resp) = ui.allocate_exact_size(
                                                Vec2::splat(side),
                                                Sense::hover(),
                                            );
                                            ui.painter().image(
                                                tex.id(),
                                                r,
                                                Rect::from_min_max(
                                                    Pos2::ZERO,
                                                    Pos2::new(1.0, 1.0),
                                                ),
                                                Color32::WHITE,
                                            );
                                        });
                                    let h_cell = logo_outer;
                                    let (sep_cell, _resp) = ui.allocate_exact_size(
                                        Vec2::new(16.0, h_cell),
                                        Sense::hover(),
                                    );
                                    let x = sep_cell.left() + 3.0;
                                    let y0 = sep_cell.top() + 3.0;
                                    let y1 = sep_cell.bottom() - 3.0;
                                    ui.painter().line_segment(
                                        [Pos2::new(x, y0), Pos2::new(x, y1)],
                                        Stroke::new(1.0, color_alpha(theme::BORDER, 100)),
                                    );
                                }
                                ui.add_space(2.0);
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
                                    // YoloVet（Times New Roman）+ 与徽标呼应的 Yolo / Vet 分色
                                    ui.label(
                                        RichText::new("Yolo")
                                            .font(name_yolo)
                                            .color(theme::TEXT),
                                    );
                                    ui.add_space(1.0);
                                    ui.label(
                                        RichText::new("Vet")
                                            .font(name_vet)
                                            .color(color_alpha(theme::ACCENT, 250)),
                                    );
                                });
                                ui.add_space(6.0);
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if self.training {
                                            status_chip(ui, "训练中", theme::WARN);
                                        } else {
                                            status_chip(ui, "待命", theme::OK);
                                        }
                                    },
                                );
                            },
                        );
                    });

                if middle_w > 48.0 {
                    let mut middle_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(middle_rect)
                            .layout(egui::Layout::top_down(egui::Align::Min)),
                    );
                    middle_ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(8.0, 0.0);
                        let sep_w = 10.0_f32;
                        let strip_w = (middle_w * 0.48)
                            .clamp(120.0, 280.0)
                            .min((middle_w - sep_w - 100.0).max(100.0));
                        let nav_w = (middle_w - strip_w - sep_w).max(72.0);
                        ui.allocate_ui_with_layout(
                            Vec2::new(strip_w, top_h),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                self.ui_annotated_strip_top_bar(ui, top_h);
                            },
                        );
                        ui.separator();
                        ui.allocate_ui_with_layout(
                            Vec2::new(nav_w, top_h),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                self.ui_top_bar_image_nav(ui);
                            },
                        );
                    });
                } else if middle_rect.width() > 1.0 {
                    // 中部极窄时仍保留选目录入口（按钮已从侧栏移出）
                    let mut middle_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(middle_rect)
                            .layout(egui::Layout::top_down(egui::Align::Min)),
                    );
                    middle_ui.vertical(|ui| {
                        let compact_pick_w = ui.available_width().max(1.0);
                        self.ui_source_picker_pair(ui, compact_pick_w, 28.0);
                    });
                }

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
            let header_h = ui.spacing().interact_size.y.max(28.0);
            let header_rect =
                Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), header_h));
            ui.allocate_rect(header_rect, Sense::hover());

            let left_rect = Rect::from_min_max(
                header_rect.left_top(),
                Pos2::new(header_rect.left() + 340.0, header_rect.bottom()),
            );
            let mut left_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(left_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            let toggle_text = if self.train_log_expanded {
                "收起训练日志"
            } else {
                "展开训练日志"
            };
            if left_ui
                .button(toggle_text)
                .on_hover_text("点击切换训练日志面板的展开与折叠")
                .clicked()
            {
                self.train_log_expanded = !self.train_log_expanded;
            }
            left_ui.label(RichText::new("训练日志").strong().size(15.0).color(theme::TEXT));
            left_ui.label(
                RichText::new(format!("{} 行", self.train_log.len()))
                    .small()
                    .color(theme::TEXT_MUTED),
            );

            let right_rect = Rect::from_min_max(
                Pos2::new(header_rect.right() - 120.0, header_rect.top()),
                header_rect.right_bottom(),
            );
            let mut right_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(right_rect)
                    .layout(egui::Layout::right_to_left(egui::Align::Center)),
            );
            if self.train_log_expanded && right_ui.small_button("清空日志").clicked() {
                self.train_log.clear();
            }

            if self.video_session.is_some() {
                self.ui_bottom_video_progress(ui, header_rect, left_rect.right(), right_rect.left());
            }
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

    fn ui_bottom_video_progress(
        &mut self,
        ui: &mut Ui,
        header_rect: Rect,
        left_limit: f32,
        right_limit: f32,
    ) {
        let total_images = self.image_paths.len();
        if total_images == 0 {
            return;
        }
        let canvas_center_x = self
            .last_canvas_inner
            .map(|r| r.center().x)
            .unwrap_or_else(|| header_rect.center().x);
        let canvas_w = self
            .last_canvas_inner
            .map(|r| r.width())
            .unwrap_or_else(|| header_rect.width());
        let available_w = (right_limit - left_limit - 24.0).max(0.0);
        if available_w < 120.0 {
            return;
        }
        let strip_w = (canvas_w * 0.74)
            .clamp(480.0, 1020.0)
            .min(available_w);
        let mut x0 = canvas_center_x - strip_w * 0.5;
        x0 = x0.clamp(left_limit + 10.0, right_limit - strip_w - 10.0);
        let strip_rect = Rect::from_center_size(
            Pos2::new(x0 + strip_w * 0.5, header_rect.center().y),
            Vec2::new(strip_w + 34.0, 22.0),
        );
        let strip_response = ui.interact(
            strip_rect,
            ui.id().with("bottom_video_progress"),
            Sense::click().union(Sense::drag()),
        )
        .on_hover_cursor(CursorIcon::PointingHand);
        let progress_id = ui.id().with("bottom_video_progress_fx");
        let hover = ui
            .ctx()
            .animate_bool_responsive(progress_id.with("hover"), strip_response.hovered());
        let active = ui.ctx().animate_bool_responsive(
            progress_id.with("active"),
            strip_response.is_pointer_button_down_on() || strip_response.dragged(),
        );
        if hover > 0.001 || active > 0.001 {
            ui.ctx().request_repaint();
        }
        let painter = ui.painter_at(strip_rect);
        let primary = Color32::from_rgb(98, 210, 130);
        let primary_edge = Color32::from_rgb(132, 232, 160);
        let red = Color32::from_rgb(222, 76, 76);
        let neg_blue = Color32::from_rgb(96, 168, 255);
        let neg_blue_stroke = Color32::from_rgb(140, 200, 255);
        let magnet_amber = Color32::from_rgb(255, 200, 95);
        let track = strip_rect.shrink2(Vec2::new(9.0, 4.1 - active * 0.6));
        let outer_track = track.expand2(Vec2::new(6.0 + hover * 2.0, 1.8 + hover * 0.9));
        painter.rect_filled(
            outer_track,
            999.0,
            color_alpha(primary, (18.0 + hover * 22.0 + active * 28.0) as u8),
        );
        painter.rect_filled(track, 999.0, Color32::from_rgb(19, 25, 31));
        painter.rect_filled(
            track,
            999.0,
            color_alpha(primary_edge, (8.0 + hover * 12.0) as u8),
        );

        const MAGNET_PX: f32 = 22.0; // 屏内距离：进入此圈即磁吸
        let bn = self.image_nav_progress_bucket_n;
        let b_pos = &self.image_nav_progress_bucket_has_box;
        let b_neg = &self.image_nav_progress_bucket_has_neg;
        let bar_w = track.width();
        let bar_left = track.left();
        let bar_cy = track.center().y;
        let t_anim = ui.ctx().input(|i| i.time) as f32;
        let mag_pulse = 0.5 + 0.5 * (t_anim * 6.2).sin();
        // 悬停且指针靠近某个红点（桶心）时：用于轨迹线、加大绘制与点击/拖动时吸附
        let magnet: Option<(usize, Pos2, f32)> = (|| {
            if !strip_response.hovered() || bn == 0 {
                return None;
            }
            let hp = strip_response.hover_pos()?;
            let r2 = MAGNET_PX * MAGNET_PX;
            let mut best: Option<(usize, f32, f32)> = None; // (bi, d2, t_snap_01)
            for bi in 0..b_pos.len() {
                let is_hit = b_pos.get(bi).copied().unwrap_or(false)
                    || b_neg.get(bi).copied().unwrap_or(false);
                if !is_hit {
                    continue;
                }
                let t_snap = (bi as f32 + 0.5) / bn as f32;
                let x = bar_left + t_snap * bar_w;
                let p = Pos2::new(x, bar_cy);
                let d2 = (hp - p).length_sq();
                if d2 < r2 && best.map_or(true, |b| d2 < b.1) {
                    best = Some((bi, d2, t_snap));
                }
            }
            best.map(|(bi, d2, t_snap)| {
                let s = 1.0 - d2.sqrt() / MAGNET_PX;
                (bi, Pos2::new(bar_left + t_snap * bar_w, bar_cy), s.clamp(0.0, 1.0))
            })
        })();
        if strip_response.hovered() && magnet.as_ref().is_some_and(|m| m.2 > 0.02) {
            ui.ctx().request_repaint();
        }

        let play_x =
            track.left() + (self.current_index as f32 + 0.5) / total_images as f32 * track.width();
        let play_x = play_x.clamp(track.left() + 2.0, track.right() - 2.0);
        let done_rect = Rect::from_min_max(
            track.left_top(),
            Pos2::new(play_x, track.bottom()),
        );
        painter.rect_filled(
            done_rect,
            999.0,
            color_alpha(primary, (122.0 + hover * 26.0 + active * 24.0) as u8),
        );
        if bn > 0 {
            if let (Some(m), Some(hp)) = (&magnet, strip_response.hover_pos()) {
                if m.2 > 0.12 {
                    painter.line_segment(
                        [hp, m.1],
                        Stroke::new(
                            1.0 + 0.5 * m.2 * mag_pulse,
                            color_alpha(
                                magnet_amber,
                                (50.0 + 130.0 * m.2 * mag_pulse) as u8,
                            ),
                        ),
                    );
                }
            }
            for bi in 0..b_pos.len() {
                let pos_hit = b_pos.get(bi).copied().unwrap_or(false);
                let neg_hit = b_neg.get(bi).copied().unwrap_or(false);
                if !pos_hit && !neg_hit {
                    continue;
                }
                let t_snap = (bi as f32 + 0.5) / bn as f32;
                let x = bar_left + t_snap * bar_w;
                let p = Pos2::new(x, bar_cy);
                let (dot, stroke_hi) = if pos_hit {
                    (red, red)
                } else {
                    (neg_blue, neg_blue_stroke)
                };
                let is_mag = magnet
                    .as_ref()
                    .is_some_and(|(mb, _, s)| *mb == bi && *s > 0.12);
                let s_mag = if is_mag {
                    magnet
                        .as_ref()
                        .filter(|(mb, _, _)| *mb == bi)
                        .map(|(_, _, s)| *s)
                } else {
                    None
                }
                .unwrap_or(0.0);
                if is_mag {
                    let ph = 0.35 + 0.65 * mag_pulse;
                    for (k, er) in [(0.0_f32, 9.0), (0.6, 6.0), (1.2, 3.0)] {
                        let rr = (er * (0.6 + 0.45 * s_mag) + 2.0 * ph) * (1.0 - k * 0.1);
                        painter.circle_stroke(
                            p,
                            rr,
                            Stroke::new(
                                0.7,
                                color_alpha(
                                    magnet_amber,
                                    ((22.0 + 55.0 * s_mag) * (1.0 - k * 0.2)) as u8,
                                ),
                            ),
                        );
                    }
                }
                let base_r = if is_mag {
                    3.0 + 2.0 * mag_pulse * s_mag + hover * 0.35
                } else {
                    3.0 + hover * 0.35
                };
                painter.circle_filled(p, base_r, color_alpha(dot, 236));
                if is_mag {
                    painter.circle_filled(
                        p,
                        1.6 * (0.85 + 0.15 * mag_pulse),
                        color_alpha(Color32::from_rgb(255, 255, 240), 230),
                    );
                }
                let stroke_c = if is_mag {
                    color_alpha(magnet_amber, (140.0 + 100.0 * mag_pulse) as u8)
                } else {
                    color_alpha(stroke_hi, (102.0 + hover * 34.0) as u8)
                };
                painter.circle_stroke(
                    p,
                    4.6 + hover * 0.9 + if is_mag { 2.0 + 2.0 * mag_pulse * s_mag } else { 0.0 },
                    Stroke::new(0.9, stroke_c),
                );
            }
        }
        painter.rect_stroke(
            track,
            999.0,
            Stroke::new(
                1.05 + hover * 0.35,
                color_alpha(primary_edge, (132.0 + hover * 72.0 + active * 18.0) as u8),
            ),
        );
        let left_cap = Pos2::new(track.left(), track.center().y);
        let right_cap = Pos2::new(track.right(), track.center().y);
        painter.circle_filled(left_cap, track.height() * 0.5, color_alpha(primary, 72));
        painter.circle_filled(right_cap, track.height() * 0.5, color_alpha(primary, 42));
        let knob = Pos2::new(play_x, track.center().y);
        painter.circle_filled(
            knob,
            5.6 + hover * 0.8 + active * 1.2,
            color_alpha(primary_edge, 228),
        );
        painter.circle_filled(
            knob,
            2.6 + hover * 0.3,
            color_alpha(Color32::from_rgb(245, 250, 247), 248),
        );
        painter.circle_stroke(
            knob,
            8.2 + hover * 1.6 + active * 1.2,
            Stroke::new(1.0 + hover * 0.35, color_alpha(primary, 140)),
        );

        if strip_response.clicked() || strip_response.dragged() {
            if let Some(pos) = strip_response.interact_pointer_pos() {
                let t_raw = ((pos.x - bar_left) / bar_w.max(1.0e-3)).clamp(0.0, 1.0);
                let t_use = if bn > 0 {
                    let r2 = MAGNET_PX * MAGNET_PX;
                    let mut best: Option<(f32, f32)> = None;
                    for bi in 0..b_pos.len() {
                        let is_hit = b_pos.get(bi).copied().unwrap_or(false)
                            || b_neg.get(bi).copied().unwrap_or(false);
                        if !is_hit {
                            continue;
                        }
                        let t_snap = (bi as f32 + 0.5) / bn as f32;
                        let dot = Pos2::new(bar_left + t_snap * bar_w, bar_cy);
                        let d2 = (pos - dot).length_sq();
                        if d2 < r2 && best.map_or(true, |b| d2 < b.1) {
                            best = Some((t_snap, d2));
                        }
                    }
                    best.map(|(t, _)| t).unwrap_or(t_raw)
                } else {
                    t_raw
                };
                let idx = (t_use * total_images as f32).floor() as usize;
                self.go_to_image_index(idx.min(total_images.saturating_sub(1)));
            }
        }
        strip_response.on_hover_text(format!(
            "视频进度：点击或拖动跳转，靠近圆点磁吸；绿条=当前；红=有框；蓝=负样本。有框 {} 帧，仅负样本 {} 帧。",
            self.dataset_n_with_boxes, self.dataset_n_neg_only
        ));
    }

    fn ui_video_load_progress_window(&mut self, ctx: &Context) {
        let progress = self.video_load_progress.clamp(0.0, 1.0);
        egui::Window::new("载入视频")
            .collapsible(false)
            .resizable(false)
            .order(Order::Foreground)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("正在读取完整帧")
                            .strong()
                            .size(18.0)
                            .color(theme::TEXT),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(self.video_load_status.as_str())
                            .small()
                            .color(theme::TEXT_MUTED),
                    );
                    ui.add_space(10.0);
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(340.0)
                            .show_percentage()
                            .animate(true),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("视频较大时需要一些时间，完成后会自动进入逐帧标注。")
                            .small()
                            .color(theme::TEXT_MUTED),
                    );
                });
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
        self.poll_video_load(ctx);
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

        // 不得在 `ctx.input` 闭包内再调用 `ctx.memory` / `ctx.input`，否则会死锁（egui 文档）。
        let label_te_focused = ctx.memory(|m| m.has_focus(label_draft_textedit_id()));
        ctx.input(|i| {
            let esc = i.key_pressed(egui::Key::Escape);
            let q_label_cancel = self.show_label_window
                && i.key_pressed(egui::Key::Q)
                && !label_te_focused;
            if self.show_label_window && (esc || q_label_cancel) {
                if q_label_cancel {
                    self.suppress_bbox_q_delete_once = true;
                }
                self.cancel_label_dialog();
            } else if esc {
                if self.scribble_active {
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

        if self.video_load_busy {
            self.ui_video_load_progress_window(ctx);
        }

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
                        if ui
                            .button("确定（空格）")
                            .on_hover_text("输入框聚焦时按回车也可确定")
                            .clicked()
                            || confirm_keys
                        {
                            self.finalize_pending_with_label();
                        }
                        if ui
                            .button("取消（q）")
                            .on_hover_text(
                                "新框命名时会直接取消这次新框；修改已有框标签时只会取消本次编辑。按 Esc 效果相同。输入框聚焦时请先点别处再按 Q，以免误输入。",
                            )
                            .clicked()
                        {
                            self.cancel_label_dialog();
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
            .with_title("YoloVet"),
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
