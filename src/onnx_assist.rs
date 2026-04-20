//! Ultralytics YOLO ONNX inference helper in Rust.
//! Supports both:
//! - export(..., nms=True): output Nx6 (x1,y1,x2,y2,conf,cls)
//! - export(..., nms=False): output raw predictions, then class-wise NMS in Rust

use image::{imageops::FilterType, RgbImage, RgbaImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use std::cmp::Ordering;
use std::path::Path;

const INPUT_SIZE: u32 = 640;

/// Keep default close to Ultralytics predict default behavior.
pub const ASSIST_ONNX_CONF: f32 = 0.25;
const ASSIST_ONNX_IOU: f32 = 0.7;

#[derive(Clone, Debug)]
pub struct AssistDetection {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub model_class_id: usize,
    pub conf: f32,
}

pub fn load_session(onnx_path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("ONNX SessionBuilder: {e:?}"))?
        .commit_from_file(onnx_path)
        .map_err(|e| format!("加载 ONNX 失败: {e:?}"))
}

#[inline]
fn ultralytics_gain_pad(w0: u32, h0: u32) -> (f32, f32, f32) {
    let img1_h = INPUT_SIZE as f32;
    let img1_w = INPUT_SIZE as f32;
    let gain = (img1_h / h0 as f32).min(img1_w / w0 as f32);
    let pad_x = ((img1_w - (w0 as f32 * gain).round()) / 2.0 - 0.1).round();
    let pad_y = ((img1_h - (h0 as f32 * gain).round()) / 2.0 - 0.1).round();
    (gain, pad_x, pad_y)
}

fn letterbox_nchw(rgba: &RgbaImage) -> Result<(Array4<f32>, f32, f32, f32), String> {
    let (w0, h0) = rgba.dimensions();
    if w0 == 0 || h0 == 0 {
        return Err("图像尺寸为 0".to_string());
    }
    let (gain, pad_x, pad_y) = ultralytics_gain_pad(w0, h0);
    let nw = (w0 as f32 * gain).round().max(1.0) as u32;
    let nh = (h0 as f32 * gain).round().max(1.0) as u32;

    let rgb: RgbImage = RgbImage::from_fn(w0, h0, |x, y| {
        let p = rgba.get_pixel(x, y);
        image::Rgb([p[0], p[1], p[2]])
    });
    // Close to OpenCV INTER_LINEAR / Ultralytics LetterBox default.
    let resized = image::imageops::resize(&rgb, nw, nh, FilterType::Triangle);

    let fill = 114.0_f32 / 255.0;
    let mut arr = Array4::<f32>::from_elem((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize), fill);

    let top = pad_y as i64;
    let left = pad_x as i64;
    for y in 0..nh {
        for x in 0..nw {
            let p = resized.get_pixel(x, y);
            let yy = top + y as i64;
            let xx = left + x as i64;
            if yy < 0 || xx < 0 || yy >= INPUT_SIZE as i64 || xx >= INPUT_SIZE as i64 {
                continue;
            }
            let uy = yy as usize;
            let ux = xx as usize;
            arr[[0, 0, uy, ux]] = p[0] as f32 / 255.0;
            arr[[0, 1, uy, ux]] = p[1] as f32 / 255.0;
            arr[[0, 2, uy, ux]] = p[2] as f32 / 255.0;
        }
    }

    Ok((arr, gain, pad_x, pad_y))
}

fn scale_to_original(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    gain: f32,
    pad_x: f32,
    pad_y: f32,
    w0: u32,
    h0: u32,
) -> (f32, f32, f32, f32) {
    let x1 = ((x1 - pad_x) / gain).clamp(0.0, w0 as f32);
    let x2 = ((x2 - pad_x) / gain).clamp(0.0, w0 as f32);
    let y1 = ((y1 - pad_y) / gain).clamp(0.0, h0 as f32);
    let y2 = ((y2 - pad_y) / gain).clamp(0.0, h0 as f32);
    let (min_x, max_x) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
    let (min_y, max_y) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
    (min_x, min_y, max_x, max_y)
}

fn parse_nms_output(
    shape: &[i64],
    data: &[f32],
    conf_min: f32,
    w0: u32,
    h0: u32,
    gain: f32,
    pad_x: f32,
    pad_y: f32,
) -> Option<Vec<AssistDetection>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();

    let push_row = |mut x1: f32,
                    mut y1: f32,
                    mut x2: f32,
                    mut y2: f32,
                    conf: f32,
                    cls_f: f32|
     -> Option<AssistDetection> {
        if conf < conf_min {
            return None;
        }
        if x1.abs() < 1e-6
            && y1.abs() < 1e-6
            && x2.abs() < 1e-6
            && y2.abs() < 1e-6
            && conf < 1e-6
        {
            return None;
        }
        let mv = x1.max(y1).max(x2).max(y2);
        if mv > 0.0 && mv <= 1.5 {
            let s = INPUT_SIZE as f32;
            x1 *= s;
            y1 *= s;
            x2 *= s;
            y2 *= s;
        }
        let cls = cls_f.round().clamp(0.0, 1_000_000.0) as usize;
        let (min_x, min_y, max_x, max_y) =
            scale_to_original(x1, y1, x2, y2, gain, pad_x, pad_y, w0, h0);
        if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
            return None;
        }
        Some(AssistDetection {
            min_x,
            min_y,
            max_x,
            max_y,
            model_class_id: cls,
            conf,
        })
    };

    let mut out = Vec::new();
    match dims.as_slice() {
        [_, n, six] if *six == 6 => {
            let n = *n;
            for i in 0..n {
                let base = i * 6;
                if base + 5 >= data.len() {
                    break;
                }
                if let Some(d) = push_row(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                ) {
                    out.push(d);
                }
            }
            Some(out)
        }
        [n, six] if *six == 6 => {
            let n = *n;
            for i in 0..n {
                let base = i * 6;
                if base + 5 >= data.len() {
                    break;
                }
                if let Some(d) = push_row(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                ) {
                    out.push(d);
                }
            }
            Some(out)
        }
        [_, six, n] if *six == 6 => {
            let n = *n;
            if 6 * n > data.len() {
                return None;
            }
            for i in 0..n {
                let x1 = data[i];
                let y1 = data[n + i];
                let x2 = data[2 * n + i];
                let y2 = data[3 * n + i];
                let conf = data[4 * n + i];
                let cls_f = data[5 * n + i];
                if let Some(d) = push_row(x1, y1, x2, y2, conf, cls_f) {
                    out.push(d);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

#[inline]
fn iou_xyxy(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let aw = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let bw = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = aw + bw - inter;
    if union <= 1e-6 {
        0.0
    } else {
        inter / union
    }
}

fn parse_raw_output_with_nms(
    shape: &[i64],
    data: &[f32],
    conf_min: f32,
    w0: u32,
    h0: u32,
    gain: f32,
    pad_x: f32,
    pad_y: f32,
) -> Option<Vec<AssistDetection>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    let (channels, n, row_layout) = match dims.as_slice() {
        [_, c, n] if *c >= 6 => (*c, *n, true),
        [c, n] if *c >= 6 => (*c, *n, true),
        [_, n, c] if *c >= 6 => (*c, *n, false),
        [n, c] if *c >= 6 => (*c, *n, false),
        _ => return None,
    };
    if channels.saturating_mul(n) > data.len() {
        return None;
    }

    let get = |ch: usize, i: usize| -> f32 {
        if row_layout {
            data[ch * n + i]
        } else {
            data[i * channels + ch]
        }
    };

    let mut candidates = Vec::<AssistDetection>::new();
    for i in 0..n {
        let mut cx = get(0, i);
        let mut cy = get(1, i);
        let mut bw = get(2, i);
        let mut bh = get(3, i);
        let mv = cx.max(cy).max(bw).max(bh);
        if mv > 0.0 && mv <= 1.5 {
            let s = INPUT_SIZE as f32;
            cx *= s;
            cy *= s;
            bw *= s;
            bh *= s;
        }
        if bw <= 1e-6 || bh <= 1e-6 {
            continue;
        }

        // YOLOv8/11 ONNX raw layout: [x, y, w, h, cls0, cls1, ...]
        let mut best_conf = 0.0_f32;
        let mut best_cls = 0usize;
        for ch in 4..channels {
            let conf = get(ch, i);
            if conf > best_conf {
                best_conf = conf;
                best_cls = ch - 4;
            }
        }
        if best_conf < conf_min {
            continue;
        }

        let x1 = cx - bw * 0.5;
        let y1 = cy - bh * 0.5;
        let x2 = cx + bw * 0.5;
        let y2 = cy + bh * 0.5;
        let (min_x, min_y, max_x, max_y) =
            scale_to_original(x1, y1, x2, y2, gain, pad_x, pad_y, w0, h0);
        if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
            continue;
        }

        candidates.push(AssistDetection {
            min_x,
            min_y,
            max_x,
            max_y,
            model_class_id: best_cls,
            conf: best_conf,
        });
    }

    if candidates.is_empty() {
        return Some(Vec::new());
    }

    // Class-wise NMS, close to Ultralytics default behavior.
    candidates.sort_by(|a, b| {
        b.conf
            .partial_cmp(&a.conf)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.model_class_id.cmp(&b.model_class_id))
    });
    let mut keep = Vec::<AssistDetection>::new();
    let mut removed = vec![false; candidates.len()];
    for i in 0..candidates.len() {
        if removed[i] {
            continue;
        }
        let cur = candidates[i].clone();
        let cur_box = (cur.min_x, cur.min_y, cur.max_x, cur.max_y);
        keep.push(cur.clone());
        for j in (i + 1)..candidates.len() {
            if removed[j] {
                continue;
            }
            if candidates[j].model_class_id != cur.model_class_id {
                continue;
            }
            let other = (
                candidates[j].min_x,
                candidates[j].min_y,
                candidates[j].max_x,
                candidates[j].max_y,
            );
            if iou_xyxy(&cur_box, &other) > ASSIST_ONNX_IOU {
                removed[j] = true;
            }
        }
    }
    Some(keep)
}

fn parse_output_auto(
    shape: &[i64],
    data: &[f32],
    conf_min: f32,
    w0: u32,
    h0: u32,
    gain: f32,
    pad_x: f32,
    pad_y: f32,
) -> Result<Vec<AssistDetection>, String> {
    if let Some(v) = parse_nms_output(shape, data, conf_min, w0, h0, gain, pad_x, pad_y) {
        return Ok(v);
    }
    if let Some(v) = parse_raw_output_with_nms(shape, data, conf_min, w0, h0, gain, pad_x, pad_y) {
        return Ok(v);
    }
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    Err(format!(
        "不支持的 ONNX 输出形状 {dims:?}。请使用 Ultralytics 导出：model.export(format='onnx', nms=False, batch=1)；也兼容 nms=True 的 Nx6 输出。"
    ))
}

pub fn predict_with_session(
    session: &mut Session,
    image_path: &Path,
    conf_min: f32,
) -> Result<Vec<AssistDetection>, String> {
    let rgba = image::open(image_path)
        .map_err(|e| format!("打开图像失败: {e}"))?
        .to_rgba8();
    let (w0, h0) = rgba.dimensions();
    let (tensor, gain, pad_x, pad_y) = letterbox_nchw(&rgba)?;
    let input_name = session
        .inputs()
        .first()
        .ok_or_else(|| "模型无输入".to_string())?
        .name()
        .to_string();

    let shape_ix: [usize; 4] = [1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize];
    let input_view = TensorRef::from_array_view((
        shape_ix,
        tensor
            .as_slice()
            .ok_or_else(|| "输入张量非连续内存，无法提交 ONNX".to_string())?,
    ))
    .map_err(|e| format!("构建输入张量失败: {e:?}"))?;
    let outputs = session
        .run(ort::inputs![input_name.as_str() => input_view])
        .map_err(|e| format!("ONNX 推理失败: {e:?}"))?;

    let out0 = &outputs[0];
    let (shape, data) = out0
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取输出张量失败: {e:?}"))?;

    parse_output_auto(shape, data, conf_min, w0, h0, gain, pad_x, pad_y)
}
