#!/usr/bin/env python
"""
将每个 episode 的 data.hdf5 中的左右相机 RGB 帧解码并拼接成视频，保存到输出目录。
默认按 RDT2 约定：camera0=右、camera1=左，视频帧按 [右, 左] 横向拼接。

依赖：h5py, numpy, opencv-python/opencv-python-headless
建议在 conda 环境 rdt2 中运行。

用法示例（从仓库根目录运行）：
  python pika_test_scripts/make_episode_videos.py \
    --input-root pika_raw_data --output-dir pika-videos --pattern 'episode*'
可选参数查看 -h
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import h5py
import cv2
from glob import glob


def to_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    # h5py 变量长字节标量常见为 numpy.void 或 np.ndarray(uint8, varlen)
    try:
        return x.tobytes()
    except Exception:
        try:
            # 某些情形为标量 numpy 数组，转为 Python bytes
            return bytes(np.array(x).tobytes())
        except Exception:
            return b""


def to_text(x) -> Optional[str]:
    # 尝试把对象解释为 UTF-8 文本（例如文件名或时间戳字符串）
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray, memoryview)):
        for enc in ('utf-8', 'ascii'):
            try:
                return bytes(x).decode(enc)
            except Exception:
                pass
        return None
    try:
        arr = np.array(x)
        if arr.dtype.kind in {'U'}:
            try:
                return arr.item() if arr.shape == () else str(arr)
            except Exception:
                pass
        if arr.dtype.kind in {'S', 'O'}:
            try:
                return arr.astype('U').item() if arr.shape == () else str(arr.astype('U'))
            except Exception:
                pass
        # 数字转文本
        if arr.shape == () and np.issubdtype(arr.dtype, np.number):
            return str(arr.item())
        # 直接作为字节解码
        b = arr.tobytes()
        for enc in ('utf-8', 'ascii'):
            try:
                return b.decode(enc)
            except Exception:
                pass
        return None
    except Exception:
        return None


def decode_bgr(img_bytes: bytes) -> Optional[np.ndarray]:
    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def hstack_resize_to_same_height(img_right: np.ndarray, img_left: np.ndarray) -> np.ndarray:
    h0, w0 = img_right.shape[:2]
    h1, w1 = img_left.shape[:2]
    if h0 != h1:
        # 以右图高度为基准，等比缩放左图
        new_w1 = int(round(w1 * (h0 / h1)))
        img_left = cv2.resize(img_left, (new_w1, h0), interpolation=cv2.INTER_AREA)
    return np.hstack([img_right, img_left])


def estimate_fps(timestamps: np.ndarray) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 30.0
    dt = np.diff(timestamps)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        return 30.0
    fps = float(np.clip(1.0 / np.median(dt), 1.0, 240.0))
    return fps


def write_video(frames_iter, out_path: Path, fps: float) -> Tuple[int, Tuple[int, int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    size = None
    for frame in frames_iter:
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if writer is None:
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if (w, h) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    if writer is not None:
        writer.release()
    return count, size if size is not None else (0, 0)


def _find_file_by_name(dirpath: Path, name: str) -> Optional[Path]:
    # 清理空字符与空白
    s = name.strip().strip('\x00')
    if not s:
        return None
    p = dirpath / s
    if p.is_file():
        return p
    # 若无扩展名，尝试常见扩展
    base = s
    if '.' not in Path(s).name:
        for ext in ('.jpg', '.jpeg', '.png'):
            p2 = dirpath / (base + ext)
            if p2.is_file():
                return p2
    # 模糊匹配：同名前缀（时间戳）
    stem = Path(s).stem
    candidates = list(dirpath.glob(stem + '.*'))
    if candidates:
        return sorted(candidates)[0]
    return None


def _read_frame_from_ds(ds, idx: int, fallback_dir: Optional[Path], filename_hint: Optional[str]) -> Optional[np.ndarray]:
    # 1) 尝试按字节解码（编码图像）
    x = ds[idx]
    img = decode_bgr(to_bytes(x))
    if img is not None:
        return img
    # 2) 元素可能是已展开的像素数组
    try:
        arr = np.array(x)
        if arr.ndim == 3 and arr.dtype == np.uint8:
            # (H,W,3) 或 (H,W,4)
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.ndim == 2 and arr.dtype == np.uint8:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass
    # 3) 尝试按字符串路径读取
    if fallback_dir is not None:
        name = to_text(x)
        if name:
            p = _find_file_by_name(fallback_dir, name)
            if p is not None:
                img2 = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img2 is not None and img2.size > 0:
                    return img2
    return None


def process_episode(h5_path: Path, out_dir: Path, use_fisheye: bool = False) -> Optional[Path]:
    ep_name = h5_path.parent.name
    with h5py.File(h5_path, 'r') as f:
        # 选择相机键
        cam_group = 'camera/color'
        right_key = 'pikaFisheyeCamera_r' if use_fisheye else 'pikaDepthCamera_r'
        left_key = 'pikaFisheyeCamera_l' if use_fisheye else 'pikaDepthCamera_l'
        ds_right = f[f'{cam_group}/{right_key}']
        ds_left = f[f'{cam_group}/{left_key}']
        timestamps = np.array(f['timestamp']) if 'timestamp' in f else None
        fps = estimate_fps(timestamps)

        # 回退目录（若数据集存的是文件名）
        fallback_right_dir = h5_path.parent / 'camera' / 'color' / right_key
        fallback_left_dir = h5_path.parent / 'camera' / 'color' / left_key

        # 准备输出路径
        suffix = 'fisheye' if use_fisheye else 'rgb'
        out_path = out_dir / f'{ep_name}_{suffix}_right_left.mp4'

        def frames():
            n = min(len(ds_right), len(ds_left))
            for i in range(n):
                img_r = _read_frame_from_ds(ds_right, i, fallback_right_dir, None)
                img_l = _read_frame_from_ds(ds_left, i, fallback_left_dir, None)
                if img_r is None or img_l is None:
                    continue
                yield hstack_resize_to_same_height(img_r, img_l)

        count, size = write_video(frames(), out_path, fps)
        if count == 0:
            return None
        return out_path


def find_episodes(input_root: Path, pattern: str) -> list[Path]:
    eps = []
    for p in sorted(input_root.glob(pattern)):
        h5 = p / 'data.hdf5'
        if h5.is_file():
            eps.append(h5)
    return eps


def main():
    parser = argparse.ArgumentParser(description='将 data.hdf5 中的左右相机帧拼接导出为视频')
    parser.add_argument('--input-root', type=Path, default=Path('pika_raw_data'), help='包含 episode* 的根目录')
    parser.add_argument('--output-dir', type=Path, default=Path('pika-videos'), help='输出视频目录')
    parser.add_argument('--pattern', type=str, default='episode*', help='匹配 episode 目录的 glob 模式')
    parser.add_argument('--use-fisheye', action='store_true', help='使用鱼眼相机而非深度相机的彩色图')
    args = parser.parse_args()

    episodes = find_episodes(args.input_root, args.pattern)
    if not episodes:
        print(f'[WARN] 未找到 episodes 于 {args.input_root} / pattern={args.pattern}')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for h5 in episodes:
        try:
            out = process_episode(h5, args.output_dir, use_fisheye=args.use_fisheye)
            if out is None:
                print(f'[FAIL] {h5} 无有效帧')
                fail += 1
            else:
                print(f'[OK] {h5} → {out}')
                ok += 1
        except Exception as e:
            print(f'[ERROR] {h5}: {e}')
            fail += 1
    print(f'完成：成功 {ok}，失败 {fail}，输出目录：{args.output_dir}')


if __name__ == '__main__':
    main()


