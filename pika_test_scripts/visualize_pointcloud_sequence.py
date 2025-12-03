import os
import argparse
import numpy as np
import math
import matplotlib
# Commenting out forced Agg backend to allow interactive display when --show
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except ImportError:
    o3d = None

# Backend / headless handling
def init_backend(requested_backend: str, want_show: bool) -> bool:
    headless = (os.environ.get('DISPLAY') is None)
    if not want_show:
        return False
    if requested_backend.lower() == 'auto':
        if headless:
            print('[WARN] Headless environment detected (no DISPLAY); disabling --show.')
            return False
        # Try Qt5Agg then fallback
        for b in ['Qt5Agg', 'TkAgg']:
            try:
                matplotlib.use(b)  # This may reload backend if early enough
                print(f'[INFO] Using backend {b}')
                return True
            except Exception:
                continue
        print('[WARN] No interactive backend available; disabling --show.')
        return False
    else:
        if headless and requested_backend.lower() != 'agg':
            print(f'[WARN] Requested backend {requested_backend} but headless; forcing Agg and disabling --show.')
            matplotlib.use('Agg')
            return False
        try:
            matplotlib.use(requested_backend)
            print(f'[INFO] Using backend {requested_backend}')
            return not headless and requested_backend.lower() != 'agg'
        except Exception as e:
            print(f'[WARN] Failed to set backend {requested_backend}: {e}; using Agg.')
            matplotlib.use('Agg')
            return False


def load_point_cloud(path):
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Failed loading {path}: {e}")
        return None
    # Handle possible dict or object arrays
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.size == 1:
            arr = arr.item()
    if isinstance(arr, dict):
        for k in ['points', 'pointcloud', 'pc', 'xyz']:
            if k in arr:
                arr = arr[k]
                break
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        print(f"[WARN] Unexpected shape {arr.shape} in {path}")
        return None
    return arr  # keep all columns (XYZ[RGB] if present)


def downsample(pc, max_points):
    if max_points and pc.shape[0] > max_points:
        idx = np.random.choice(pc.shape[0], max_points, replace=False)
        return pc[idx]
    return pc


def figure_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)
    except AttributeError:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[:, :, 1:4]
    # Convert RGB->BGR without cv2
    return img[:, :, ::-1]


def render_views(pc, timestamp, views=('top','front','side'), produce_image=True):
    n = len(views)
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    # Axis limits (shared) for consistency
    xmin, ymin, zmin = pc.min(axis=0)
    xmax, ymax, zmax = pc.max(axis=0)
    # Pad
    pad = 0.02 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-6)
    for ax, view in zip(axes, views):
        if view == 'top':
            ax.scatter(pc[:,0], pc[:,1], s=1, c=pc[:,2], cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Top XY')
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
        elif view == 'front':
            ax.scatter(pc[:,0], pc[:,2], s=1, c=pc[:,1], cmap='magma')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title('Front XZ')
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(zmin - pad, zmax + pad)
        elif view == 'side':
            ax.scatter(pc[:,1], pc[:,2], s=1, c=pc[:,0], cmap='plasma')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_title('Side YZ')
            ax.set_xlim(ymin - pad, ymax + pad)
            ax.set_ylim(zmin - pad, zmax + pad)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=8)
    fig.suptitle(f'Time {timestamp}  Points {pc.shape[0]}', fontsize=12)
    if not produce_image:
        plt.show(block=False)
        return fig, None
    img = figure_to_bgr(fig)
    plt.close(fig)
    return fig, img


def build_video(frames, out_path, fps):
    if not frames:
        print('[ERROR] No frames to write.')
        return
    import cv2  # local import only when needed
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            import cv2 as _cv2
            f = _cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()
    print(f'[OK] Video saved: {out_path}')


def show_open3d_sequence(point_clouds, timestamps, pause, limit=None, axis_size=0.2, axis_origin=(0.0,0.0,0.0)):
    if o3d is None:
        print('[WARN] open3d not installed; falling back to matplotlib.')
        for pc, ts in zip(point_clouds, timestamps):
            render_views(pc[:, :3], ts, produce_image=False)
        return
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointCloud Sequence', width=1280, height=720)
    # Add coordinate frame at origin
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=axis_origin)
    vis.add_geometry(axis)
    geom = None
    for idx, (pc, ts) in enumerate(zip(point_clouds, timestamps)):
        if limit and idx >= limit:
            break
        pcd = o3d.geometry.PointCloud()
        xyz = pc[:, :3].astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # Color: prefer columns 3:6 as RGB
        if pc.shape[1] >= 6:
            col = pc[:, 3:6].astype(np.float64)
            maxv = col.max()
            if maxv > 1.5:
                col = col / 255.0
            col = np.clip(col, 0.0, 1.0)
        else:
            # fallback single color
            col = np.tile(np.array([[0.1, 0.7, 1.0]]), (xyz.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(col)
        if geom is None:
            geom = pcd
            vis.add_geometry(geom)
        else:
            geom.points = pcd.points
            geom.colors = pcd.colors
            vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        vis.get_render_option().point_size = 2.0
        print(f'[SHOW] {ts} points={xyz.shape[0]}')
        import time; time.sleep(pause)
    print('[OK] Finished Open3D sequence; close window to exit.')
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Visualize sequence of point cloud npy files as a video.')
    parser.add_argument('--dir', default='pika_raw_data/episode0/camera/pointCloud/pikaDepthCamera_l-normalization', help='Directory containing .npy point clouds')
    parser.add_argument('--pattern', default='.npy', help='Filename pattern to include')
    parser.add_argument('--skip', type=int, default=1, help='Load every Nth file')
    parser.add_argument('--max-points', type=int, default=20000, help='Random downsample per cloud')
    parser.add_argument('--views', type=str, default='top,front,side', help='Comma separated subset of top,front,side')
    parser.add_argument('--fps', type=float, default=30, help='Output video FPS')
    parser.add_argument('--out', type=str, default='pika-videos/pointcloud_sequence_l.mp4', help='Output video path')
    parser.add_argument('--save-frames', action='store_true', help='Also save individual frame images next to video')
    parser.add_argument('--show', action='store_true', help='Show frames interactively instead of saving video')
    parser.add_argument('--pause', type=float, default=0.1, help='Pause seconds between shown frames when --show')
    parser.add_argument('--backend', type=str, default='auto', help='Matplotlib backend: auto|Qt5Agg|TkAgg|Agg')
    parser.add_argument('--limit-show', type=int, default=50, help='Max frames to iterate in Open3D window when --show')
    parser.add_argument('--axis-size', type=float, default=0.2, help='Size (length) of coordinate frame axes in Open3D viewer')
    parser.add_argument('--axis-origin', type=float, nargs=3, default=(0.0,0.0,0.0), help='Origin (x y z) for coordinate frame')
    args = parser.parse_args()

    use_o3d = bool(args.show and (o3d is not None))
    if not use_o3d and args.show:
        # Only init matplotlib interactive backend if not using Open3D
        show_enabled = init_backend(args.backend, args.show)
        args.show = args.show and show_enabled
    else:
        # Using Open3D path, do not initialize matplotlib interactive backends
        pass

    if args.show:
        frame_dir = None  # never save when showing
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        frame_dir = None

    views = [v.strip() for v in args.views.split(',') if v.strip()]

    all_files = [f for f in os.listdir(args.dir) if f.endswith(args.pattern)]
    def ts_key(name):
        base = os.path.splitext(name)[0]
        try:
            return float(base)
        except:
            return base
    all_files.sort(key=ts_key)

    frames = []
    persistent_fig = None
    seq_pcs, seq_ts = [], []
    for i, fname in enumerate(all_files):
        if i % args.skip != 0:
            continue
        path = os.path.join(args.dir, fname)
        pc = load_point_cloud(path)
        if pc is None:
            continue
        pc = downsample(pc, args.max_points)
        timestamp = os.path.splitext(fname)[0]

        if use_o3d:
            # Only collect for Open3D; no matplotlib figures created
            seq_pcs.append(pc)
            seq_ts.append(timestamp)
        else:
            fig, img = render_views(pc[:, :3], timestamp, views, produce_image=not args.show)
            if args.show:
                persistent_fig = fig
                plt.pause(args.pause)
            else:
                frames.append(img)
        if (i+1) % 50 == 0:
            print(f'[INFO] Processed {i+1}/{len(all_files)} files')

    if use_o3d:
        show_open3d_sequence(seq_pcs, seq_ts, args.pause, limit=args.limit_show, axis_size=args.axis_size, axis_origin=tuple(args.axis_origin))
    elif not args.show:
        build_video(frames, args.out, args.fps)
    else:
        print('[OK] Interactive display finished. Close figures to exit.')
        if persistent_fig is not None:
            plt.show()


if __name__ == '__main__':
    main()



