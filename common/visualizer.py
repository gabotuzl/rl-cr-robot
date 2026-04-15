"""
visualizer.py
-------------
Renders a .mp4 video of the continuum robot's motion from position history.
 
Usage in test.py:
    from visualizer import render_episode
 
    render_episode(
        position_data=rod_data['position'],
        target_position=target_pos,
        output_path="test_results/videos/test_run_1.mp4",
    )
"""
 
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot — prevents display errors on headless servers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3D projection
from moviepy import VideoClip
 
 
def render_episode(
    position_data: list,
    target_position: np.ndarray,
    output_path: str,
    base_length: float = 0.25,
    fps: float = 10.0,
):
    """
    Render a video of one episode from position history.
 
    Parameters
    ----------
    position_data    : list of frames, each frame is an array of shape (3, n_nodes)
                       this is rod_data['position'] directly from the callback
    target_position  : (3,) array — rendered as a red star marker
    output_path      : full path to output .mp4 file
    base_length      : rod length in metres, used for axis limits
    fps              : frames per second of the output video
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
 
    position_array = np.array(position_data)  # (n_frames, 3, n_nodes)
    n_frames = len(position_array)
    duration = n_frames / fps
 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frame_idx = [0]  # list so the closure can mutate it
 
    def make_frame(t: float) -> np.ndarray:
        idx = frame_idx[0]
        ax.clear()
 
        xs = position_array[idx][0]
        ys = position_array[idx][1]
        zs = position_array[idx][2]
 
        # Rod body
        ax.plot(xs, ys, zs, color="steelblue", linewidth=2.0)
        ax.scatter(xs, ys, zs, c="steelblue", s=8)
 
        # Tip marker and annotation
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="blue", s=50, zorder=5)
        ax.text(xs[-1], ys[-1], zs[-1],
                f"  ({xs[-1]:.3f}, {ys[-1]:.3f}, {zs[-1]:.3f})",
                color="blue", fontsize=7)
 
        # Target marker
        ax.scatter(*target_position, c="red", s=100, marker="*", zorder=6, label="Target")
 
        # Distance in title
        dist = np.linalg.norm(np.array([xs[-1], ys[-1], zs[-1]]) - target_position)
        ax.set_title(f"Frame {idx + 1}/{n_frames}  |  Dist to target: {dist:.4f} m", fontsize=9)
 
        # Axis formatting
        ax.set_xlim(-base_length / 2, base_length)
        ax.set_ylim(-base_length, base_length)
        ax.set_zlim(-base_length, base_length)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=7)
 
        frame_idx[0] = min(idx + 1, n_frames - 1)
 
        canvas = FigureCanvas(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3]
 
    clip = VideoClip(make_frame, duration=duration)
    clip.write_videofile(output_path, codec="libx264", fps=fps, logger=None)
    plt.close(fig)
    print(f"Video saved: {output_path}")
 