"""
3D Face Point Cloud Generator
=============================
Author: Viraksak KOY (@hustsak)
Theme: AI / Data Science / Computer Vision
Visual Style: Cinematic Black + Silver

Generates a seamless looping 3D particle/point-cloud reconstruction of
Viraksak KOY's face from portrait imagery.

Requirements:
    numpy, pillow, opencv-python
"""

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw


def generate_face_pointcloud(
    input_path="assets/portrait.jpg",
    output_gif="assets/face-pointcloud.gif",
    output_preview="assets/face-pointcloud-preview.png",
    num_frames=64,
    canvas_w=1000,
    canvas_h=420,
    fps=16,
):
    print(f"[1/5] Loading portrait image from: {input_path}")
    if not os.path.exists(input_path):
        if os.path.exists("original.png"):
            print("  Falling back to original.png...")
            input_path = "original.png"
        else:
            raise FileNotFoundError(f"Cannot find input portrait at {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image from {input_path}")

    h_orig, w_orig, _ = img.shape

    # Crop tightly and symmetrically around head, face and neck
    crop_y1 = max(0, int(h_orig * 0.04))
    crop_y2 = min(h_orig, int(h_orig * 0.80))
    center_x = int(w_orig * 0.505)
    half_w = int(w_orig * 0.41)
    crop_x1 = max(0, center_x - half_w)
    crop_x2 = min(w_orig, center_x + half_w)

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    h, w, _ = cropped.shape
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    Y, X = np.indices((h, w))

    # Segment foreground from pure white studio background
    bg_mask = (cropped[:, :, 0] > 238) & (cropped[:, :, 1] > 238) & (cropped[:, :, 2] > 238)
    fg_mask = ~bg_mask

    print("[2/5] Building anatomical 3D depth field...")
    # Anatomical landmark centers relative to cropped image
    xc = w * 0.508
    yc = h * 0.440

    # 1. Base cranial ellipsoid
    rx = w * 0.32
    ry = h * 0.38
    rad_sq = ((X - xc) / rx) ** 2 + ((Y - yc) / ry) ** 2
    cranial_z = np.zeros((h, w), dtype=np.float32)
    valid = rad_sq < 1.0
    cranial_z[valid] = np.sqrt(1.0 - rad_sq[valid]) * 88.0

    # 2. Nose ridge and tip (dominant Z protrusion)
    nose_x = xc
    nose_tip_y = yc + h * 0.050
    nose_bridge_y = yc - h * 0.030
    nose_tip = np.exp(-(((X - nose_x) / 32.0) ** 2 + ((Y - nose_tip_y) / 40.0) ** 2)) * 56.0
    nose_bridge = np.exp(-(((X - nose_x) / 18.0) ** 2 + ((Y - nose_bridge_y) / 65.0) ** 2)) * 38.0
    nostrils = (
        -np.exp(-(((X - (nose_x - 28.0)) / 14.0) ** 2 + ((Y - (nose_tip_y + 10.0)) / 12.0) ** 2)) * 16.0
        -np.exp(-(((X - (nose_x + 28.0)) / 14.0) ** 2 + ((Y - (nose_tip_y + 10.0)) / 12.0) ** 2)) * 16.0
    )

    # 3. Eye sockets and eyeballs
    eye_y = yc - h * 0.045
    left_eye_x = xc - w * 0.116
    right_eye_x = xc + w * 0.118
    left_socket = -np.exp(-(((X - left_eye_x) / 42.0) ** 2 + ((Y - eye_y) / 28.0) ** 2)) * 26.0
    right_socket = -np.exp(-(((X - right_eye_x) / 42.0) ** 2 + ((Y - eye_y) / 28.0) ** 2)) * 26.0
    left_ball = np.exp(-(((X - left_eye_x) / 20.0) ** 2 + ((Y - eye_y) / 16.0) ** 2)) * 12.0
    right_ball = np.exp(-(((X - right_eye_x) / 20.0) ** 2 + ((Y - eye_y) / 16.0) ** 2)) * 12.0

    # 4. Cheeks (zygomatic arches)
    cheek_y = yc + h * 0.030
    left_cheek = np.exp(-(((X - (xc - w * 0.16)) / 55.0) ** 2 + ((Y - cheek_y) / 65.0) ** 2)) * 22.0
    right_cheek = np.exp(-(((X - (xc + w * 0.16)) / 55.0) ** 2 + ((Y - cheek_y) / 65.0) ** 2)) * 22.0

    # 5. Lips and Chin
    mouth_y = yc + h * 0.145
    chin_y = yc + h * 0.245
    lips = np.exp(-(((X - xc) / 55.0) ** 2 + ((Y - mouth_y) / 25.0) ** 2)) * 24.0
    chin = np.exp(-(((X - xc) / 65.0) ** 2 + ((Y - chin_y) / 45.0) ** 2)) * 30.0

    # 6. Brow ridge
    brow_y = yc - h * 0.100
    brow = np.exp(-(((X - xc) / 165.0) ** 2 + ((Y - brow_y) / 28.0) ** 2)) * 16.0

    # 7. Hair dome
    hair_mask = (gray < 85) & (Y < yc - h * 0.05) & fg_mask
    hair_dome = np.zeros((h, w), dtype=np.float32)
    hair_dome[hair_mask] = np.clip(62.0 - ((Y[hair_mask] - h * 0.20) / 160.0) ** 2 * 25.0, 20.0, 72.0)

    # Combine full 3D depth field
    Z_full = (
        cranial_z
        + nose_tip
        + nose_bridge
        + nostrils
        + left_socket
        + right_socket
        + left_ball
        + right_ball
        + left_cheek
        + right_cheek
        + lips
        + chin
        + brow
        + hair_dome
    )

    # Soft taper towards shoulders/collar
    fade_y = np.clip((Y - h * 0.72) / (h * 0.26), 0.0, 1.0)
    Z_full = Z_full * (1.0 - 0.72 * fade_y)
    Z_full[~fg_mask] = 0.0

    # Compute surface normals for realistic computational lighting
    grad_zy, grad_zx = np.gradient(Z_full)
    norm_len = np.sqrt(grad_zx**2 + grad_zy**2 + 1.0)
    nx = -grad_zx / norm_len
    ny = -grad_zy / norm_len
    nz = 1.0 / norm_len

    light_dir = np.array([0.15, -0.25, 0.95], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    shading = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0.2, 1.0)

    print("[3/5] Sampling multi-strata point cloud...")
    # 1. Base stratified lattice across face and hair
    step = 6
    ys_grid, xs_grid = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    keep_prob = 1.0 - 0.78 * np.clip((ys_grid - h * 0.74) / (h * 0.23), 0, 1)
    rnd = np.random.RandomState(42).uniform(0, 1, size=ys_grid.shape)
    grid_mask = fg_mask[ys_grid, xs_grid] & (rnd < keep_prob)

    grid_ys = ys_grid[grid_mask]
    grid_xs = xs_grid[grid_mask]
    jitter_x = np.random.RandomState(43).uniform(-1.8, 1.8, size=len(grid_xs))
    jitter_y = np.random.RandomState(44).uniform(-1.8, 1.8, size=len(grid_ys))
    sample_xs = np.clip(grid_xs + jitter_x, 0, w - 1).astype(int)
    sample_ys = np.clip(grid_ys + jitter_y, 0, h - 1).astype(int)
    keep = fg_mask[sample_ys, sample_xs]
    sample_xs = sample_xs[keep]
    sample_ys = sample_ys[keep]

    # 2. High-frequency edge points for facial features (focused on head and neck)
    edges = cv2.Canny(gray, 30, 85)
    edge_filter = (edges > 0) & fg_mask & (Y < h * 0.76)
    edge_ys, edge_xs = np.where(edge_filter)
    edge_idx = np.random.RandomState(45).choice(len(edge_xs), size=min(4500, len(edge_xs)), replace=False)
    sample_xs = np.concatenate([sample_xs, edge_xs[edge_idx]])
    sample_ys = np.concatenate([sample_ys, edge_ys[edge_idx]])

    # 3. Dense eye, pupil, and iris points
    eye_box_y1 = int(eye_y - 30)
    eye_box_y2 = int(eye_y + 30)
    eye_filter = (
        ((X >= left_eye_x - 45) & (X <= left_eye_x + 45) & (Y >= eye_box_y1) & (Y <= eye_box_y2))
        | ((X >= right_eye_x - 45) & (X <= right_eye_x + 45) & (Y >= eye_box_y1) & (Y <= eye_box_y2))
    ) & (edges > 0)
    eye_ys, eye_xs = np.where(eye_filter)
    sample_xs = np.concatenate([sample_xs, eye_xs])
    sample_ys = np.concatenate([sample_ys, eye_ys])

    num_pts = len(sample_xs)
    print(f"  Total 3D face points sampled: {num_pts}")

    # Extract depth, shading, luminance for each point
    zs = Z_full[sample_ys, sample_xs]
    shades = shading[sample_ys, sample_xs]
    lums = gray[sample_ys, sample_xs] / 255.0

    # Base silver-white brightness
    base_bright = (0.28 + 0.42 * (zs / zs.max()) + 0.30 * shades) * (0.75 + 0.25 * lums)
    collar_fade = 1.0 - 0.72 * np.clip((sample_ys - h * 0.74) / (h * 0.23), 0, 1)
    base_bright = base_bright * collar_fade

    # Center and scale points to canvas (face occupies ~360px vertically)
    scale = 360.0 / (h * 0.78)
    X3d = (sample_xs - xc) * scale
    Y3d = (sample_ys - (yc + h * 0.035)) * scale
    Z3d = zs * scale

    # Deterministic dispersion vectors for Phase 1 materialization & Phase 6 loop
    np.random.seed(101)
    disp_x = np.random.normal(0, 24.0, num_pts)
    disp_y = np.random.normal(0, 24.0, num_pts)
    disp_z = np.random.normal(0, 32.0, num_pts)

    # Ambient floating dust particles
    ambient_n = 220
    amb_x = np.random.uniform(40, canvas_w - 40, ambient_n)
    amb_y = np.random.uniform(20, canvas_h - 20, ambient_n)
    amb_b = np.random.uniform(25, 80, ambient_n)

    print(f"[4/5] Rendering {num_frames} animation frames...")
    frames = []
    focal = 700.0

    for frame_idx in range(num_frames):
        t = frame_idx / float(num_frames)

        # ----------------------------------------------------
        # PHASE 1 & 6: MATERIALIZATION & SMOOTH LOOP DISPERSION
        # ----------------------------------------------------
        if t < 0.20:
            tau = t / 0.20
            mat = 0.06 + 0.94 * (0.5 - 0.5 * np.cos(np.pi * tau))
            disp_scale = (1.0 - (0.5 - 0.5 * np.cos(np.pi * tau))) ** 1.5
        elif t > 0.82:
            tau = (t - 0.82) / 0.18
            mat = 1.0 - 0.94 * (0.5 - 0.5 * np.cos(np.pi * tau))
            disp_scale = (0.5 - 0.5 * np.cos(np.pi * tau)) ** 1.5
        else:
            mat = 1.0
            disp_scale = 0.0

        # ----------------------------------------------------
        # PHASE 3 & 4: 3D DEPTH & CINEMATIC ROTATION
        # ----------------------------------------------------
        yaw = np.radians(6.0 * np.sin(2.0 * np.pi * t))
        pitch = np.radians(1.4 * np.sin(4.0 * np.pi * t))

        cur_x = X3d + disp_x * disp_scale
        cur_y = Y3d + disp_y * disp_scale
        cur_z = Z3d + disp_z * disp_scale

        # 3D Euler rotation
        X_rot = cur_x * np.cos(yaw) + cur_z * np.sin(yaw)
        Z_temp = -cur_x * np.sin(yaw) + cur_z * np.cos(yaw)
        Y_rot = cur_y * np.cos(pitch) - Z_temp * np.sin(pitch)
        Z_rot = cur_y * np.sin(pitch) + Z_temp * np.cos(pitch)

        # Perspective projection
        proj_x = X_rot * focal / (focal + Z_rot) + canvas_w / 2.0
        proj_y = Y_rot * focal / (focal + Z_rot) + canvas_h / 2.0

        # ----------------------------------------------------
        # PHASE 5: THIN SILVER SCAN LINE (t in [0.35, 0.65])
        # ----------------------------------------------------
        scan_active = False
        scan_x = -999.0
        scan_boost = np.zeros(num_pts, dtype=np.float32)
        if 0.35 <= t <= 0.65:
            scan_active = True
            u = (t - 0.35) / 0.30
            scan_x = canvas_w / 2.0 - 150.0 + 300.0 * u
            scan_dist = np.abs(proj_x - scan_x)
            scan_boost = np.exp(-((scan_dist / 12.0) ** 2)) * 0.45

        cur_bright = np.clip((base_bright * mat + scan_boost), 0.04, 1.0)

        # Pure cinematic black canvas
        img_pil = Image.new("RGB", (canvas_w, canvas_h), (2, 3, 5))
        draw = ImageDraw.Draw(img_pil)

        # Ambient dust in 3D background
        for ax, ay, ab in zip(amb_x, amb_y, amb_b):
            if abs(ax - canvas_w / 2.0) > 150 or ay < 50 or ay > 370:
                val = int(ab * mat)
                draw.point([int(ax), int(ay)], fill=(int(val * 0.9), int(val * 0.95), val))

        # Thin silver scanning beam across face height
        if scan_active and 0 <= scan_x < canvas_w:
            sx_int = int(scan_x)
            for dy in range(40, canvas_h - 40):
                # Vertical fade at top and bottom
                vert_fade = np.sin(np.pi * (dy - 40) / (canvas_h - 80))
                intensity = int(120 * vert_fade)
                draw.point([sx_int, dy], fill=(int(intensity * 0.92), int(intensity * 0.96), intensity))

        # Depth-sorted particle rendering
        order = np.argsort(Z_rot)
        z_norm = np.clip((Z_rot - Z_rot.min()) / (Z_rot.max() - Z_rot.min() + 1e-5), 0.2, 1.0)

        for i in order:
            px, py = proj_x[i], proj_y[i]
            if 2 <= px < canvas_w - 2 and 2 <= py < canvas_h - 2:
                zn = z_norm[i]
                b_val = cur_bright[i]

                # Silver / White palette: crisp highlights, graphite shadows
                sr = int(np.clip(b_val * 238, 12, 255))
                sg = int(np.clip(b_val * 245, 15, 255))
                sb = int(np.clip(b_val * 255, 18, 255))

                if (zn > 0.65 or scan_boost[i] > 0.12) and mat > 0.5:
                    draw.ellipse([px - 1.2, py - 1.2, px + 1.2, py + 1.2], fill=(sr, sg, sb))
                elif zn > 0.35 and mat > 0.3:
                    draw.ellipse([px - 0.8, py - 0.8, px + 0.8, py + 0.8], fill=(sr, sg, sb))
                else:
                    draw.point([px, py], fill=(sr, sg, sb))

        # Save static preview at frame 20 (full reconstruction and 3D depth)
        if frame_idx == 20:
            img_pil.save(output_preview)

        # Convert to adaptive palette for GIF optimization
        frames.append(img_pil.convert("P", palette=Image.ADAPTIVE, colors=128))

    print(f"[5/5] Compiling and optimizing GIF to: {output_gif}")
    duration_ms = int(1000.0 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    file_size_mb = os.path.getsize(output_gif) / (1024 * 1024)
    print(f"Done! Created '{output_gif}' ({file_size_mb:.2f} MB, {num_frames} frames, {num_frames/fps:.1f}s loop)")
    if os.path.exists(output_preview):
        print(f"Created preview image: '{output_preview}'")


if __name__ == "__main__":
    t0 = time.time()
    generate_face_pointcloud()
    print(f"Total execution time: {time.time() - t0:.2f} seconds")
