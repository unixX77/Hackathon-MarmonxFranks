"""Synthetic RGB-D LEGO brick scenes in Isaac Sim, written like real D435 captures.

Scene: grey studded LEGO baseplate with black tape strips on a light desk, 2x4 bricks
(31.8 x 15.8 x 9.6 mm + studs) in the five project colours, either loose or stacked in
layers that alternate direction (with a partial top layer). Each scene is rendered from
several random camera poses (30-45 cm above the plate, slight tilt, image up along +x
like the hand-mounted camera) under random lamps.

Every scene is written as a normal recording, so the real pipeline runs on it unchanged:
  <out>/scene_00000/rgb/000000.png         colour image
  <out>/scene_00000/depth/000000.png       uint16 depth in mm, aligned to colour
  <out>/scene_00000/camera_info.json       real D435 colour intrinsics (from --camera-info)
  <out>/scene_00000/instance/000000.png    uint16 brick id per pixel (0 = background)
  <out>/scene_00000/labels/000000.json     ground truth per brick: colour, layer, top layer,
                                           top-face centre in the camera optical frame (m),
                                           image angle (deg, +x toward +y, -90..90], pixels

Run with the Isaac Sim Python (pip environment):
  ~/env_isaacsim/bin/python sim/generate_brick_scenes.py --out data/sim/run1 --scenes 50
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--scenes', type=int, default=20)
parser.add_argument('--views', type=int, default=2, help='camera/light variations per brick layout')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--camera-info', type=Path, default=ROOT / 'data/captures/20261026_124136/camera_info.json',
                    help='real capture camera_info.json whose colour intrinsics are copied')
parser.add_argument('--rt-subframes', type=int, default=16)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must start before other omni imports)

app = SimulationApp({'renderer': 'RealTimePathTracing', 'headless': True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

# Brick and plate geometry (metres). LEGO 2x4 brick body without studs.
BRICK_L, BRICK_W, BRICK_H = 0.0318, 0.0158, 0.0096
STUD_R, STUD_H, PITCH = 0.0024, 0.0017, 0.008
PLATE_X, PLATE_Y = 0.80, 0.50
# Official LEGO colours (sRGB hex), converted to linear for the renderer.
COLOURS = {'red': 'C91A09', 'dark_red': '720E0F', 'dark_orange': 'A95500',
           'medium_nougat': 'AA7D55', 'reddish_brown': '582A12'}
MAX_BRICKS = 48
STACK_COLS, STACK_ROWS = 4, 8          # even layers: 4 bricks along x by 8 along y (128 x 128 mm)


def linear(hex_colour):
    srgb = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
rep.orchestrator.set_capture_on_play(False)
rng = np.random.default_rng(args.seed)
rep.set_global_seed(args.seed)


def preview_material(name, rgb, roughness):
    mat = UsdShade.Material.Define(stage, f'/World/Looks/{name}')
    shader = UsdShade.Shader.Define(stage, f'/World/Looks/{name}/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat, shader


def place(prim, translate=(0, 0, 0), scale=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def box(path, size, centre, material):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    place(cube, centre, size)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


# ---------------------------------------------------------------- static scene
desk_mat, _ = preview_material('desk', linear('E8E6E1'), 0.85)
plate_mat, plate_shader = preview_material('plate', linear('A0A5A9'), 0.55)
tape_mat, _ = preview_material('tape', linear('141414'), 0.35)
box('/World/desk', (2.0, 2.0, 0.02), (0, 0, -0.0132), desk_mat)
box('/World/plate', (PLATE_X, PLATE_Y, 0.0032), (0, 0, -0.0016), plate_mat)
for side in (-1, 1):
    box(f'/World/tape_{side + 1}', (PLATE_X + 0.04, 0.05, 0.0006),
        (0, side * (PLATE_Y / 2 + 0.025), 0.0003), tape_mat)
# Baseplate studs as one point instancer (thousands of cylinders, cheap to render).
stud_proto = UsdGeom.Cylinder.Define(stage, '/World/plate_studs/proto')
stud_proto.CreateRadiusAttr(STUD_R)
stud_proto.CreateHeightAttr(STUD_H)
stud_proto.CreateAxisAttr('Z')
UsdShade.MaterialBindingAPI.Apply(stud_proto.GetPrim()).Bind(plate_mat)
xs = np.arange(-PLATE_X / 2 + PITCH / 2, PLATE_X / 2, PITCH)
ys = np.arange(-PLATE_Y / 2 + PITCH / 2, PLATE_Y / 2, PITCH)
grid = np.array([(x, y, STUD_H / 2) for x in xs for y in ys])
instancer = UsdGeom.PointInstancer.Define(stage, '/World/plate_studs')
instancer.CreatePrototypesRel().SetTargets([stud_proto.GetPath()])
instancer.CreateProtoIndicesAttr(Vt.IntArray([0] * len(grid)))
instancer.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in grid]))

# Brick colour materials (one per colour, roughness re-randomized per view).
brick_mats = {name: preview_material(f'brick_{name}', linear(hexc), 0.3) for name, hexc in COLOURS.items()}

# Pool of bricks: body + 8 studs, re-posed and re-coloured for every scene. Materials are
# bound on the body and stud prims themselves: the renderer does not pick up a changed
# binding inherited from the parent, so re-colouring via the parent left stale colours.
bricks, brick_parts = [], []
for i in range(MAX_BRICKS):
    root = UsdGeom.Xform.Define(stage, f'/World/Bricks/brick_{i:02d}')
    body = UsdGeom.Cube.Define(stage, f'{root.GetPath()}/body')
    body.CreateSizeAttr(1.0)
    place(body, (0, 0, BRICK_H / 2), (BRICK_L, BRICK_W, BRICK_H))
    parts = [body.GetPrim()]
    for k, (sx, sy) in enumerate([(sx, sy) for sx in (-1.5, -0.5, 0.5, 1.5) for sy in (-0.5, 0.5)]):
        stud = UsdGeom.Cylinder.Define(stage, f'{root.GetPath()}/stud_{k}')
        stud.CreateRadiusAttr(STUD_R)
        stud.CreateHeightAttr(STUD_H)
        stud.CreateAxisAttr('Z')
        place(stud, (sx * PITCH, sy * PITCH, BRICK_H + STUD_H / 2))
        parts.append(stud.GetPrim())
    bricks.append(root)
    brick_parts.append([UsdShade.MaterialBindingAPI.Apply(part) for part in parts])

# Lights: dome for ambient plus up to three lamps.
dome = UsdLux.DomeLight.Define(stage, '/World/Lights/dome')
lamps = [UsdLux.SphereLight.Define(stage, f'/World/Lights/lamp_{k}') for k in range(3)]
for light in [dome, *lamps]:
    light.CreateEnableColorTemperatureAttr(True)

# Camera with the real D435 colour intrinsics.
info = json.loads(args.camera_info.read_text())
intr = info['aligned_depth_intrinsics']
W, H, FX, FY, CX, CY = intr['width'], intr['height'], intr['fx'], intr['fy'], intr['ppx'], intr['ppy']
camera = UsdGeom.Camera.Define(stage, '/World/d435')
APERTURE = 20.955
camera.CreateFocalLengthAttr(FX * APERTURE / W)
camera.CreateHorizontalApertureAttr(APERTURE)
camera.CreateVerticalApertureAttr(H * (FX * APERTURE / W) / FY)
# Principal point. Signs verified against rendered masks: the offsets move the image the
# opposite way to the principal point.
camera.CreateHorizontalApertureOffsetAttr(-(CX - W / 2) * APERTURE / W)
camera.CreateVerticalApertureOffsetAttr((CY - H / 2) * APERTURE / W)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
cam_xf = UsdGeom.Xformable(camera)
cam_op = cam_xf.AddTransformOp()

render = rep.create.render_product(str(camera.GetPath()), (W, H))
annotators = {name: rep.annotators.get(name) for name in
              ('rgb', 'distance_to_image_plane', 'instance_id_segmentation')}
for annotator in annotators.values():
    annotator.attach(render)


# ---------------------------------------------------------------- randomization
def loose_layout():
    """1-10 bricks lying on the plate, apart from each other."""
    placed = []
    for _ in range(rng.integers(1, 11)):
        for _attempt in range(50):
            xy = rng.uniform([-0.12, -0.10], [0.12, 0.10])
            if all(np.linalg.norm(xy - p['xy']) > 0.045 for p in placed):
                yaw = rng.choice([0.0, 90.0]) + rng.normal(0, 12) if rng.random() < 0.5 else rng.uniform(0, 180)
                placed.append({'xy': xy, 'yaw': float(yaw), 'layer': 0})
                break
    return placed


def stack_layout():
    """Interlocked stack: full lower layers alternating direction, partial top layer."""
    layers = int(rng.integers(2, 5))
    origin = rng.uniform([-0.06, -0.05], [0.06, 0.05])
    yaw0 = float(rng.choice([0.0, 90.0]) + rng.normal(0, 3))
    rot = np.array([[math.cos(math.radians(yaw0)), -math.sin(math.radians(yaw0))],
                    [math.sin(math.radians(yaw0)), math.cos(math.radians(yaw0))]])
    keep_top = rng.uniform(0.15, 0.85)
    placed = []
    for layer in range(layers):
        cols, rows, along_x = (STACK_COLS, STACK_ROWS, True) if layer % 2 == 0 else (STACK_ROWS, STACK_COLS, False)
        size_x = 0.128 / cols
        size_y = 0.128 / rows
        for c in range(cols):
            for r in range(rows):
                if layer == layers - 1 and rng.random() > keep_top:
                    continue
                local = np.array([-0.064 + (c + 0.5) * size_x, -0.064 + (r + 0.5) * size_y])
                placed.append({'xy': origin + rot @ local, 'yaw': yaw0 + (0.0 if along_x else 90.0),
                               'layer': layer})
    if len(placed) > MAX_BRICKS:  # brick pool limit: drop the lowest (mostly hidden) layer first
        placed = [p for p in placed if p['layer'] >= layers - 2]
    return placed[:MAX_BRICKS]


def look_at(position, target, up_hint):
    forward = np.asarray(target, float) - position
    z_cam = -forward / np.linalg.norm(forward)               # USD camera looks down -Z
    x_cam = np.cross(up_hint, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.column_stack([x_cam, y_cam, z_cam])            # world-from-camera rotation


def set_lights():
    dome.GetIntensityAttr().Set(float(rng.uniform(150, 700)))
    dome.GetColorTemperatureAttr().Set(float(rng.uniform(3500, 7000)))
    active = int(rng.integers(1, 4))
    for k, lamp in enumerate(lamps):
        on = k < active
        lamp.GetIntensityAttr().Set(float(rng.uniform(1.5e4, 8e4)) if on else 0.0)
        lamp.GetRadiusAttr().Set(float(rng.uniform(0.02, 0.15)))
        lamp.GetColorTemperatureAttr().Set(float(rng.uniform(2700, 6500)))
        azimuth, distance = rng.uniform(0, 2 * math.pi), rng.uniform(0.25, 0.8)
        place(lamp, (distance * math.cos(azimuth), distance * math.sin(azimuth), rng.uniform(0.3, 0.9)))
    return active


def set_brick_materials():
    for _, (mat, shader) in brick_mats.items():
        shader.GetInput('roughness').Set(float(rng.uniform(0.15, 0.55)))


# ---------------------------------------------------------------- capture
def camera_frame(position, rotation, point):
    """World point -> camera optical frame (x right, y down, z forward)."""
    q = rotation.T @ (np.asarray(point) - position)
    return np.array([q[0], -q[1], -q[2]])


def project(p):
    return np.array([FX * p[0] / p[2] + CX, FY * p[1] / p[2] + CY])


def image_angle(a, b):
    dx, dy = b - a
    angle = math.degrees(math.atan2(dy, dx))
    return angle - 180 if angle > 90 else angle + 180 if angle <= -90 else angle


def camera_response(rgb, instance):
    """Imitate the D435's auto white balance and auto exposure: the real camera always
    renders the plate roughly neutral grey at a steady brightness, whatever the lamps."""
    linear = (rgb.astype(np.float32) / 255.0) ** 2.2
    background = linear[instance == 0]
    brightness = background.mean(axis=1)
    low, high = np.percentile(brightness, [40, 98])          # the plate; excludes tape and highlights
    grey = np.median(background[(brightness >= low) & (brightness <= high)], axis=0)
    wb = grey.mean() / np.maximum(grey, 1e-6)
    wb = 1 + (wb - 1) * rng.uniform(0.8, 1.0)                 # auto white balance is never perfect
    gain = ((rng.uniform(140, 180) / 255.0) ** 2.2) / max(float((grey * wb).mean()), 1e-6)
    out = np.clip(linear * wb * gain, 0, 1) ** (1 / 2.2)
    return (out * 255 + 0.5).astype(np.uint8), wb, gain


def capture(scene_dir, frame, layout, colours, cam_pos, cam_rot, active_lamps):
    # The path tracer accumulates across frames: after bricks move or change colour, the
    # first render still blends in the previous scene (ghost bricks, wrong colours).
    # Render once to flush it, then capture.
    rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)
    rgb = np.asarray(annotators['rgb'].get_data())[..., :3]
    depth = np.asarray(annotators['distance_to_image_plane'].get_data())
    seg = annotators['instance_id_segmentation'].get_data()
    seg_ids = np.asarray(seg['data'])
    id_to_path = {int(k): v for k, v in seg['info']['idToLabels'].items()}
    instance = np.zeros(seg_ids.shape, np.uint16)
    for seg_id, path in id_to_path.items():
        if '/World/Bricks/brick_' in path:
            index = int(path.split('/World/Bricks/brick_')[1][:2])
            instance[seg_ids == seg_id] = index + 1
    rgb, wb_gains, exposure_gain = camera_response(rgb, instance)
    depth_mm = np.where(np.isfinite(depth), np.clip(np.rint(depth * 1000), 0, 65535), 0).astype(np.uint16)
    for sub, image in (('rgb', rgb), ('depth', depth_mm), ('instance', instance)):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(scene_dir / sub / f'{frame:06d}.png')

    top_layer = max(p['layer'] for p in layout)
    labels = []
    for index, (p, colour) in enumerate(zip(layout, colours)):
        z_top = (p['layer'] + 1) * BRICK_H
        yaw = math.radians(p['yaw'])
        axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        centre = np.array([p['xy'][0], p['xy'][1], z_top])
        c_cam = camera_frame(cam_pos, cam_rot, centre)
        ends = [project(camera_frame(cam_pos, cam_rot, centre + s * axis * BRICK_L / 2)) for s in (-1, 1)]
        labels.append({'id': index + 1, 'colour': colour, 'layer': p['layer'],
                       'top_layer': p['layer'] == top_layer,
                       'centre_world_m': np.round(centre, 6).tolist(), 'yaw_world_deg': round(p['yaw'] % 180, 3),
                       'centre_cam_m': np.round(c_cam, 6).tolist(),
                       'centre_px': np.round(project(c_cam), 2).tolist(),
                       'angle_img_deg': round(image_angle(*ends), 3),
                       'visible_px': int((instance == index + 1).sum())})
    (scene_dir / 'labels').mkdir(exist_ok=True)
    (scene_dir / 'labels' / f'{frame:06d}.json').write_text(json.dumps({
        'camera_position_world_m': cam_pos.round(6).tolist(),
        'camera_rotation_world': cam_rot.round(6).tolist(), 'lamps': active_lamps,
        'camera_response': {'white_balance_gains': np.round(wb_gains, 4).tolist(),
                            'exposure_gain': round(float(exposure_gain), 4)},
        'top_layer': top_layer, 'bricks': labels}, indent=1))


def write_camera_info(scene_dir):
    camera_info = {key: info[key] for key in ('color_intrinsics', 'aligned_depth_intrinsics') if key in info}
    camera_info.update({'device': {'name': 'Isaac Sim D435 (simulated)'}, 'depth_scale_m': 0.001,
                        'depth_aligned_to_color': True, 'depth_invalid_value': 0, 'simulated': True,
                        'camera_coordinate_convention': 'x right, y down, z forward; metres'})
    (scene_dir / 'camera_info.json').write_text(json.dumps(camera_info, indent=2))


args.out.mkdir(parents=True, exist_ok=True)
start = time.time()
for scene in range(args.scenes):
    layout = stack_layout() if rng.random() < 0.5 else loose_layout()
    colours = [str(rng.choice(list(COLOURS))) for _ in layout]
    for index, root in enumerate(bricks):
        imageable = UsdGeom.Imageable(root)
        if index < len(layout):
            p = layout[index]
            xf = UsdGeom.Xformable(root)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(float(p['xy'][0]), float(p['xy'][1]), p['layer'] * BRICK_H))
            xf.AddRotateZOp().Set(float(p['yaw']))
            for binding in brick_parts[index]:
                binding.Bind(brick_mats[colours[index]][0])
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    centroid = np.mean([p['xy'] for p in layout], axis=0)
    scene_dir = args.out / f'scene_{scene:05d}'
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_camera_info(scene_dir)
    for view in range(args.views):
        height = rng.uniform(0.30, 0.45) + (max(p['layer'] for p in layout) + 1) * BRICK_H
        cam_pos = np.array([centroid[0] + rng.uniform(-0.04, 0.04), centroid[1] + rng.uniform(-0.04, 0.04), height])
        target = np.array([centroid[0] + rng.uniform(-0.03, 0.03), centroid[1] + rng.uniform(-0.03, 0.03), 0.0])
        up_yaw = math.radians(rng.normal(0, 8))
        cam_rot = look_at(cam_pos, target, np.array([math.cos(up_yaw), math.sin(up_yaw), 0.0]))
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = cam_rot, cam_pos
        cam_op.Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))  # USD matrices are row-vector (transposed)
        active = set_lights()
        set_brick_materials()
        capture(scene_dir, view, layout, colours, cam_pos, cam_rot, active)
    print(f'[sim] scene {scene + 1}/{args.scenes}: {len(layout)} bricks, '
          f'{(time.time() - start) / (scene + 1):.1f}s per scene', flush=True)

rep.orchestrator.wait_until_complete()
print(f'[sim] wrote {args.scenes} scenes x {args.views} views to {args.out}', flush=True)
app.close()
