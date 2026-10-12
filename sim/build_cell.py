#!/usr/bin/env python3
"""
Phase 2-1: 하역 셀 씬 뼈대 (build_cell.py)

기준: docs/cell_spec.md v1.0, Isaac Sim 6.0.1
좌표: 단위 m, Z-up, 팔레트 중심 바닥 = 원점, +y = 로봇에서 팔레트 쪽
구성: 바닥, KS T11 팔레트(단순 박스), 조명, 상단 카메라(3.0m 수직 하향, 1280x720 pinhole)

실행 (저장소 루트에서, ~/isaacsim_env 활성화 상태)
  python sim/build_cell.py          # usda 저장 + 카메라 확인 이미지 + 팔레트 픽셀 크기 검사
  python sim/build_cell.py --gui    # 창을 띄워 씬 확인 (창을 닫으면 종료)
"""
import argparse
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--usd", default="sim/scenes/cell_base.usda")
ap.add_argument("--img", default=os.path.expanduser("~/isaac_scratch/phase2/cam_check_2-1.png"))
args = ap.parse_args()

from isaacsim.simulation_app import SimulationApp

app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720})

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

# ---- 사양서 숫자 (cell_spec.md) ----
PALLET_SIZE = (1.10, 1.10, 0.15)   # KS T11
CAM_PATH = "/World/TopCamera"
CAM_Z = 3.0                        # 바닥에서 3.0m
RES = (1280, 720)
VFOV_DEG = 43.0                    # 수직 화각 -> 수평 약 70도
H_APERTURE = 20.955                # mm (USD 기본값), 초점거리는 화각으로 역산


def add_box(stage, path, size, center, color, collider=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(cube)
    xf.SetTranslate(Gf.Vec3d(*center))
    xf.SetScale(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def build(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    # 바닥 (20m x 20m, 윗면 z = 0)
    add_box(stage, "/World/Ground", (20.0, 20.0, 0.02), (0, 0, -0.01), (0.45, 0.45, 0.45))

    # 팔레트 (단순 박스, 윗면 z = 0.15)
    px, py, pz = PALLET_SIZE
    add_box(stage, "/World/Pallet", PALLET_SIZE, (0, 0, pz / 2), (0.60, 0.42, 0.25))

    # 조명
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(1000.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(35.0, 0.0, 30.0))

    # 상단 카메라: USD 카메라는 기본이 -Z를 바라봄 -> 회전 없이 위치만 주면 수직 하향
    # 화면 위쪽 = +y (로봇 반대편, 팔레트 먼 쪽)
    cam = UsdGeom.Camera.Define(stage, CAM_PATH)
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0, 0, CAM_Z))
    v_ap = H_APERTURE * RES[1] / RES[0]
    focal = (v_ap / 2) / math.tan(math.radians(VFOV_DEG / 2))
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    hfov = 2 * math.degrees(math.atan((H_APERTURE / 2) / focal))
    print(f"[카메라] focal {focal:.2f}mm, 화각 수직 {VFOV_DEG:.1f} / 수평 {hfov:.1f} deg")


def expected_pallet_px():
    """팔레트 윗면(거리 CAM_Z - 0.15)에서 팔레트 1.1m가 몇 픽셀인지."""
    d = CAM_Z - PALLET_SIZE[2]
    visible_h = 2 * d * math.tan(math.radians(VFOV_DEG / 2))
    return PALLET_SIZE[0] / visible_h * RES[1]


def render_check(stage):
    ext = omni.kit.app.get_app().get_extension_manager()
    ext.set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep

    for _ in range(10):
        app.update()
    rp = rep.create.render_product(CAM_PATH, RES)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)

    img = None
    for _ in range(5):
        rep.orchestrator.step(rt_subframes=16)
        data = rgb.get_data()
        if data is not None and getattr(data, "size", 0) > 0:
            img = np.asarray(data)[:, :, :3]
            break
    if img is None:
        print("[오류] 카메라 이미지를 받지 못함")
        return

    os.makedirs(os.path.dirname(args.img), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(args.img)
    except ImportError:
        np.save(args.img.replace(".png", ".npy"), img)
        print("[주의] PIL 없음 -> .npy로 저장")
    print(f"[저장] 카메라 이미지: {args.img}")

    # 팔레트(갈색) 영역 = R이 B보다 확실히 큰 픽셀
    r, b = img[:, :, 0].astype(int), img[:, :, 2].astype(int)
    g = np.median(r[:50, :50] - b[:50, :50])  # 바닥(왼쪽 위 모서리) 기준
    ys, xs = np.nonzero((r - b) - g > 15)
    exp = expected_pallet_px()
    if len(xs) == 0:
        print("[검사] 팔레트 픽셀을 찾지 못함 -> 이미지를 직접 확인")
        return
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    print(f"[검사] 팔레트 픽셀 크기 {w} x {h} (예상 약 {exp:.0f} x {exp:.0f})")
    print(f"[검사] 팔레트 중심 ({cx:.0f}, {cy:.0f}) (예상 (640, 360))")
    ok = abs(w - exp) < 0.05 * exp and abs(h - exp) < 0.05 * exp \
        and abs(cx - 640) < 10 and abs(cy - 360) < 10
    print("[판정] OK" if ok else "[판정] 확인 필요 (크기 5% 또는 중심 10px 초과)")


def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    build(stage)

    os.makedirs(os.path.dirname(args.usd), exist_ok=True)
    stage.GetRootLayer().Export(args.usd)
    print(f"[저장] 씬: {args.usd}")

    if args.gui:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().camera_path = CAM_PATH
        except Exception as e:
            print(f"[주의] 뷰포트 카메라 전환 실패: {e}")
        while app.is_running():
            app.update()
    else:
        render_check(stage)


main()
app.close()
