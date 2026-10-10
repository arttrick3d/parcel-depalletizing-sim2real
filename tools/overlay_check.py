#!/usr/bin/env python3
"""
Day 4-C 윤곽 정합 확인 (overlay_check.py)

Cosmos 변환 영상의 윤곽이 시뮬레이션 입력과 같은 자리에 있는지 숫자로 확인한다.
  - 기준 윤곽: Isaac Sim CosmosWriter가 만든 edges 영상 (Cosmos가 받은 edge 컨트롤)
  - 비교 윤곽: 변환 영상 프레임에 Canny를 적용해 뽑은 윤곽
  - edge 근처 비율: 기준 윤곽 픽셀 중, 비교 윤곽이 tol px 이내에 있는 비율

함께 계산하는 값
  - 기준선: 입력 rgb에 같은 Canny를 적용했을 때의 비율 (변환이 완벽할 때 기대할 수 있는 상한)
  - 우연 수준: 윤곽이 무작위로 흩어져 있어도 나오는 비율 (질감이 많으면 높아짐, 하한)
  - 새 윤곽 비율: 변환 영상 윤곽 중 입력 윤곽에서 tol px보다 먼 것의 비율 (새로 생긴 질감·물체)

실행 (위치: ~/isaac_scratch)
  python3 day4c/overlay_check.py

결과 (기본 위치: day4c/overlay/)
  near_ratio.csv              프레임별 수치
  <설정>_fNNN.png             윤곽 겹침 이미지
                              초록 = 입력 윤곽이 유지됨
                              빨강 = 입력 윤곽이 사라지거나 어긋남
                              파랑 = 변환에서 새로 생긴 윤곽
  seg_<설정>_fNNN.png         시뮬레이션 세그멘테이션을 변환 프레임에 반투명으로 겹친 이미지
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np


def read_frames(path, wanted):
    """영상에서 wanted에 든 프레임 번호만 골라 {번호: 이미지}로 돌려준다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[오류] 영상을 열 수 없음: {path}")
    frames, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    missing = sorted(set(wanted) - frames.keys())
    if missing:
        raise SystemExit(f"[오류] {path}: 프레임 {missing} 없음 (총 {i}프레임)")
    return frames, i


def edges_from_control(bgr):
    """edge 컨트롤 영상(검은 바탕에 흰 선)을 이진 윤곽으로."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray > 127


def edges_canny(bgr):
    """일반 영상에서 윤곽 추출. 밝기 중앙값 기준으로 임계값을 자동 설정."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    lo = int(max(0, 0.67 * med))
    hi = int(min(255, 1.33 * med))
    return cv2.Canny(gray, lo, max(hi, lo + 1)) > 0


def dist_to(edges):
    """각 픽셀에서 가장 가까운 윤곽 픽셀까지의 거리(px)."""
    src = np.where(edges, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def chance_ratio(density, tol):
    """윤곽이 무작위로 흩어져 있다고 가정했을 때 tol px 안에 하나라도 걸릴 확률."""
    k = sum(1 for dy in range(-tol, tol + 1) for dx in range(-tol, tol + 1)
            if dx * dx + dy * dy <= tol * tol)
    return 1.0 - (1.0 - density) ** k


def compare(ref, out, tol):
    d_out = dist_to(out)
    d_ref = dist_to(ref)
    near = float((d_out[ref] <= tol).mean()) if ref.any() else math.nan
    new = float((d_ref[out] > tol).mean()) if out.any() else math.nan
    density = float(out.mean())
    return near, new, density, chance_ratio(density, tol), d_out, d_ref


def put_title(img, text):
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_overlay(frame, ref, out, d_out, d_ref, tol, title):
    img = (frame.astype(np.float32) * 0.45).astype(np.uint8)
    img[out & (d_ref > tol)] = (255, 120, 0)   # 파랑: 새로 생긴 윤곽
    img[ref & (d_out <= tol)] = (0, 255, 0)    # 초록: 유지
    img[ref & (d_out > tol)] = (0, 0, 255)     # 빨강: 사라짐·어긋남
    put_title(img, title)
    return img


def draw_seg(frame, seg, title):
    mask = seg.max(axis=2) > 20
    blend = cv2.addWeighted(frame, 0.5, seg, 0.5, 0)
    img = frame.copy()
    img[mask] = blend[mask]
    put_title(img, title)
    return img


def main():
    ap = argparse.ArgumentParser(description="Cosmos 변환 영상 윤곽 정합 확인")
    ap.add_argument("--input-rgb", default="cosmos_input/rgb_16fps.mp4")
    ap.add_argument("--input-edge", default="cosmos_input/edges_16fps.mp4")
    ap.add_argument("--input-seg", default="cosmos_input/segmentation_16fps.mp4")
    ap.add_argument("--outputs", nargs="+", default=[
        "edge_base=cosmos_output2/edge_base/wh_edge_base.mp4",
        "edge_seg=cosmos_output2/edge_seg/wh_edge_seg.mp4",
    ], help="이름=경로 형식, 여러 개 가능")
    ap.add_argument("--step", type=int, default=10, help="몇 프레임마다 검사할지")
    ap.add_argument("--tol", type=int, default=3, help="허용 거리(px)")
    ap.add_argument("--out-dir", default="day4c/overlay")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input_edge)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f"[오류] 프레임 수를 읽을 수 없음: {args.input_edge}")

    sample = list(range(0, n, args.step))
    showcase = sorted({0, n // 2, n - 1})
    wanted = set(sample) | set(showcase)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"입력 {n}프레임, 검사 프레임 {sample}, 허용 거리 {args.tol}px")
    ref_frames, _ = read_frames(args.input_edge, wanted)
    seg_frames, _ = read_frames(args.input_seg, set(showcase))

    targets = [("baseline", read_frames(args.input_rgb, wanted)[0])]
    for spec in args.outputs:
        name, path = spec.split("=", 1)
        frames, count = read_frames(path, wanted)
        if count != n:
            print(f"[주의] {name}: 프레임 수 {count} (입력 {n}과 다름)")
        targets.append((name, frames))

    rows = []
    for name, frames in targets:
        for i in sample:
            ref = edges_from_control(ref_frames[i])
            out = edges_canny(frames[i])
            near, new, dens, chance, d_out, d_ref = compare(ref, out, args.tol)
            rows.append({"config": name, "frame": i,
                         "near_ratio": round(near, 4), "new_edge_ratio": round(new, 4),
                         "edge_density": round(dens, 4), "chance": round(chance, 4)})
            if name != "baseline":
                title = f"{name}  f{i:03d}  near {near:.2f}"
                cv2.imwrite(os.path.join(args.out_dir, f"{name}_f{i:03d}.png"),
                            draw_overlay(frames[i], ref, out, d_out, d_ref, args.tol, title))
        if name != "baseline":
            for i in showcase:
                cv2.imwrite(os.path.join(args.out_dir, f"seg_{name}_f{i:03d}.png"),
                            draw_seg(frames[i], seg_frames[i], f"{name} + sim seg  f{i:03d}"))

    csv_path = os.path.join(args.out_dir, "near_ratio.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    names = [name for name, _ in targets]
    table = {(r["config"], r["frame"]): r for r in rows}

    print("\n[프레임별 edge 근처 비율]")
    print("frame " + "".join(f"{nm:>12}" for nm in names))
    for i in sample:
        print(f"{i:>5} " + "".join(f"{table[(nm, i)]['near_ratio']:>12.3f}" for nm in names))

    print("\n[요약]")
    print(f"{'config':<12}{'near평균':>10}{'near최저':>10}{'최저frame':>10}"
          f"{'새윤곽평균':>10}{'우연수준':>10}")
    for nm in names:
        rs = [table[(nm, i)] for i in sample]
        nears = [r["near_ratio"] for r in rs]
        worst = min(rs, key=lambda r: r["near_ratio"])
        print(f"{nm:<12}{np.mean(nears):>10.3f}{min(nears):>10.3f}{worst['frame']:>10}"
              f"{np.mean([r['new_edge_ratio'] for r in rs]):>10.3f}"
              f"{np.mean([r['chance'] for r in rs]):>10.3f}")

    print(f"\n저장: {csv_path}, 겹침 이미지 {args.out_dir}/")


if __name__ == "__main__":
    main()
