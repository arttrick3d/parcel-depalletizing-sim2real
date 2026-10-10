# Parcel Depalletizing Sim2Real

> **Status: Phase 0 complete (2026-10)** · 6-month portfolio project, in progress

Train a box top-face segmentation model for a robotic parcel-depalletizing cell **using only synthetic data** from an Isaac Sim digital twin, measure its accuracy on real photos, and quantify how much **NVIDIA Cosmos Transfer** augmentation closes the sim-to-real gap. The perception model then drives a UR20 + suction gripper through ROS 2 / MoveIt 2 to unload pallets in simulation.

## Problem (virtual client brief)

| | |
|---|---|
| Client | (Fictional) parcel logistics center, Icheon, Korea |
| Task | Robot arm picks mixed boxes (size, color, packaging) one by one from the top of a pallet onto a conveyor |
| Constraint | Live operations make large-scale real data collection and labeling impractical; box types change daily |
| Approach | Generate labeled synthetic data from a digital twin, train perception on it, verify on real photos |

## Pipeline

~~~
Isaac Sim digital twin --> Replicator SDG --> YOLO segmentation (box top faces)
        |                       |                        |
        |                       +--> Cosmos Transfer ----+  (augmented vs. synthetic-only)
        |                                                v
        +------------ ROS 2 / MoveIt 2 <---- pick point from mask + depth --> unload
~~~

## Results (target)

| Metric | How it is measured | Result |
|---|---|---|
| Real-photo mAP (synthetic only) | 100 hand-labeled photos of 10-20 real parcel boxes | TBD (Phase 4) |
| Cosmos augmentation gain | Synthetic-only vs. synthetic + Cosmos, same real test set | TBD (Phase 5) |
| Unloading success rate / cycle time | 100 unseen pallets in simulation, failures categorized | TBD (Phase 6) |

## Findings so far

**Cosmos Transfer 2.5 feasibility test** (Phase 0, warehouse sample clip, 1280x720, 93 frames)

- Runs on H100 (~57 GB VRAM peak): 80 GB GPUs are enough, 48 GB are not; ~13-15 min and ~$0.7-0.8 per clip
- Output keeps input resolution and frame count, so simulation labels can be reused frame-by-frame
- Contour alignment check (`tools/overlay_check.py`): at 1 px tolerance, 60% of input edges are preserved vs. a 16% chance level, so labels stay aligned within 1-3 px
- `edge + seg` control beats `edge` only on realism and object identity at only 1.12x the per-step cost, chosen for Phase 5
- Pitfall: on RunPod's PyTorch template, `HF_HUB_ENABLE_HF_TRANSFER=1` silently breaks a config download and surfaces as an unrelated `size mismatch` error

**Cell spec** ([docs/cell_spec.md](docs/cell_spec.md)): reach analysis ruled out UR10 (1300 mm) for full T11 pallet coverage, so UR20 (1750 mm) on a 0.7 m pedestal.

## Stack

| Area | Version |
|---|---|
| Simulation | Isaac Sim 6.0.1 (pip), Replicator, OpenUSD |
| Augmentation | NVIDIA Cosmos Transfer 2.5 (cloud GPU) |
| Perception | PyTorch, Ultralytics YOLO (segmentation) |
| Robotics | ROS 2 Jazzy, MoveIt 2, UR20 + Isaac Sim Surface Gripper |
| Local machine | Ubuntu 24.04, RTX 5070 Ti 16 GB |

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Environment, cell spec, Cosmos feasibility | Done |
| 2 | Depalletizing cell digital twin | Next |
| 3 | SDG pipeline (randomization, auto labels) | |
| 4 | YOLO training, real-photo evaluation | |
| 5 | Cosmos augmentation comparison | |
| 6 | ROS 2 / MoveIt 2 unloading, success-rate report | |
| 7 | Packaging, demo video | |

## Repository layout

~~~
docs/    specs and evaluation notes
tools/   standalone utilities (e.g. label alignment check)
sim/     Isaac Sim scene and SDG scripts (Phase 2+)
~~~

NVIDIA assets are referenced by path and not redistributed. Generated datasets and videos are not committed.

---

## 한국어 요약

택배 물류센터 혼합 상자 하역 셀을 Isaac Sim 디지털트윈으로 만들고, 합성 데이터만으로 상자 윗면 인식 모델을 학습해 실사 사진에서 정확도를 측정합니다. Cosmos Transfer 증강 전후 성능을 비교하고, ROS 2 / MoveIt 2로 UR20 로봇팔 하역까지 연결합니다. 6개월 계획 중 Phase 0(환경 세팅, 셀 사양, Cosmos 실행 검증)을 마쳤습니다.

## License

Code in this repository: MIT. Third-party assets and models follow their own licenses.
