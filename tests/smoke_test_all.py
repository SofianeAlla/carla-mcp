"""End-to-end smoke test that exercises every tool except the ones that
need external repos (run_openscenario) or that are intentionally heavy
(scenario_sweep, failure_snapshot — covered by separate quick checks).

Run with the CARLA simulator listening on localhost:2000 and CARLA_MCP_TRANSPORT
left at default (or unset).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from carla_mcp import server as S


PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def step(name: str):
    def deco(fn):
        def wrapper():
            try:
                t0 = time.time()
                out = fn()
                dt = time.time() - t0
                msg = ""
                if isinstance(out, dict):
                    keys = ", ".join(list(out)[:6])
                    msg = f"keys=[{keys}]"
                elif hasattr(out, "data") and hasattr(out, "format"):
                    msg = f"image {len(out.data)} bytes"
                elif isinstance(out, list):
                    msg = f"list len={len(out)}"
                else:
                    msg = str(out)[:80]
                results.append((name, PASS, f"{dt:.2f}s  {msg}"))
            except Exception as e:
                tb = traceback.format_exc().splitlines()[-1]
                results.append((name, FAIL, f"{type(e).__name__}: {e}  | {tb}"))
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# Shared state across steps
state: dict = {}


@step("01 world_status")
def t01():
    ws = S.world_status()
    state["map"] = ws["map"]
    return ws


@step("02 load_map Town03")
def t02():
    if state.get("map") != "Town03":
        return S.load_map("Town03")
    return {"already_on": "Town03"}


@step("03 set_weather ClearNoon")
def t03():
    return S.set_weather(preset="ClearNoon")


@step("04 spawn_vehicle ego (auto-follow)")
def t04():
    res = S.spawn_vehicle(model="vehicle.tesla.model3", autopilot=False, follow_with_spectator=True)
    state["ego_id"] = res["actor_id"]
    return res


@step("05 set_spectator chase ego @ 12m/5m")
def t05():
    return S.set_spectator(actor_id=state["ego_id"], distance=12.0, height=5.0)


@step("06 spawn_traffic n=10")
def t06():
    return S.spawn_traffic(n_vehicles=10, n_pedestrians=0)


@step("07 spawn_pedestrian n=3")
def t07():
    return S.spawn_pedestrian(n=3, ai=True)


@step("08 wait 1s")
def t08():
    return S.wait(seconds=1.0)


@step("09 list_traffic_lights")
def t09():
    out = S.list_traffic_lights()
    state["tl"] = out[:1]
    return out[:3]


@step("10 set_traffic_light first->red")
def t10():
    if not state.get("tl"):
        return {"skipped": "no traffic lights"}
    return S.set_traffic_light(actor_id=state["tl"][0]["id"], state="red", freeze=True)


@step("11 set_actor_behavior ego ignore_lights=100, +20% speed")
def t11():
    return S.set_actor_behavior(
        actor_id=state["ego_id"], autopilot=True,
        ignore_traffic_lights_pct=100.0, speed_difference_pct=-20.0,
        auto_lane_change=True,
    )


@step("12 capture_sensor rgb")
def t12():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="rgb", width=480, height=270)


@step("13 capture_sensor depth")
def t13():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="depth", width=480, height=270)


@step("14 capture_sensor semantic")
def t14():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="semantic", width=480, height=270)


@step("15 capture_sensor instance_segmentation")
def t15():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="instance_segmentation", width=480, height=270)


@step("16 capture_sensor optical_flow")
def t16():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="optical_flow", width=480, height=270)


@step("17 capture_sensor dvs")
def t17():
    return S.capture_sensor(actor_id=state["ego_id"], sensor="dvs", width=480, height=270)


@step("18 capture_lidar BEV")
def t18():
    return S.capture_lidar(actor_id=state["ego_id"], channels=32, range_m=50.0,
                           points_per_second=200_000)


@step("19 capture_semantic_lidar BEV")
def t19():
    return S.capture_semantic_lidar(actor_id=state["ego_id"], channels=32, range_m=50.0,
                                    points_per_second=200_000)


@step("20 render_lidar_3d iso semantic")
def t20():
    return S.render_lidar_3d(actor_id=state["ego_id"], view="iso",
                             channels=32, range_m=50.0, points_per_second=200_000, semantic=True)


@step("21 point_cloud_clusters DBSCAN")
def t21():
    out = S.point_cloud_clusters(actor_id=state["ego_id"], eps=0.7, min_samples=5,
                                 channels=32, range_m=50.0, points_per_second=200_000)
    state["n_clusters"] = out.get("clusters", 0)
    return {k: v for k, v in out.items() if k != "image"}


@step("22 extract_3d_bboxes json")
def t22():
    return S.extract_3d_bboxes(observer_actor_id=state["ego_id"], format="json", max_distance=80.0)


@step("23 extract_3d_bboxes kitti")
def t23():
    return S.extract_3d_bboxes(observer_actor_id=state["ego_id"], format="kitti", max_distance=80.0)


@step("24 extract_3d_bboxes nuscenes")
def t24():
    return S.extract_3d_bboxes(observer_actor_id=state["ego_id"], format="nuscenes", max_distance=80.0)


@step("25 render_bev_segmentation")
def t25():
    return S.render_bev_segmentation(actor_id=state["ego_id"], range_m=50.0)


@step("26 compare_seg_with_truth")
def t26():
    return S.compare_seg_with_truth(actor_id=state["ego_id"], width=480, height=270)


@step("27 render_topdown")
def t27():
    return S.render_topdown(focus_actor_id=state["ego_id"], radius=80.0)


@step("28 render_trajectory 4s @ 4Hz")
def t28():
    return S.render_trajectory(actor_id=state["ego_id"], duration_s=4.0, sample_hz=4.0, range_m=80.0)


@step("29 attach_sensor_rig minimal")
def t29():
    out = S.attach_sensor_rig(actor_id=state["ego_id"], preset="minimal")
    state["rig_min"] = out["rig_id"]
    return out


@step("30 export_calibration json")
def t30():
    return S.export_calibration(rig_id=state["rig_min"], format="json")


@step("31 render_sensor_montage")
def t31():
    return S.render_sensor_montage(rig_id=state["rig_min"])


@step("32 capture_synchronized 3 frames")
def t32():
    return S.capture_synchronized(rig_id=state["rig_min"], n_frames=3, delta_seconds=0.1)


@step("33 auto_label kitti")
def t33():
    return S.auto_label(observer_actor_id=state["ego_id"], format="kitti")


@step("34 export_dataset 2 frames kitti")
def t34():
    return S.export_dataset(rig_id=state["rig_min"], n_frames=2, delta_seconds=0.1, label_format="kitti")


@step("35 export_point_cloud ply")
def t35():
    return S.export_point_cloud(actor_id=state["ego_id"], format="ply",
                                channels=32, range_m=40.0, points_per_second=100_000)


@step("36 export_point_cloud npy")
def t36():
    return S.export_point_cloud(actor_id=state["ego_id"], format="npy",
                                channels=32, range_m=40.0, points_per_second=100_000)


@step("37 export_point_cloud bin")
def t37():
    return S.export_point_cloud(actor_id=state["ego_id"], format="bin",
                                channels=32, range_m=40.0, points_per_second=100_000)


@step("38 compute_lidar_stats")
def t38():
    return S.compute_lidar_stats(actor_id=state["ego_id"], channels=32, range_m=60.0,
                                 points_per_second=200_000)


@step("39 voxelize 0.5m")
def t39():
    out = S.voxelize(actor_id=state["ego_id"], voxel_size=0.5, range_m=40.0,
                     channels=32, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "occupied_voxels_truncated"}


@step("40 ground_plane_segment RANSAC")
def t40():
    out = S.ground_plane_segment(actor_id=state["ego_id"], distance_threshold=0.2,
                                 iterations=100, channels=32, range_m=50.0,
                                 points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "image"}


@step("41 lidar_to_camera_overlay")
def t41():
    return S.lidar_to_camera_overlay(actor_id=state["ego_id"], width=640, height=360,
                                     channels=32, range_m=60.0, points_per_second=200_000)


@step("42 spawn_adversarial cut_in vs ego")
def t42():
    out = S.spawn_adversarial(behavior="cut_in", target_actor_id=state["ego_id"], distance_m=18.0)
    state["adv_id"] = out["actor_id"]
    return out


@step("43 iou_3d ego vs adversarial")
def t43():
    return S.iou_3d(actor_a_id=state["ego_id"], actor_b_id=state["adv_id"])


@step("44 start_recorder + stop_recorder")
def t44():
    S.start_recorder(path="smoketest.log", additional_data=True)
    time.sleep(0.5)
    return S.stop_recorder()


@step("45 compile_scenario small spec")
def t45():
    spec = {
        "weather": {"preset": "WetCloudyNoon"},
        "spectator": "ego",
        "wait_s": 0.5,
    }
    return S.compile_scenario(spec)


# --- v2.1: 3D perception evaluation suite ---

@step("46 actor_visibility (all)")
def t46():
    out = S.actor_visibility(observer_actor_id=state["ego_id"],
                             channels=32, range_m=60.0, points_per_second=200_000)
    if out["actors"]:
        state["nearby_actor"] = out["actors"][0]["actor_id"]
    return {"n_actors_visible": out["n_actors_visible"]}


@step("47 actor_visibility (target=ego)")
def t47():
    return S.actor_visibility(observer_actor_id=state["ego_id"],
                              target_actor_id=state["ego_id"],
                              channels=32, range_m=60.0, points_per_second=200_000)


@step("48 extract_actor_points")
def t48():
    target = state.get("nearby_actor", state["ego_id"])
    out = S.extract_actor_points(observer_actor_id=state["ego_id"], target_actor_id=target,
                                 channels=32, range_m=60.0, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "image"}


@step("49 class_conditional_bev Car+Pedestrian")
def t49():
    return S.class_conditional_bev(observer_actor_id=state["ego_id"],
                                   classes=["Car", "Pedestrian"],
                                   channels=32, range_m=60.0, points_per_second=200_000)


@step("50 evaluate_clustering")
def t50():
    out = S.evaluate_clustering(observer_actor_id=state["ego_id"],
                                eps=0.7, min_samples=5, iou_threshold=0.2,
                                channels=32, range_m=60.0, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "image"}


@step("51 lidar_to_camera_segmentation")
def t51():
    return S.lidar_to_camera_segmentation(observer_actor_id=state["ego_id"],
                                          width=640, height=360,
                                          channels=32, range_m=60.0, points_per_second=200_000)


@step("52 check_sensor_consistency")
def t52():
    out = S.check_sensor_consistency(observer_actor_id=state["ego_id"],
                                     width=640, height=360,
                                     channels=32, range_m=60.0, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "image"}


@step("53 semantic_voxelize all classes")
def t53():
    out = S.semantic_voxelize(observer_actor_id=state["ego_id"], voxel_size=0.5,
                              range_m=40.0, channels=32, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "occupied_voxels_truncated"}


@step("54 semantic_voxelize Car only")
def t54():
    out = S.semantic_voxelize(observer_actor_id=state["ego_id"], voxel_size=0.5,
                              range_m=40.0, classes=["Car"],
                              channels=32, points_per_second=200_000)
    return {k: v for k, v in out.items() if k != "occupied_voxels_truncated"}


@step("55 reset_world")
def t55():
    return S.reset_world(also_clear_tracked=True)


# Ordered list
ALL = [
    t01, t02, t03, t04, t05, t06, t07, t08, t09, t10,
    t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
    t21, t22, t23, t24, t25, t26, t27, t28, t29, t30,
    t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
    t41, t42, t43, t44, t45,
    t46, t47, t48, t49, t50, t51, t52, t53, t54, t55,
]

if __name__ == "__main__":
    for fn in ALL:
        fn()
        # Tight progress print
        n, ok, msg = results[-1]
        print(f"{n:<48} [{ok}] {msg}")

    n_pass = sum(1 for _, ok, _ in results if ok == PASS)
    print(f"\n{n_pass}/{len(results)} passed")
    if n_pass < len(results):
        print("\nFailures:")
        for n, ok, msg in results:
            if ok == FAIL:
                print(f"  - {n}: {msg}")
        sys.exit(1)
