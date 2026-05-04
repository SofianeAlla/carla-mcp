# Roadmap

v1 (current) is the smallest surface that makes Claude useful inside CARLA. v2 below is "max functionality" — the full set of capabilities an AD ML engineer would reach for. Items are intentionally grouped so each block can ship independently as a sub-package, keeping the core install lean.

## v0.1 — shipped

- [x] `world_status`
- [x] `load_map`
- [x] `set_weather`
- [x] `spawn_vehicle`
- [x] `spawn_traffic`
- [x] `capture_sensor` (RGB / depth / semantic)
- [x] `wait`
- [x] `reset_world`

## v0.2 — shipped

- [x] HTTP / streamable-http transport for claude.ai Custom Connectors
- [x] Bearer-token middleware (`CARLA_MCP_TOKEN`)
- [x] Cloudflared tunnel walkthrough in README

## v0.3 — shipped

- [x] `set_spectator(actor_id?, x?,y?,z?)` — point the sim's main viewport at whatever Claude is working with
- [x] `spawn_vehicle` auto-follows with the spectator (toggle via `follow_with_spectator`)
- [x] Cross-process actor tracking: `_spawned` set persists to `%TEMP%/carla-mcp-tracked.json`, so `reset_world()` cleans up actors spawned by *any* prior session
- [x] `load_map` returns a warning that it destroys all actors (incl. external clients like `manual_control.py`)

## v0.4 — shipped

- [x] **Stdout isolation fix**: CARLA's native C++ logs (`INFO: Found the required file in cache!  …`) were corrupting the MCP JSON-RPC stream in stdio mode and producing `Unexpected token 'I'` errors on the client. We now redirect fd 1 to a log file (`%TEMP%/carla-mcp.log`, override with `CARLA_MCP_LOG`) on import, while keeping `sys.stdout` bound to the original fd for protocol traffic.

## v2 — scenarios & replay

- [ ] `run_openscenario(file)` — execute OpenSCENARIO 1.x via ScenarioRunner *(needs `scenario_runner` repo)*
- [ ] `start_recorder(path)` / `replay(path, start, dur, actor)` — built-in `.log` recording
- [ ] `scenario_sweep(template, params)` — vary weather/density/seed across N runs, return aggregated metrics
- [ ] `compile_scenario(natural_language)` — Claude writes a structured scenario spec, server materializes it

## v2 — actors & traffic (advanced)

- [ ] `spawn_pedestrian(n, ai)` — walkers with AI controllers (basic version is in v1's `spawn_traffic`)
- [ ] `spawn_adversarial(behavior)` — cut-in / sudden-brake / jaywalker prefabs
- [ ] `set_traffic_light(id, state)` — manual signal control
- [ ] `set_actor_behavior(id, params)` — Traffic Manager fine-tuning (lane changes, speed offsets)

## v2 — sensor rigs at scale

- [ ] `attach_sensor_rig(actor, preset)` — front/rear/L/R cams + 360 lidar + IMU/GNSS/radar
- [ ] `capture_synchronized(rig_id, hz, n_frames)` — paired frames in synchronous mode
- [ ] `export_calibration(rig_id, format)` — intrinsics/extrinsics → JSON / nuScenes / KITTI
- [ ] Full sensor catalogue: lidar, semantic_lidar, radar, optical_flow, DVS, IMU, GNSS

## v2 — dataset generation

- [ ] `auto_label(frame, format)` — 3D bboxes of all actors → KITTI / nuScenes / COCO
- [ ] `dataset_sweep(scenarios, output_dir)` — packaged dataset with manifest
- [ ] `pack_hdf5(run_id)` / `pack_parquet(run_id)`

## v2 — planner / policy evaluation

- [ ] `register_planner(python_module)` — user code controls ego
- [ ] `run_benchmark(suite)` — NHTSA precrash typology, EuroNCAP, NoCrash, LeaderBoard
- [ ] `compute_metrics(run_id)` — success, TTC, jerk, lane invasion, route completion, infraction score
- [ ] `ab_compare(planner_a, planner_b, seeds)` — head-to-head on identical seeds
- [ ] `mine_edge_cases(planner, threshold)` — randomized seed sweep, return failure cases
- [ ] `counterfactual(run_id, change)` — rerun with one variable flipped

## v2 — visualization (high-leverage for Claude)

- [ ] `render_topdown(time)` — matplotlib bird's-eye with all actors → PNG
- [ ] `render_trajectory(actor_id)` — overlay path on map
- [ ] `render_sensor_montage(rig_id)` — RGB+semantic+depth grid
- [ ] `failure_snapshot(run_id)` — auto-capture frames around collision / lane-invasion events
- [ ] `play_clip(run_id, t0, t1)` — render MP4 *(needs ffmpeg)*

## v2 — co-simulation & bridges *(separate sub-package: `carla-mcp-cosim`)*

- [ ] `start_sumo_cosim(net_file)` — city-scale traffic *(needs SUMO)*
- [ ] `start_ros2_bridge()` — *(needs ROS2 + carla-ros-bridge)*
- [ ] `start_autoware_bridge()` — test real stacks

## v2 — training-loop integration *(separate sub-package: `carla-mcp-train`)*

- [ ] `gym_env(config)` — Gymnasium-API env for RL
- [ ] `collect_rollouts(policy, n)` — imitation-learning data collection
- [ ] `parallel_servers(n)` — manage N CARLA processes, distribute work
- [ ] `log_to_wandb(run_id)` / `log_to_mlflow(run_id)`

## v2 — map authoring *(advanced)*

- [ ] `generate_opendrive(spec)` — programmatic road network creation
- [ ] `import_osm(extract)` — convert OpenStreetMap regions to drivable maps

## v2.1 — shipped (3D perception eval)

Seven new tools that turn carla-mcp into a real perception-evaluation harness, using fields the semantic-lidar sensor already provides (per-point `ObjIdx` + `ObjTag`):

- `extract_actor_points(observer, target_actor)` — slice points by actor instance id; per-object centroid/extent/dominant class
- `actor_visibility(observer, target?)` — per-actor lidar hit counts + `high`/`medium`/`low`/`occluded` classification
- `class_conditional_bev(observer, classes)` — BEV filtered to specific semantic classes
- `evaluate_clustering(observer, eps, min_samples, iou_threshold)` — DBSCAN-vs-ground-truth precision / recall / mean IoU + BEV PNG (matched green, FP red, missed orange)
- `lidar_to_camera_segmentation(observer)` — project semantic lidar onto RGB, color each point by class
- `check_sensor_consistency(observer)` — semantic camera vs semantic lidar pixel agreement %
- `semantic_voxelize(observer, voxel_size, classes?)` — voxelize with dominant class per voxel (OccNet ground truth)

## v0.5–2.0 — shipped

A consolidated v2.0 release; **42 tools** total. Categories:

- **Sensors**: full coverage of CARLA's catalogue (RGB, depth, semantic, instance, optical_flow, DVS, lidar, semantic lidar)
- **3D analysis**: BEV / semantic / 3D scatter renders, DBSCAN clustering, RANSAC ground plane, voxelization, lidar→camera projection, 3D-IoU, point-cloud export to PLY/PCD/NPY/BIN
- **Datasets**: sensor rigs (minimal/perception/full), synchronized multi-sensor capture, calibration export (JSON/KITTI), auto-labeling (KITTI/nuScenes/JSON), `export_dataset` packaged KITTI folder
- **Scenarios**: `compile_scenario` (declarative spec), `scenario_sweep` (param variation), adversarial prefabs (cut_in / sudden_brake / jaywalker), `failure_snapshot`, `run_openscenario` (subprocess wrapper)
- **Recording**: `start_recorder` / `stop_recorder` / `replay`
- **Visualization**: `render_topdown`, `render_trajectory`, `render_sensor_montage`, `render_bev_segmentation`, `compare_seg_with_truth`

## v2.5 — planner evaluation

- [ ] `register_planner_endpoint(http_url)` — planner runs in its own process; carla-mcp POSTs world state per tick
- [ ] `run_benchmark(suite)` — NoCrash, NHTSA precrash typology, EuroNCAP, LeaderBoard
- [ ] `compute_metrics(run_id)` — success, TTC, jerk, lane invasion, route completion, infraction score
- [ ] `ab_compare(planner_a_url, planner_b_url, seeds)` — head-to-head on identical seeds
- [ ] `mine_edge_cases(planner_url, threshold)` — randomized seed sweep, return failure cases
- [ ] `counterfactual(run_id, change)` — rerun with one variable flipped

## v3 — co-sim & training (separate sub-packages)

- [ ] `carla-mcp-cosim`: SUMO co-simulation, ROS2 bridge, Autoware bridge launchers
- [ ] `carla-mcp-train`: Gymnasium env wrapping the same primitives, parallel CARLA orchestration, W&B / MLflow run logging

## v3+ — auth & deployment

- [ ] Native OAuth 2.1 for claude.ai connectors (vs. relying on token + tunnel secrecy)
- [ ] Helm chart / Docker image for fleet deployment
- [ ] Multi-tenant mode (one MCP server, many CARLA instances)
- [ ] HDF5 / Parquet packaging (`pack_hdf5(run_id)` / `pack_parquet(run_id)` — needs `h5py` / `pyarrow`)
- [ ] MP4 clip export from recordings (needs `ffmpeg`)

## Non-goals

- Full ChatGPT Apps support — keep this Claude-first. A Skybridge port can come later if there's demand.
- Wrapping CARLA's source build / Unreal Editor toolchain.
- Re-implementing CARLA itself; we strictly wrap the existing Python API.
