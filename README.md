# carla-mcp

> The **3D-simulation-and-analysis** MCP connector for [CARLA](https://carla.org).
> Strongly biased toward **lidar, point clouds, semantic segmentation, and perception evaluation** — not just camera screenshots.

A FastMCP server giving Claude eyes-and-hands control of the CARLA autonomous-driving simulator, with a dedicated 3D suite that no other MCP connector ships:

- **3D point-cloud capture** — lidar BEV, semantic lidar, headless 3D scatter renders (iso / bev / rear / side views).
- **3D analysis** — DBSCAN object proposals, RANSAC ground-plane fit, voxelization (geometric **and** semantic), 3D-IoU between actors, lidar→camera projection (depth- or class-coloured), per-actor visibility audits.
- **Perception evaluation, no model needed** — `evaluate_clustering` matches DBSCAN proposals against CARLA's ground-truth `ObjIdx` and reports precision / recall / mean IoU. `check_sensor_consistency` cross-validates semantic camera vs semantic lidar pixel-by-pixel.
- **3D dataset export** — sensor rigs (front/rear/L/R cams + 64-ch semantic lidar + IMU/GNSS/radar), synchronized capture, KITTI-style packaged dataset folder, point-cloud export to PLY / PCD / NPY / KITTI BIN.

Camera tools (RGB / depth / semantic / instance / optical-flow / DVS) are still here, but they're support-cast for the 3D pipeline — not the headline.

Built for AD ML / perception engineers who want an LLM in the loop for scenario authoring, regression testing, lidar / segmentation evaluation, and edge-case mining.

## Status

**v2.2 — 52 tools shipped**, including simulator-process orchestration so Claude can launch CARLA itself. The full AD ML loop, fully agentic:

- World control, weather, spectator, traffic
- All CARLA sensor types (RGB, depth, semantic, instance, optical flow, DVS, lidar, semantic lidar)
- Sensor rigs (minimal / perception / full) + synchronized multi-sensor capture + KITTI-style dataset export
- **3D analysis suite**: BEV lidar, semantic lidar, 3D point-cloud render, DBSCAN clustering, ground-plane RANSAC, voxelization, lidar→camera projection overlay, 3D-IoU
- 3D bbox extraction → JSON / KITTI / nuScenes formats
- BEV semantic raster, semantic vs ground-truth side-by-side
- Recording + replay
- Scenario compilation, scenario sweeps, adversarial prefabs (cut-in, sudden brake, jaywalker)
- Failure-snapshot (auto frame on collision/lane-invasion)
- HTTP transport for claude.ai Custom Connectors

v2.5 / v3 roadmap in [ROADMAP.md](./ROADMAP.md).

## Tools (52)

### Simulator orchestration (process-level)
| Tool | What it does |
| --- | --- |
| `simulator_status` | Non-destructive probe: are any `CarlaUE4*` processes alive, and is port 2000 open? Returns `ready=true` when the next sim RPC will succeed. |
| `start_simulator(carla_root?, quality_level, render_off_screen, wait_ready_seconds)` | Launches CARLA detached if it isn't already running, polls port 2000 until ready. Short-circuits with `already_running=true` when the simulator is already up — safe to call unconditionally as step 0 of any chain. |
| `stop_simulator` | Terminates running `CarlaUE4*` processes (uses `taskkill` on Windows, `pkill` on Linux). |

`carla_root` resolution order: explicit arg → `CARLA_ROOT` env var → `C:\Users\<USER>\CARLA_0.9.16` (Windows) → `~/carla` (Linux).

### World & control
| Tool | What it does |
| --- | --- |
| `world_status` | Map, weather, actor counts, sim time, list of tracked actor ids. |
| `load_map(name)` | Switch town. Town01–07, Town10HD_Opt, Town11–15. **Destroys all actors.** |
| `set_weather(preset, ...)` | Apply named preset or override cloudiness/precipitation/sun/fog/wetness. |
| `set_spectator(actor_id?, x?,y?,z?, distance, height, pitch)` | Move the simulator's viewport — chase an actor or teleport. |
| `wait(seconds)` | Let the sim run between actions. |
| `reset_world()` | Destroy all actors spawned through this server (cross-session via persistent JSON). |

### Actors & traffic
| Tool | What it does |
| --- | --- |
| `spawn_vehicle(model, spawn_point_index?, autopilot?, follow_with_spectator?)` | Spawn one vehicle, optionally chase-camera follow. |
| `spawn_traffic(n_vehicles, n_pedestrians)` | Populate the world with autopilot agents. |
| `spawn_pedestrian(n, ai, max_speed)` | Walkers with optional AI controllers. |
| `spawn_adversarial(behavior, target_actor_id?, distance_m)` | Prefabs: `cut_in`, `sudden_brake`, `jaywalker`. |
| `set_actor_behavior(actor_id, autopilot?, ignore_traffic_lights_pct?, …)` | Per-actor Traffic Manager configuration. |
| `list_traffic_lights` / `set_traffic_light(id, state, freeze)` | Read + override signal states. |

### Sensors (single-shot)
| Tool | What it does |
| --- | --- |
| `capture_sensor(actor_id, sensor, width, height, fov)` | RGB / depth / semantic / instance_segmentation / optical_flow / dvs camera → PNG. |
| `capture_lidar(actor_id, channels, range_m, …)` | One lidar sweep → BEV intensity PNG. |
| `capture_semantic_lidar(actor_id, …)` | Semantic-tagged sweep → BEV PNG with CityScapes colors. |

### 3D analysis & ML data — the headline suite
| Tool | What it does |
| --- | --- |
| `render_lidar_3d(actor_id, view, semantic)` | Matplotlib 3D scatter render of a sweep (`iso`/`bev`/`rear`/`side`). |
| `point_cloud_clusters(actor_id, eps, min_samples)` | DBSCAN proposals → BEV PNG + cluster centroids/extents. |
| `extract_3d_bboxes(observer_actor_id, max_distance, format)` | Ground-truth 3D bboxes in JSON / KITTI / nuScenes formats. |
| `render_bev_segmentation(actor_id, range_m, resolution)` | Top-down semantic raster (drivable area + actors). |
| `compare_seg_with_truth(actor_id, …)` | Side-by-side: front semantic camera ⟷ BEV ground-truth. |
| `voxelize(actor_id, voxel_size, range_m, z_range)` | Lidar → sparse 3D occupancy grid. |
| `ground_plane_segment(actor_id, distance_threshold, iterations)` | RANSAC plane fit, ground vs. obstacle BEV PNG. |
| `lidar_to_camera_overlay(actor_id, width, height, fov)` | Pinhole-project lidar onto RGB, color by depth. |
| `compute_lidar_stats(actor_id, …)` | Density / range / per-ring counts / uniformity ratio. |
| `iou_3d(actor_a_id, actor_b_id)` | AABB 3D IoU between two actors. |
| `export_point_cloud(actor_id, format, output_path?)` | Write to disk: `ply` / `pcd` / `npy` / `bin` (KITTI). |

### 3D perception evaluation (v2.1) — uses semantic-lidar's `ObjIdx`
| Tool | What it does |
| --- | --- |
| `extract_actor_points(observer_actor_id, target_actor_id, …)` | Slice a sweep to one actor's points (count, centroid, extent, dominant class, BEV PNG). Foundation for instance-level analyses. |
| `actor_visibility(observer_actor_id, target_actor_id?)` | Per-actor lidar hit count + visibility class (`high`/`medium`/`low`/`occluded`). Filter for autolabel quality. |
| `class_conditional_bev(observer_actor_id, classes)` | BEV showing only specified classes (e.g. `["Car","Pedestrian"]` for dynamic-actor map). |
| `evaluate_clustering(observer_actor_id, eps, min_samples, iou_threshold)` | DBSCAN proposals matched against ground-truth `ObjIdx` → precision, recall, mean IoU, per-actor matches, BEV PNG (matched green / FP red / missed orange). **Real ML evaluation, no detector model needed.** |
| `lidar_to_camera_segmentation(observer_actor_id, …)` | Project semantic lidar into RGB camera, color each projected point by class. Sparse pixel-perfect semantic GT from sensor calibration alone. |
| `check_sensor_consistency(observer_actor_id, …)` | Capture semantic camera + project semantic lidar into it. Per-class agreement %, plus overlay PNG with disagreements highlighted red. |
| `semantic_voxelize(observer_actor_id, voxel_size, classes?)` | Voxelize but each occupied voxel keeps its dominant class. Direct ground truth for OccNet / BEVFusion. |

### Sensor rigs & datasets
| Tool | What it does |
| --- | --- |
| `attach_sensor_rig(actor_id, preset)` | Attach `minimal` / `perception` / `full` rig in one call → `rig_id`. |
| `export_calibration(rig_id, format)` | Per-sensor intrinsic + extrinsic, JSON or KITTI calib.txt. |
| `render_sensor_montage(rig_id, max_cells)` | Capture all cameras in the rig, lay out as a grid PNG. |
| `capture_synchronized(rig_id, n_frames, delta_seconds, output_dir?)` | Sync-mode capture, paired frames + manifest.json. |
| `auto_label(observer_actor_id, format, output_path?)` | One-frame ground-truth labels (KITTI / nuScenes / JSON). |
| `export_dataset(rig_id, n_frames, label_format, output_dir?)` | Full KITTI-style dataset folder (calib + image_2 + velodyne + label_2). |

### Recording, scenarios, viz
| Tool | What it does |
| --- | --- |
| `start_recorder(path, additional_data)` / `stop_recorder()` / `replay(path, …)` | CARLA's built-in `.log` recording. |
| `compile_scenario(spec)` | Materialize a structured spec (map+weather+ego+traffic+adversarials+spectator+wait) by chaining tools. |
| `scenario_sweep(base_spec, vary, capture_after_s)` | Run the same scenario with a parameter swept across a list of values; aggregates world_status per run. |
| `failure_snapshot(actor_id, watch_seconds, sensor)` | Watch for collision / lane-invasion, return event log + frame at moment of impact. |
| `run_openscenario(file, scenario_runner_root?)` | Run an OpenSCENARIO 1.x file via cloned `scenario_runner`. |
| `render_topdown(focus_actor_id?, radius)` | Map + actors bird's-eye PNG. |
| `render_trajectory(actor_id, duration_s, sample_hz, range_m)` | Sample pose for N seconds, plot path on top-down map. |

## Install

Requires CARLA 0.9.16 already installed. The server is a Python package that runs in CARLA's Python venv.

```powershell
# from a clone of this repo
C:\Users\allas\CARLA_0.9.16\.venv\Scripts\python.exe -m pip install -e C:\Users\allas\carla-mcp
```

Verify:
```powershell
C:\Users\allas\CARLA_0.9.16\.venv\Scripts\python.exe -m carla_mcp
```
(Server runs on stdio; Ctrl+C to exit. It's meant to be launched by Claude, not run interactively.)

## Wire to Claude

### Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` (see [`examples/claude_desktop_config.json`](./examples/claude_desktop_config.json)):

```json
{
  "mcpServers": {
    "carla": {
      "command": "C:\\Users\\allas\\CARLA_0.9.16\\.venv\\Scripts\\python.exe",
      "args": ["-m", "carla_mcp"]
    }
  }
}
```

Restart Claude Desktop. The `carla` server should appear in the tools panel.

### Claude Code

Drop [`examples/.mcp.json`](./examples/.mcp.json) into any project root, or merge into `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "carla": {
      "command": "C:\\Users\\allas\\CARLA_0.9.16\\.venv\\Scripts\\python.exe",
      "args": ["-m", "carla_mcp"]
    }
  }
}
```

### claude.ai Custom Connectors (remote, web + mobile)

claude.ai's "Add custom connector" expects a public HTTPS URL. Since CARLA runs on your own machine, expose the MCP server through a tunnel:

```powershell
# 1) Pick a long random secret token (one-time, store somewhere safe)
$env:CARLA_MCP_TOKEN = -join ((1..48) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })
echo $env:CARLA_MCP_TOKEN   # save this value

# 2) Start carla-mcp in HTTP mode
C:\Users\allas\CARLA_0.9.16\.venv\Scripts\python.exe -m carla_mcp --transport http --port 8765
```

In a second terminal, start a quick tunnel (Cloudflare, no account needed):

```powershell
# install once: winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8765
```

Cloudflared prints a URL like `https://random-words-1234.trycloudflare.com`. In claude.ai → **Settings → Connectors → Add custom connector**:

- **URL**: `https://random-words-1234.trycloudflare.com/mcp`
- **Authentication**: Bearer token, value = the `CARLA_MCP_TOKEN` you generated

Then chat from claude.ai (web or mobile) and the model can drive your local CARLA.

**Security notes**

- Without `CARLA_MCP_TOKEN`, the HTTP server is unauthenticated — anyone who guesses the tunnel URL can spawn vehicles in your sim.
- A `trycloudflare.com` URL has decent entropy but is not a substitute for the token.
- For production / shared use, terminate the tunnel with **Cloudflare Access** (free for ≤ 50 users) which adds a real OAuth gate at the edge.
- Stop the tunnel + `carla-mcp` process when you're done.

## Usage examples

Just paste a prompt into Claude. v2.2 onwards, **Claude can start the CARLA simulator itself** via `start_simulator` — you don't need to launch the binary first. The tool short-circuits if it's already running, so it's safe to use unconditionally.

### Quick smoke test

> Make sure CARLA is running (`start_simulator`). Then load Town03, set weather to HardRainNoon, spawn a Tesla and 20 traffic vehicles, wait 3 seconds, and show me what the Tesla sees from the front camera.

Claude chains `start_simulator → load_map → set_weather → spawn_vehicle → spawn_traffic → wait → capture_sensor` and the rendered RGB frame appears inline.

### End-to-end perception eval (the demo run)

Paste this into a fresh chat. Produces ~6 inline images plus a KITTI label dump and an engineer-eye-view summary in roughly 60 seconds of execution (after the initial CARLA launch on first run; ~3 minutes if cold).

> Set up an end-to-end perception eval in CARLA. Use the carla tools and chain them — don't ask me between steps:
>
> 1. **Start the simulator** if it isn't already running: call `start_simulator()`. If it returns `already_running=true` proceed; if `started=true` it just launched and is ready.
> 2. Probe `world_status`. If the current map isn't `Town10HD_Opt`, call `load_map("Town10HD_Opt")`; otherwise skip the reload (avoids the cold-start `load_world()` cost on heavy maps).
> 3. Set weather to `MidRainSunset`.
> 4. **Populate the world first**: 25 traffic vehicles + 8 pedestrians, so the ego spawns into a live scene rather than empty streets.
> 5. Wait 3 seconds for traffic to disperse onto the network.
> 6. **Now spawn the ego**: a Tesla Model 3 with autopilot and `follow_with_spectator=True`. Keep its `actor_id`. The simulator window's spectator camera continuously chases the moving Tesla (v2.1.2+).
> 7. Wait another 2 seconds so the chase shot beds in.
> 8. Front camera RGB, then `compare_seg_with_truth` for the semantic ⟷ BEV ground-truth side-by-side.
> 9. Lidar perception: `render_lidar_3d` with `view="iso"`, then a BEV semantic raster via `capture_semantic_lidar`.
> 10. `compute_lidar_stats` and report point count, max range, and ring uniformity.
> 11. `extract_3d_bboxes` in KITTI format with `max_distance=80`. Show me the first 3 lines.
> 12. Spawn an adversarial cut-in 15m ahead of the ego. Wait 3 seconds.
> 13. Capture the front camera at the moment of the swerve — that's our "interesting frame".
> 14. Reset the world.
>
> After step 13, summarize in one paragraph what an AD perception engineer would learn from this run.

For a different visual flavor, swap step 2 for `Town04` + `HardRainNoon` (highway loop, faster cut-in dynamics) or `Town07` + `CloudySunset` (rural countryside, vegetation-dominated BEV).

## Configuration

Environment variables read at server start:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CARLA_HOST` | `localhost` | Host running the CARLA simulator. |
| `CARLA_PORT` | `2000` | CARLA RPC port. |
| `CARLA_TIMEOUT` | `10.0` | Client connect timeout (seconds). |
| `CARLA_MCP_TRANSPORT` | `stdio` | `stdio` or `http`. |
| `CARLA_MCP_HTTP_HOST` | `127.0.0.1` | Bind addr in HTTP mode (use `0.0.0.0` to expose). |
| `CARLA_MCP_HTTP_PORT` | `8765` | HTTP port. |
| `CARLA_MCP_TOKEN` | *(empty)* | Bearer token required in HTTP mode. Empty = no auth. |
| `CARLA_MCP_LOG` | `%TEMP%/carla-mcp.log` | Path where libcarla's native stdout is redirected in stdio mode (so its `INFO: …` lines don't corrupt the MCP JSON-RPC stream). Tail this file to debug. |

Same flags are also accepted on the command line: `--transport`, `--host`, `--port`.

## License

MIT — see [LICENSE](./LICENSE).
