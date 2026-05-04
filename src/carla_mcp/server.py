"""FastMCP server exposing CARLA simulator control to Claude.

Two transports are supported:

- ``stdio`` (default): launched as a subprocess by Claude Desktop / Claude Code.
- ``http``: streamable HTTP, suitable for claude.ai Custom Connectors via a
  cloudflared tunnel. Optionally protected by a bearer token.

The server connects lazily on first tool call so it can be started before the
CARLA simulator process is ready. Spawned actors are tracked so `reset_world`
only destroys what we created, leaving any user-spawned actors alone.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import carla
import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("carla-mcp")

_CARLA_HOST = os.environ.get("CARLA_HOST", "localhost")
_CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
_CARLA_TIMEOUT = float(os.environ.get("CARLA_TIMEOUT", "10.0"))

_client: carla.Client | None = None
_TRACKED_FILE = Path(tempfile.gettempdir()) / "carla-mcp-tracked.json"
_RIGS_FILE = Path(tempfile.gettempdir()) / "carla-mcp-rigs.json"


# CARLA semantic-tag → CityScapes RGB. Indexed by CARLA's class id.
# Source: CARLA semantic segmentation reference (29-class set in 0.9.16).
CITYSCAPES_PALETTE = np.array(
    [
        [0, 0, 0],          # 0  Unlabeled
        [128, 64, 128],     # 1  Road
        [244, 35, 232],     # 2  Sidewalk
        [70, 70, 70],       # 3  Building
        [102, 102, 156],    # 4  Wall
        [190, 153, 153],    # 5  Fence
        [153, 153, 153],    # 6  Pole
        [250, 170, 30],     # 7  TrafficLight
        [220, 220, 0],      # 8  TrafficSign
        [107, 142, 35],     # 9  Vegetation
        [152, 251, 152],    # 10 Terrain
        [70, 130, 180],     # 11 Sky
        [220, 20, 60],      # 12 Pedestrian
        [255, 0, 0],        # 13 Rider
        [0, 0, 142],        # 14 Car
        [0, 0, 70],         # 15 Truck
        [0, 60, 100],       # 16 Bus
        [0, 80, 100],       # 17 Train
        [0, 0, 230],        # 18 Motorcycle
        [119, 11, 32],      # 19 Bicycle
        [110, 190, 160],    # 20 Static
        [170, 120, 50],     # 21 Dynamic
        [55, 90, 80],       # 22 Other
        [45, 60, 150],      # 23 Water
        [157, 234, 50],     # 24 RoadLine
        [81, 0, 81],        # 25 Ground
        [150, 100, 100],    # 26 Bridge
        [230, 150, 140],    # 27 RailTrack
        [180, 165, 180],    # 28 GuardRail
    ],
    dtype=np.uint8,
)


# CARLA actor type prefix → KITTI class label
_KITTI_CLASS = {
    "vehicle.bicycle": "Cyclist",
    "vehicle.motorcycle": "Cyclist",
    "vehicle": "Car",
    "walker": "Pedestrian",
}


# CARLA semantic class id → human-readable name (29 classes in 0.9.16).
# Index matches CITYSCAPES_PALETTE.
SEMANTIC_NAMES = [
    "Unlabeled", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
    "TrafficLight", "TrafficSign", "Vegetation", "Terrain", "Sky",
    "Pedestrian", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle",
    "Bicycle", "Static", "Dynamic", "Other", "Water", "RoadLine", "Ground",
    "Bridge", "RailTrack", "GuardRail",
]
_NAME_TO_TAG = {n.lower(): i for i, n in enumerate(SEMANTIC_NAMES)}


def _resolve_class_filter(classes: list[str] | None) -> set[int] | None:
    """Convert a list of class names (case-insensitive) to a set of CARLA tag ids.
    Unknown names raise ValueError. None means no filter (keep all classes)."""
    if classes is None:
        return None
    out: set[int] = set()
    for c in classes:
        key = c.strip().lower()
        if key not in _NAME_TO_TAG:
            raise ValueError(
                f"Unknown semantic class {c!r}. Valid names: {SEMANTIC_NAMES}"
            )
        out.add(_NAME_TO_TAG[key])
    return out


def _get_client() -> carla.Client:
    global _client
    if _client is None:
        _client = carla.Client(_CARLA_HOST, _CARLA_PORT)
        _client.set_timeout(_CARLA_TIMEOUT)
    return _client


def _world() -> carla.World:
    return _get_client().get_world()


def _load_tracked() -> set[int]:
    """Read the cross-process tracked-actor set from a temp JSON file.

    Persisting on disk lets multiple Claude sessions / Desktop+Code coexist:
    each subprocess sees the same set, so reset_world() can clean up actors
    spawned by *any* previous session.
    """
    try:
        return set(json.loads(_TRACKED_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return set()


def _save_tracked(ids: set[int]) -> None:
    tmp = _TRACKED_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(ids)), encoding="utf-8")
    tmp.replace(_TRACKED_FILE)


def _track(actor: carla.Actor) -> int:
    ids = _load_tracked()
    ids.add(actor.id)
    _save_tracked(ids)
    return actor.id


def _load_rigs() -> dict[str, Any]:
    try:
        return json.loads(_RIGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save_rigs(rigs: dict[str, Any]) -> None:
    tmp = _RIGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rigs), encoding="utf-8")
    tmp.replace(_RIGS_FILE)


def _png(arr: np.ndarray) -> Image:
    """Encode an HxWx3 uint8 ndarray as a PNG MCP Image."""
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return Image(data=buf.getvalue(), format="png")


def _capture_one_camera(
    parent: carla.Actor,
    bp_id: str,
    converter: carla.ColorConverter | None,
    width: int,
    height: int,
    fov: float,
    transform: carla.Transform | None = None,
) -> np.ndarray:
    """Spawn a one-shot camera, capture a single frame, return RGB ndarray."""
    w = _world()
    cam_bp = w.get_blueprint_library().find(bp_id)
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))

    if transform is None:
        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    camera = w.spawn_actor(cam_bp, transform, attach_to=parent)
    q: queue.Queue[carla.Image] = queue.Queue()
    camera.listen(q.put)
    try:
        if w.get_settings().synchronous_mode:
            w.tick()
        else:
            w.wait_for_tick()
        img = q.get(timeout=5.0)
        if converter is not None:
            img.convert(converter)
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            (img.height, img.width, 4)
        )
        return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
    finally:
        camera.stop()
        camera.destroy()


def _capture_one_lidar(
    parent: carla.Actor,
    semantic: bool,
    channels: int = 32,
    range_m: float = 50.0,
    pps: int = 100_000,
    rotation_freq: float = 10.0,
    upper_fov: float = 10.0,
    lower_fov: float = -30.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Capture one lidar sweep. Returns a 3-tuple (points, tags, obj_idx).

    Non-semantic:  (N,4) points (x,y,z,intensity), tags=None, obj_idx=None.
    Semantic:      (N,3) points (x,y,z),           tags=(N,) uint32 class ids,
                                                    obj_idx=(N,) uint32 instance ids.
    """
    w = _world()
    bp_id = "sensor.lidar.ray_cast_semantic" if semantic else "sensor.lidar.ray_cast"
    bp = w.get_blueprint_library().find(bp_id)
    bp.set_attribute("channels", str(channels))
    bp.set_attribute("range", str(range_m))
    bp.set_attribute("points_per_second", str(pps))
    bp.set_attribute("rotation_frequency", str(rotation_freq))
    bp.set_attribute("upper_fov", str(upper_fov))
    bp.set_attribute("lower_fov", str(lower_fov))

    transform = carla.Transform(carla.Location(z=2.5))
    lidar = w.spawn_actor(bp, transform, attach_to=parent)
    q: queue.Queue[Any] = queue.Queue()
    lidar.listen(q.put)
    try:
        # Lidar fires once per rotation. Wait up to 2 rotation periods for
        # the first complete sweep. Avoid drain loops — they're unbounded
        # in async mode (the simulator keeps emitting new sweeps faster
        # than we can drain).
        timeout_s = max(0.5, 2.0 / float(rotation_freq))
        last = q.get(timeout=timeout_s)
        if semantic:
            data = np.frombuffer(
                last.raw_data,
                dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                                ("CosAngle", "f4"), ("ObjIdx", "u4"), ("ObjTag", "u4")]),
            )
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
            tags = data["ObjTag"].astype(np.uint32)
            obj_idx = data["ObjIdx"].astype(np.uint32)
            return xyz, tags, obj_idx
        else:
            data = np.frombuffer(last.raw_data, dtype=np.float32).reshape(-1, 4)
            return data, None, None
    finally:
        lidar.stop()
        lidar.destroy()


def _bev_from_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    range_m: float = 50.0,
    resolution: float = 0.2,
    bg: tuple[int, int, int] = (15, 15, 25),
) -> np.ndarray:
    """Rasterize XY of a point cloud into a top-down image.

    points : (N,2+) ndarray of x,y,(z,intensity,...) in sensor frame
    colors : optional (N,3) uint8 per-point colors. If None, intensity-coded grayscale.
    """
    size = int(2 * range_m / resolution)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    xs = points[:, 0]
    ys = points[:, 1]
    mask = (np.abs(xs) < range_m) & (np.abs(ys) < range_m)
    xs, ys = xs[mask], ys[mask]
    px = ((xs + range_m) / resolution).astype(np.int32)
    py = ((range_m - ys) / resolution).astype(np.int32)
    if colors is None:
        if points.shape[1] >= 4:
            inten = points[mask, 3]
            inten = np.clip(inten / max(1e-6, inten.max()), 0, 1)
            g = (inten * 255).astype(np.uint8)
            img[py, px] = np.stack([g, g, g], axis=1)
        else:
            img[py, px] = (200, 200, 200)
    else:
        img[py, px] = colors[mask]
    return img


def _follow_with_spectator(actor: carla.Actor, distance: float = 8.0, height: float = 4.0, pitch: float = -15.0) -> None:
    w = _world()
    t = actor.get_transform()
    forward = t.get_forward_vector()
    loc = carla.Location(
        x=t.location.x - forward.x * distance,
        y=t.location.y - forward.y * distance,
        z=t.location.z + height,
    )
    w.get_spectator().set_transform(
        carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=t.rotation.yaw))
    )


WEATHER_PRESETS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
}


@mcp.tool()
def world_status() -> dict[str, Any]:
    """Return current map name, weather, actor counts, and sim time.

    Use this to confirm the simulator is reachable and inspect the live world.
    """
    w = _world()
    weather = w.get_weather()
    actors = w.get_actors()
    snap = w.get_snapshot()
    tracked = _load_tracked()
    return {
        "carla_version": _get_client().get_server_version(),
        "map": w.get_map().name.split("/")[-1],
        "available_maps": [m.split("/")[-1] for m in _get_client().get_available_maps()],
        "weather": {
            "cloudiness": weather.cloudiness,
            "precipitation": weather.precipitation,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density,
            "wetness": weather.wetness,
        },
        "actors": {
            "vehicles": len(actors.filter("vehicle.*")),
            "walkers": len(actors.filter("walker.*")),
            "sensors": len(actors.filter("sensor.*")),
            "total": len(actors),
        },
        "tracked_by_server": len(tracked),
        "tracked_actor_ids": sorted(tracked),
        "sim_time": snap.timestamp.elapsed_seconds,
        "frame": snap.frame,
    }


@mcp.tool()
def load_map(name: str) -> dict[str, Any]:
    """Reload the simulator with a different town. Destroys all current actors.

    Args:
        name: Map name like "Town01", "Town03", "Town10HD_Opt", "Town12".
              Use world_status() to see available_maps.
    """
    available = [m.split("/")[-1] for m in _get_client().get_available_maps()]
    if name not in available:
        matches = [m for m in available if name.lower() in m.lower()]
        raise ValueError(
            f"Map {name!r} not found. Did you mean one of {matches}? "
            f"Full list available via world_status()."
        )
    _save_tracked(set())  # reload destroys all actors anyway
    new_world = _get_client().load_world(name)
    return {
        "loaded": new_world.get_map().name.split("/")[-1],
        "spawn_points": len(new_world.get_map().get_spawn_points()),
        "warning": (
            "load_map destroys ALL actors in the world, including any "
            "manual_control.py / pygame ego vehicle that was running. "
            "Restart that client if you want a drivable car back."
        ),
    }


@mcp.tool()
def set_weather(
    preset: str | None = None,
    cloudiness: float | None = None,
    precipitation: float | None = None,
    sun_altitude_angle: float | None = None,
    fog_density: float | None = None,
    wetness: float | None = None,
) -> dict[str, Any]:
    """Apply a weather preset or override individual parameters (0-100 scales).

    Args:
        preset: One of the named presets, e.g. "ClearNoon", "HardRainNoon",
                "WetCloudySunset". Pass None to keep current and only adjust
                individual parameters.
        cloudiness: 0=clear, 100=fully overcast.
        precipitation: 0=dry, 100=heavy rain.
        sun_altitude_angle: -90 (midnight) to +90 (noon).
        fog_density: 0=clear, 100=very thick fog.
        wetness: 0=dry roads, 100=fully wet.
    """
    w = _world()
    if preset is not None:
        if preset not in WEATHER_PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {sorted(WEATHER_PRESETS)}"
            )
        weather = WEATHER_PRESETS[preset]
    else:
        weather = w.get_weather()
    for attr, val in [
        ("cloudiness", cloudiness),
        ("precipitation", precipitation),
        ("sun_altitude_angle", sun_altitude_angle),
        ("fog_density", fog_density),
        ("wetness", wetness),
    ]:
        if val is not None:
            setattr(weather, attr, float(val))
    w.set_weather(weather)
    return {
        "applied_preset": preset,
        "final": {
            "cloudiness": weather.cloudiness,
            "precipitation": weather.precipitation,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density,
            "wetness": weather.wetness,
        },
    }


@mcp.tool()
def spawn_vehicle(
    model: str = "vehicle.tesla.model3",
    spawn_point_index: int | None = None,
    autopilot: bool = False,
    follow_with_spectator: bool = True,
) -> dict[str, Any]:
    """Spawn a vehicle and return its actor id (use it with capture_sensor).

    Args:
        model: Blueprint id, e.g. "vehicle.tesla.model3", "vehicle.audi.tt",
               "vehicle.carlamotors.firetruck". Filter expressions like
               "vehicle.tesla.*" pick a random match.
        spawn_point_index: Index into world.get_map().get_spawn_points().
                           None = random.
        autopilot: Hand control to CARLA's Traffic Manager.
        follow_with_spectator: Move the simulator's main viewport to chase
            the spawned vehicle. Default True so the user immediately sees
            what was spawned in the CARLA window.
    """
    w = _world()
    bps = w.get_blueprint_library().filter(model)
    if not bps:
        raise ValueError(f"No vehicle blueprint matches {model!r}.")
    bp = random.choice(bps)
    spawn_points = w.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points on this map.")
    transform = (
        spawn_points[spawn_point_index]
        if spawn_point_index is not None
        else random.choice(spawn_points)
    )
    vehicle = w.spawn_actor(bp, transform)
    if autopilot:
        vehicle.set_autopilot(True)
    if follow_with_spectator:
        _follow_with_spectator(vehicle)
    return {
        "actor_id": _track(vehicle),
        "type_id": vehicle.type_id,
        "location": [transform.location.x, transform.location.y, transform.location.z],
        "autopilot": autopilot,
        "spectator_following": follow_with_spectator,
    }


@mcp.tool()
def set_spectator(
    actor_id: int | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    pitch: float = -15.0,
    distance: float = 8.0,
    height: float = 4.0,
) -> dict[str, Any]:
    """Move the simulator's main viewport (the spectator camera).

    The view in the CARLA window itself is the "spectator" — a free-flying
    camera. Use this tool to make the sim window track whatever you're
    working with, so the user sees Claude's actions live.

    Args:
        actor_id: Follow this actor with a chase camera (preferred mode).
        x, y, z: Or teleport to absolute world coordinates.
        pitch: Camera pitch in degrees (negative = looking down).
        distance: Chase camera distance behind the actor (meters).
        height: Chase camera height above the actor (meters).
    """
    w = _world()
    spectator = w.get_spectator()
    if actor_id is not None:
        actor = w.get_actor(actor_id)
        if actor is None:
            raise ValueError(f"Actor {actor_id} not found.")
        _follow_with_spectator(actor, distance=distance, height=height, pitch=pitch)
        loc = actor.get_transform().location
        return {"following_actor": actor_id, "actor_location": [loc.x, loc.y, loc.z]}
    if x is not None and y is not None and z is not None:
        loc = carla.Location(x=float(x), y=float(y), z=float(z))
        spectator.set_transform(
            carla.Transform(loc, carla.Rotation(pitch=float(pitch), yaw=0.0))
        )
        return {"teleported_to": [x, y, z]}
    raise ValueError("Provide actor_id (to follow) or x, y, z (to teleport).")


@mcp.tool()
def spawn_traffic(n_vehicles: int = 30, n_pedestrians: int = 0) -> dict[str, Any]:
    """Populate the world with autopilot vehicles (and optionally pedestrians).

    All spawned actors are tracked and removed by reset_world().
    """
    w = _world()
    tm = _get_client().get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)

    vehicle_bps = list(w.get_blueprint_library().filter("vehicle.*"))
    spawn_points = w.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    n_v = min(n_vehicles, len(spawn_points))

    spawned_v = 0
    for i in range(n_v):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        actor = w.try_spawn_actor(bp, spawn_points[i])
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            _track(actor)
            spawned_v += 1

    spawned_p = 0
    if n_pedestrians > 0:
        walker_bps = w.get_blueprint_library().filter("walker.pedestrian.*")
        controller_bp = w.get_blueprint_library().find("controller.ai.walker")
        for _ in range(n_pedestrians):
            loc = w.get_random_location_from_navigation()
            if loc is None:
                continue
            walker = w.try_spawn_actor(
                random.choice(walker_bps), carla.Transform(loc)
            )
            if walker is None:
                continue
            controller = w.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                continue
            w.tick() if w.get_settings().synchronous_mode else w.wait_for_tick()
            controller.start()
            controller.go_to_location(w.get_random_location_from_navigation())
            controller.set_max_speed(1.4)
            _track(walker)
            _track(controller)
            spawned_p += 1

    return {"vehicles_spawned": spawned_v, "pedestrians_spawned": spawned_p}


@mcp.tool()
def render_topdown(
    focus_actor_id: int | None = None,
    radius: float = 100.0,
    show_walkers: bool = True,
) -> Image:
    """Bird's-eye view of the world: road network + all vehicles + walkers.

    A v2 preview tool. Gives Claude map-level awareness — useful when sensor
    captures don't tell the full story (traffic positions, route layout,
    where a spawned car ended up).

    Args:
        focus_actor_id: Center the view on this actor (drawn red). If None,
            centers on the world origin.
        radius: View radius in meters.
        show_walkers: Draw pedestrians as green dots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    w = _world()

    if focus_actor_id is not None:
        focus_actor = w.get_actor(focus_actor_id)
        if focus_actor is None:
            raise ValueError(f"Actor {focus_actor_id} not found.")
        c = focus_actor.get_transform().location
        cx, cy = c.x, c.y
    else:
        cx, cy = 0.0, 0.0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)

    # Road network as light-gray scatter (cheap, no shapely needed)
    waypoints = w.get_map().generate_waypoints(2.0)
    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    ax.scatter(xs, ys, s=1, c="#cccccc", alpha=0.6, linewidths=0)

    actors = w.get_actors()
    for actor in actors.filter("vehicle.*"):
        t = actor.get_transform()
        x, y, yaw = t.location.x, t.location.y, t.rotation.yaw
        is_focus = actor.id == focus_actor_id
        rect = patches.Rectangle(
            (x - 2.3, y - 1.0),
            4.6,
            2.0,
            angle=yaw,
            rotation_point=(x, y),
            facecolor="#e63946" if is_focus else "#1d3557",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.add_patch(rect)

    if show_walkers:
        for actor in actors.filter("walker.*"):
            t = actor.get_transform()
            ax.scatter(t.location.x, t.location.y, s=18, c="#2a9d8f", marker="o")

    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Top-down — {w.get_map().name.split('/')[-1]}"
        + (f"  •  focus actor {focus_actor_id}" if focus_actor_id else "")
    )
    ax.grid(True, alpha=0.25, linestyle=":")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def capture_sensor(
    actor_id: int,
    sensor: str = "rgb",
    width: int = 800,
    height: int = 600,
    fov: float = 90.0,
) -> Image:
    """Attach a camera to the given actor, capture one frame, return PNG.

    This is the killer tool: it gives Claude direct visual feedback on what
    the simulated vehicle sees, so it can reason about scenes the way a human
    test driver would.

    Args:
        actor_id: Vehicle returned by spawn_vehicle.
        sensor: One of:
            - "rgb"                   (standard camera)
            - "depth"                 (logarithmic depth)
            - "semantic"              (semantic segmentation, CityScapes palette)
            - "instance_segmentation" (per-instance unique colors, CARLA 0.9.13+)
            - "optical_flow"          (color-coded forward flow)
            - "dvs"                   (event camera, 5-frame accumulation, R=positive B=negative)
            For lidar see capture_lidar / capture_semantic_lidar.
        width, height: Image resolution.
        fov: Horizontal field of view in degrees.
    """
    w = _world()
    parent = w.get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")

    cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    bplib = w.get_blueprint_library()

    if sensor in ("rgb", "depth", "semantic", "instance_segmentation"):
        bp_id = {
            "rgb": "sensor.camera.rgb",
            "depth": "sensor.camera.depth",
            "semantic": "sensor.camera.semantic_segmentation",
            "instance_segmentation": "sensor.camera.instance_segmentation",
        }[sensor]
        converter = {
            "rgb": carla.ColorConverter.Raw,
            "depth": carla.ColorConverter.LogarithmicDepth,
            "semantic": carla.ColorConverter.CityScapesPalette,
            "instance_segmentation": carla.ColorConverter.Raw,
        }[sensor]
        rgb = _capture_one_camera(parent, bp_id, converter, width, height, fov, cam_transform)
        return _png(rgb)

    if sensor == "optical_flow":
        cam_bp = bplib.find("sensor.camera.optical_flow")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        camera = w.spawn_actor(cam_bp, cam_transform, attach_to=parent)
        q: queue.Queue[Any] = queue.Queue()
        camera.listen(q.put)
        try:
            (w.tick() if w.get_settings().synchronous_mode else w.wait_for_tick())
            img = q.get(timeout=5.0)
            colored = img.get_color_coded_flow()  # carla.Image
            arr = np.frombuffer(colored.raw_data, dtype=np.uint8).reshape(
                (colored.height, colored.width, 4)
            )
            return _png(arr[:, :, :3][:, :, ::-1].copy())
        finally:
            camera.stop()
            camera.destroy()

    if sensor == "dvs":
        cam_bp = bplib.find("sensor.camera.dvs")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        camera = w.spawn_actor(cam_bp, cam_transform, attach_to=parent)
        q: queue.Queue[Any] = queue.Queue()
        camera.listen(q.put)
        try:
            # Accumulate up to 5 measurements. CARLA 0.9.16's
            # DVSEventArray.to_array() raises a boost::python TypeError on
            # some installs; iterate per-event manually instead.
            xs_pos: list[int] = []
            ys_pos: list[int] = []
            xs_neg: list[int] = []
            ys_neg: list[int] = []
            for _ in range(5):
                (w.tick() if w.get_settings().synchronous_mode else w.wait_for_tick())
                try:
                    measurement = q.get(timeout=2.0)
                except queue.Empty:
                    continue
                for e in measurement:
                    if e.pol:
                        xs_pos.append(e.x); ys_pos.append(e.y)
                    else:
                        xs_neg.append(e.x); ys_neg.append(e.y)
            arr = np.full((height, width, 3), 0, dtype=np.uint8)
            if xs_pos:
                xs = np.clip(np.array(xs_pos), 0, width - 1)
                ys = np.clip(np.array(ys_pos), 0, height - 1)
                arr[ys, xs] = (255, 60, 60)
            if xs_neg:
                xs = np.clip(np.array(xs_neg), 0, width - 1)
                ys = np.clip(np.array(ys_neg), 0, height - 1)
                arr[ys, xs] = (60, 60, 255)
            return _png(arr)
        finally:
            camera.stop()
            camera.destroy()

    raise ValueError(
        f"sensor must be one of: rgb | depth | semantic | instance_segmentation | "
        f"optical_flow | dvs. For lidar use capture_lidar / capture_semantic_lidar."
    )


@mcp.tool()
def wait(seconds: float = 1.0) -> dict[str, Any]:
    """Let the simulator run for a few wall-clock seconds (async mode).

    Useful between spawning a vehicle and capturing a sensor frame so the
    vehicle has time to settle on the road and surrounding traffic ticks.
    """
    seconds = max(0.0, min(seconds, 30.0))
    t0 = time.time()
    time.sleep(seconds)
    return {"slept_seconds": time.time() - t0}


@mcp.tool()
def reset_world(also_clear_tracked: bool = True) -> dict[str, Any]:
    """Destroy all actors spawned by this server. Other actors are left alone.

    Args:
        also_clear_tracked: If False, just report what would be destroyed.
    """
    w = _world()
    destroyed = 0
    failed = 0
    tracked = _load_tracked()
    if also_clear_tracked:
        for aid in list(tracked):
            actor = w.get_actor(aid)
            if actor is not None and actor.is_alive:
                try:
                    actor.destroy()
                    destroyed += 1
                except Exception:
                    failed += 1
        tracked.clear()
        _save_tracked(tracked)
    return {
        "destroyed": destroyed,
        "failed": failed,
        "still_tracked": len(tracked),
    }


# ====================================================================
# v0.5 — direct CARLA API wrappers
# ====================================================================

@mcp.tool()
def start_recorder(path: str = "carla-mcp.log", additional_data: bool = True) -> dict[str, Any]:
    """Start recording the simulation to a CARLA `.log` file (replayable).

    Args:
        path: Filename. Relative paths land in CARLA's recordings dir
              (typically `CarlaUE4/Saved/`); pass an absolute path to
              control location precisely.
        additional_data: Record extra data (vehicle physics state) for richer
              replays. Slightly larger files.
    """
    actual = _get_client().start_recorder(path, additional_data)
    return {"recording_to": actual}


@mcp.tool()
def stop_recorder() -> dict[str, Any]:
    """Stop the active CARLA recording."""
    _get_client().stop_recorder()
    return {"stopped": True}


@mcp.tool()
def replay(
    path: str,
    start: float = 0.0,
    duration: float = 0.0,
    follow_actor_id: int = 0,
    replay_sensors: bool = False,
) -> dict[str, Any]:
    """Replay a previously-recorded `.log` file.

    Args:
        path: Path to the log (same conventions as start_recorder).
        start: Start time in seconds (negative = from end).
        duration: Replay duration; 0 = until end.
        follow_actor_id: Spectator follows this actor's id from the recording.
                         0 = free-camera spectator.
        replay_sensors: Also replay sensor data. Most replays leave this off.
    """
    info = _get_client().replay_file(path, start, duration, follow_actor_id, replay_sensors)
    return {"replay_started": True, "info": info}


@mcp.tool()
def list_traffic_lights() -> list[dict[str, Any]]:
    """List all traffic lights in the world with id, location, and current state."""
    out = []
    for tl in _world().get_actors().filter("traffic.traffic_light"):
        loc = tl.get_transform().location
        out.append({
            "id": tl.id,
            "location": [loc.x, loc.y, loc.z],
            "state": str(tl.state).split(".")[-1],
            "frozen": tl.is_frozen(),
        })
    return out


@mcp.tool()
def set_traffic_light(actor_id: int, state: str = "red", freeze: bool = True) -> dict[str, Any]:
    """Override the state of a single traffic light.

    Args:
        actor_id: From list_traffic_lights().
        state: "red" | "yellow" | "green" | "off".
        freeze: If True (default), keep the light at this state until unfrozen.
    """
    tl = _world().get_actor(actor_id)
    if tl is None or not tl.type_id.startswith("traffic.traffic_light"):
        raise ValueError(f"Actor {actor_id} is not a traffic light.")
    state_map = {
        "red": carla.TrafficLightState.Red,
        "yellow": carla.TrafficLightState.Yellow,
        "green": carla.TrafficLightState.Green,
        "off": carla.TrafficLightState.Off,
    }
    if state not in state_map:
        raise ValueError(f"state must be one of {list(state_map)}")
    tl.set_state(state_map[state])
    tl.freeze(bool(freeze))
    return {"id": actor_id, "state": state, "frozen": freeze}


@mcp.tool()
def set_actor_behavior(
    actor_id: int,
    autopilot: bool | None = None,
    ignore_traffic_lights_pct: float | None = None,
    ignore_signs_pct: float | None = None,
    ignore_walkers_pct: float | None = None,
    distance_to_leading_vehicle: float | None = None,
    speed_difference_pct: float | None = None,
    auto_lane_change: bool | None = None,
) -> dict[str, Any]:
    """Configure a vehicle's Traffic Manager behavior.

    Each parameter is optional and only applied if not None.

    Args:
        autopilot: Hand control to the Traffic Manager.
        ignore_traffic_lights_pct: 0=respect, 100=always ignore.
        ignore_signs_pct: same scale.
        ignore_walkers_pct: same scale.
        distance_to_leading_vehicle: meters.
        speed_difference_pct: -50 = 50%% slower than speed limit; +20 = 20%% faster.
        auto_lane_change: allow TM-driven lane changes.
    """
    actor = _world().get_actor(actor_id)
    if actor is None or not actor.type_id.startswith("vehicle."):
        raise ValueError(f"Actor {actor_id} is not a vehicle.")
    tm = _get_client().get_trafficmanager()
    applied = {}
    if autopilot is not None:
        actor.set_autopilot(bool(autopilot), tm.get_port())
        applied["autopilot"] = autopilot
    if ignore_traffic_lights_pct is not None:
        tm.ignore_lights_percentage(actor, float(ignore_traffic_lights_pct))
        applied["ignore_traffic_lights_pct"] = ignore_traffic_lights_pct
    if ignore_signs_pct is not None:
        tm.ignore_signs_percentage(actor, float(ignore_signs_pct))
        applied["ignore_signs_pct"] = ignore_signs_pct
    if ignore_walkers_pct is not None:
        tm.ignore_walkers_percentage(actor, float(ignore_walkers_pct))
        applied["ignore_walkers_pct"] = ignore_walkers_pct
    if distance_to_leading_vehicle is not None:
        tm.distance_to_leading_vehicle(actor, float(distance_to_leading_vehicle))
        applied["distance_to_leading_vehicle"] = distance_to_leading_vehicle
    if speed_difference_pct is not None:
        tm.vehicle_percentage_speed_difference(actor, float(speed_difference_pct))
        applied["speed_difference_pct"] = speed_difference_pct
    if auto_lane_change is not None:
        tm.auto_lane_change(actor, bool(auto_lane_change))
        applied["auto_lane_change"] = auto_lane_change
    return {"actor_id": actor_id, "applied": applied}


@mcp.tool()
def spawn_pedestrian(n: int = 1, ai: bool = True, max_speed: float = 1.4) -> dict[str, Any]:
    """Spawn pedestrians (walkers) with optional AI controllers.

    Args:
        n: How many to spawn (capped by available navigation locations).
        ai: Attach AI controllers that walk to random nav points.
        max_speed: meters/second when ai=True.
    """
    w = _world()
    walker_bps = w.get_blueprint_library().filter("walker.pedestrian.*")
    controller_bp = w.get_blueprint_library().find("controller.ai.walker")
    spawned, controllers = 0, 0
    for _ in range(n):
        loc = w.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(walker_bps)
        walker = w.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            continue
        _track(walker)
        spawned += 1
        if ai:
            controller = w.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is not None:
                w.tick() if w.get_settings().synchronous_mode else w.wait_for_tick()
                controller.start()
                controller.go_to_location(w.get_random_location_from_navigation())
                controller.set_max_speed(float(max_speed))
                _track(controller)
                controllers += 1
    return {"walkers_spawned": spawned, "controllers_attached": controllers}


# ====================================================================
# v1.5 — lidar + 3D / segmentation analysis (the differentiator)
# ====================================================================

@mcp.tool()
def capture_lidar(
    actor_id: int,
    channels: int = 32,
    range_m: float = 50.0,
    points_per_second: int = 100_000,
    rotation_freq: float = 10.0,
    resolution: float = 0.2,
) -> Image:
    """Capture one lidar sweep attached to the actor, return a top-down BEV PNG.

    Intensity is encoded as grayscale brightness.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    pts, _, _ = _capture_one_lidar(
        parent, semantic=False, channels=channels, range_m=range_m,
        pps=points_per_second, rotation_freq=rotation_freq,
    )
    bev = _bev_from_points(pts, range_m=range_m, resolution=resolution)
    return _png(bev)


@mcp.tool()
def capture_semantic_lidar(
    actor_id: int,
    channels: int = 32,
    range_m: float = 50.0,
    points_per_second: int = 100_000,
    rotation_freq: float = 10.0,
    resolution: float = 0.2,
) -> Image:
    """Capture one semantic-lidar sweep, return a top-down BEV PNG with
    per-point CityScapes class colors.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    xyz, tags, _ = _capture_one_lidar(
        parent, semantic=True, channels=channels, range_m=range_m,
        pps=points_per_second, rotation_freq=rotation_freq,
    )
    tags_clipped = np.clip(tags, 0, len(CITYSCAPES_PALETTE) - 1)
    colors = CITYSCAPES_PALETTE[tags_clipped]
    bev = _bev_from_points(xyz, colors=colors, range_m=range_m, resolution=resolution)
    return _png(bev)


@mcp.tool()
def render_lidar_3d(
    actor_id: int,
    view: str = "iso",
    channels: int = 32,
    range_m: float = 50.0,
    points_per_second: int = 100_000,
    semantic: bool = True,
) -> Image:
    """Off-screen 3D render of a single lidar sweep using matplotlib.

    Args:
        view: "iso" | "bev" | "rear" | "side".
        semantic: If True, color points by semantic class; else intensity grayscale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (registers 3d projection)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")

    if semantic:
        xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
        tags_clipped = np.clip(tags, 0, len(CITYSCAPES_PALETTE) - 1)
        colors = CITYSCAPES_PALETTE[tags_clipped] / 255.0
    else:
        pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
        xyz = pts[:, :3]
        inten = pts[:, 3]
        inten = np.clip(inten / max(1e-6, inten.max()), 0, 1)
        colors = np.stack([inten, inten, inten], axis=1)

    # Subsample if huge (matplotlib 3D scatter is O(N))
    if len(xyz) > 30_000:
        idx = np.random.choice(len(xyz), 30_000, replace=False)
        xyz, colors = xyz[idx], colors[idx]

    fig = plt.figure(figsize=(8, 8), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1, depthshade=False)
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_zlim(-2, 8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Lidar {'semantic' if semantic else 'intensity'} — {view} view")
    if view == "iso":
        ax.view_init(elev=25, azim=-60)
    elif view == "bev":
        ax.view_init(elev=89, azim=-90)
    elif view == "rear":
        ax.view_init(elev=8, azim=180)
    elif view == "side":
        ax.view_init(elev=8, azim=90)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def point_cloud_clusters(
    actor_id: int,
    eps: float = 0.5,
    min_samples: int = 10,
    range_m: float = 50.0,
    channels: int = 32,
    points_per_second: int = 100_000,
    resolution: float = 0.2,
) -> dict[str, Any]:
    """Run DBSCAN on a lidar sweep, return BEV PNG with cluster colors and a list of cluster centroids/sizes (proposal boxes).

    Useful as a quick object-proposal step or as ground-truth comparison
    against an ML detector's output.
    """
    from sklearn.cluster import DBSCAN

    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
    xyz = pts[:, :3]
    # Drop ground returns (rough heuristic: z < -1.5 is ground from a 2.5m sensor)
    above_ground = xyz[:, 2] > -1.4
    xyz_obj = xyz[above_ground]

    if len(xyz_obj) < min_samples:
        return {"clusters": 0, "image": None, "warning": "too few points"}

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(xyz_obj[:, :2])
    labels = db.labels_
    n_clusters = int(labels.max() + 1)

    # color each cluster differently; -1 (noise) stays gray
    rng = np.random.default_rng(seed=0)
    palette = (rng.random((max(1, n_clusters), 3)) * 255).astype(np.uint8)
    colors = np.full((len(xyz_obj), 3), 90, dtype=np.uint8)
    for c in range(n_clusters):
        colors[labels == c] = palette[c]
    bev = _bev_from_points(xyz_obj, colors=colors, range_m=range_m, resolution=resolution)

    proposals = []
    for c in range(n_clusters):
        sel = xyz_obj[labels == c]
        if len(sel) < 5:
            continue
        cx, cy, cz = sel.mean(axis=0)
        ext = (sel.max(axis=0) - sel.min(axis=0)) / 2
        proposals.append({
            "cluster_id": c,
            "n_points": int(len(sel)),
            "centroid": [float(cx), float(cy), float(cz)],
            "extent": [float(ext[0]), float(ext[1]), float(ext[2])],
        })

    return {
        "clusters": n_clusters,
        "noise_points": int((labels == -1).sum()),
        "proposals": proposals,
        "image": _png(bev),
    }


@mcp.tool()
def extract_3d_bboxes(
    observer_actor_id: int,
    max_distance: float = 100.0,
    format: str = "json",
) -> dict[str, Any]:
    """Ground-truth 3D bounding boxes of all vehicles + walkers, in observer frame.

    Args:
        observer_actor_id: Frame of reference (typically the ego vehicle).
        max_distance: Filter actors farther than this (meters).
        format: "json" | "kitti" | "nuscenes".

    Returns the bounding box list inline. Use this to generate ground-truth
    labels for a perception-network evaluation.
    """
    w = _world()
    obs = w.get_actor(observer_actor_id)
    if obs is None:
        raise ValueError(f"Actor {observer_actor_id} not found.")
    obs_t = obs.get_transform()
    obs_inv = np.array(obs_t.get_inverse_matrix())

    boxes = []
    for actor in w.get_actors():
        if actor.id == observer_actor_id:
            continue
        if not (actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.")):
            continue
        bb = actor.bounding_box  # in actor's local frame
        a_t = actor.get_transform()
        # World-frame box center
        world_loc = a_t.transform(bb.location)
        # Skip if too far
        d = ((world_loc.x - obs_t.location.x) ** 2 + (world_loc.y - obs_t.location.y) ** 2) ** 0.5
        if d > max_distance:
            continue
        # Convert center to observer frame
        v = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
        local = obs_inv @ v
        # KITTI-style: dims = h, w, l (z, y, x extents *2)
        h, wdt, lng = 2 * bb.extent.z, 2 * bb.extent.y, 2 * bb.extent.x
        # Yaw of actor relative to observer (radians)
        yaw_rel = np.deg2rad(a_t.rotation.yaw - obs_t.rotation.yaw)
        cls = "Car"
        if actor.type_id.startswith("walker."):
            cls = "Pedestrian"
        elif "motorcycle" in actor.type_id or "bicycle" in actor.type_id:
            cls = "Cyclist"
        boxes.append({
            "actor_id": actor.id,
            "class": cls,
            "type_id": actor.type_id,
            "center_local": [float(local[0]), float(local[1]), float(local[2])],
            "dimensions_hwl": [float(h), float(wdt), float(lng)],
            "yaw_rel": float(yaw_rel),
            "distance": float(d),
        })

    if format == "json":
        return {"format": "json", "n_boxes": len(boxes), "boxes": boxes}

    if format == "kitti":
        # KITTI label format (one line per object):
        # type truncated occluded alpha bbox(4) dim(h,w,l) loc(x,y,z) ry
        lines = []
        for b in boxes:
            x, y, z = b["center_local"]
            h_, wd, ln = b["dimensions_hwl"]
            ry = b["yaw_rel"]
            lines.append(f"{b['class']} 0 0 0 0 0 0 0 {h_:.2f} {wd:.2f} {ln:.2f} "
                         f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f}")
        return {"format": "kitti", "n_boxes": len(boxes), "labels": "\n".join(lines)}

    if format == "nuscenes":
        # nuScenes-ish JSON: each box with translation/size/rotation
        items = []
        for b in boxes:
            x, y, z = b["center_local"]
            h_, wd, ln = b["dimensions_hwl"]
            items.append({
                "category_name": b["class"].lower(),
                "translation": [x, y, z],
                "size": [wd, ln, h_],
                "rotation_yaw": b["yaw_rel"],
                "instance_token": str(b["actor_id"]),
            })
        return {"format": "nuscenes", "n_boxes": len(items), "boxes": items}

    raise ValueError("format must be 'json' | 'kitti' | 'nuscenes'")


@mcp.tool()
def render_bev_segmentation(
    actor_id: int,
    range_m: float = 50.0,
    resolution: float = 0.25,
) -> Image:
    """Top-down semantic raster centered on ego: drivable area + vehicles + walkers.

    Renders from ground-truth actor positions (not lidar). Drivable area
    comes from the OpenDRIVE road waypoints. Useful as an HD-map-style
    ground-truth label for BEV perception models.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    w = _world()
    ego = w.get_actor(actor_id)
    if ego is None:
        raise ValueError(f"Actor {actor_id} not found.")
    ego_t = ego.get_transform()
    cx, cy = ego_t.location.x, ego_t.location.y

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.set_facecolor(tuple(c / 255 for c in CITYSCAPES_PALETTE[0]))  # unlabeled bg

    # Drivable area = light gray rectangles for each waypoint segment
    road_color = tuple(c / 255 for c in CITYSCAPES_PALETTE[1])
    for wp in w.get_map().generate_waypoints(2.0):
        loc = wp.transform.location
        if abs(loc.x - cx) > range_m + 5 or abs(loc.y - cy) > range_m + 5:
            continue
        ax.scatter(loc.x, loc.y, c=[road_color], s=20, marker="s", linewidths=0)

    # Vehicles
    car_color = tuple(c / 255 for c in CITYSCAPES_PALETTE[14])
    ego_color = (1.0, 1.0, 1.0)
    for actor in w.get_actors().filter("vehicle.*"):
        t = actor.get_transform()
        x, y = t.location.x, t.location.y
        if abs(x - cx) > range_m or abs(y - cy) > range_m:
            continue
        rect = patches.Rectangle(
            (x - 2.3, y - 1.0), 4.6, 2.0,
            angle=t.rotation.yaw,
            rotation_point=(x, y),
            facecolor=ego_color if actor.id == actor_id else car_color,
            edgecolor="none",
        )
        ax.add_patch(rect)

    # Walkers
    ped_color = tuple(c / 255 for c in CITYSCAPES_PALETTE[12])
    for actor in w.get_actors().filter("walker.*"):
        t = actor.get_transform()
        x, y = t.location.x, t.location.y
        if abs(x - cx) > range_m or abs(y - cy) > range_m:
            continue
        ax.scatter(x, y, c=[ped_color], s=40, marker="o")

    ax.set_xlim(cx - range_m, cx + range_m)
    ax.set_ylim(cy - range_m, cy + range_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"BEV semantic — actor {actor_id}, range {range_m}m")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=ax.get_facecolor())
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def compare_seg_with_truth(actor_id: int, width: int = 800, height: int = 600) -> Image:
    """Render predicted-vs-ground-truth side-by-side: front semantic camera (left)
    paired with the BEV semantic ground-truth (right). Useful for quick
    qualitative checks of a perception model's outputs against truth.
    """
    sem = capture_sensor(actor_id=actor_id, sensor="semantic", width=width, height=height)
    bev = render_bev_segmentation(actor_id=actor_id)
    # decode both PNGs and stitch horizontally
    left = np.array(PILImage.open(io.BytesIO(sem.data)).convert("RGB"))
    right = np.array(PILImage.open(io.BytesIO(bev.data)).convert("RGB"))
    # match heights
    target_h = max(left.shape[0], right.shape[0])
    def _pad_to_h(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])
    left = _pad_to_h(left, target_h)
    right = _pad_to_h(right, target_h)
    montage = np.hstack([left, right])
    return _png(montage)


# ====================================================================
# v1.5+ — deeper 3D analysis & export
# ====================================================================

@mcp.tool()
def export_point_cloud(
    actor_id: int,
    format: str = "ply",
    output_path: str | None = None,
    semantic: bool = False,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
) -> dict[str, Any]:
    """Capture a single lidar sweep and write it to disk in a standard format.

    Formats:
        - "ply"  (ASCII, opens in CloudCompare / MeshLab / Open3D)
        - "pcd"  (Point Cloud Library; ASCII)
        - "npy"  (numpy raw, fastest re-load for ML pipelines)
        - "bin"  (KITTI raw float32 — x,y,z,intensity per point)

    For semantic=True, the file also carries a uint32 class tag per point.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")

    if semantic:
        xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
        intensities = np.zeros(len(xyz), dtype=np.float32)
        pts4 = np.hstack([xyz.astype(np.float32),
                          intensities.reshape(-1, 1)])
    else:
        pts4, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
        tags = None

    out = Path(output_path) if output_path else (
        Path(tempfile.gettempdir()) / "carla-mcp-clouds" / f"actor{actor_id}_{int(time.time())}.{format}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        if semantic:
            np.savez(out.with_suffix(".npz"), xyz=pts4[:, :3], tags=tags)
            out = out.with_suffix(".npz")
        else:
            np.save(out, pts4)
    elif format == "bin":
        pts4.astype(np.float32).tofile(out)
    elif format == "ply":
        with out.open("w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts4)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float intensity\n")
            if semantic:
                f.write("property uchar class\n")
            f.write("end_header\n")
            for i in range(len(pts4)):
                line = f"{pts4[i,0]:.4f} {pts4[i,1]:.4f} {pts4[i,2]:.4f} {pts4[i,3]:.4f}"
                if semantic:
                    line += f" {int(tags[i]) % 256}"
                f.write(line + "\n")
    elif format == "pcd":
        header = (
            f"# .PCD v0.7 - Point Cloud Data\nVERSION 0.7\n"
            f"FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            f"WIDTH {len(pts4)}\nHEIGHT 1\n"
            f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts4)}\nDATA ascii\n"
        )
        with out.open("w", encoding="utf-8") as f:
            f.write(header)
            for i in range(len(pts4)):
                f.write(f"{pts4[i,0]:.4f} {pts4[i,1]:.4f} {pts4[i,2]:.4f} {pts4[i,3]:.4f}\n")
    else:
        raise ValueError("format must be ply | pcd | npy | bin")

    return {
        "format": format,
        "n_points": int(len(pts4)),
        "semantic": semantic,
        "path": str(out),
        "size_bytes": out.stat().st_size,
    }


@mcp.tool()
def compute_lidar_stats(
    actor_id: int,
    channels: int = 32,
    range_m: float = 80.0,
    points_per_second: int = 200_000,
) -> dict[str, Any]:
    """Compute statistics about a lidar sweep: hit rate, point density, height
    distribution, max effective range, ring uniformity.

    Useful for sanity-checking sensor params before a long capture run.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
    xyz = pts[:, :3]
    if not len(xyz):
        return {"n_points": 0}

    ranges = np.linalg.norm(xyz[:, :2], axis=1)
    z = xyz[:, 2]

    # Approximate ring index by elevation angle bucket (assumes vertical fov split into channels)
    elev = np.arctan2(z, np.linalg.norm(xyz[:, :2], axis=1))
    elev_bins = np.linspace(elev.min(), elev.max(), channels + 1)
    ring = np.clip(np.digitize(elev, elev_bins) - 1, 0, channels - 1)
    counts_per_ring = np.bincount(ring, minlength=channels).tolist()

    return {
        "n_points": int(len(xyz)),
        "range_max_m": float(ranges.max()),
        "range_p99_m": float(np.percentile(ranges, 99)),
        "range_median_m": float(np.median(ranges)),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_p10": float(np.percentile(z, 10)),
        "z_p90": float(np.percentile(z, 90)),
        "approx_points_per_ring": counts_per_ring,
        "min_ring_count": int(min(counts_per_ring)),
        "max_ring_count": int(max(counts_per_ring)),
        "ring_uniformity": float(min(counts_per_ring) / max(1, max(counts_per_ring))),
    }


@mcp.tool()
def voxelize(
    actor_id: int,
    voxel_size: float = 0.5,
    range_m: float = 50.0,
    z_range: list[float] | None = None,
    channels: int = 64,
    points_per_second: int = 600_000,
) -> dict[str, Any]:
    """Discretize a lidar sweep into a 3D voxel grid (sparse occupancy).

    Returns the list of occupied voxel coordinates (i,j,k) plus density stats.
    Useful as input to occupancy-grid models (BEVFusion, OccNet) or for
    ground-truth supervision.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
    xyz = pts[:, :3]

    z_lo, z_hi = (z_range or [-2.0, 4.0])
    mask = (
        (np.abs(xyz[:, 0]) < range_m)
        & (np.abs(xyz[:, 1]) < range_m)
        & (xyz[:, 2] > z_lo)
        & (xyz[:, 2] < z_hi)
    )
    xyz = xyz[mask]
    if not len(xyz):
        return {"voxels": 0, "shape": [0, 0, 0]}

    nx = int(2 * range_m / voxel_size)
    ny = nx
    nz = int((z_hi - z_lo) / voxel_size)
    ix = ((xyz[:, 0] + range_m) / voxel_size).astype(np.int32)
    iy = ((xyz[:, 1] + range_m) / voxel_size).astype(np.int32)
    iz = ((xyz[:, 2] - z_lo) / voxel_size).astype(np.int32)
    keys = ix * (ny * nz) + iy * nz + iz
    occ_keys, counts = np.unique(keys, return_counts=True)
    occ_i = (occ_keys // (ny * nz)).astype(int)
    occ_j = ((occ_keys % (ny * nz)) // nz).astype(int)
    occ_k = (occ_keys % nz).astype(int)

    return {
        "voxel_size": voxel_size,
        "shape": [nx, ny, nz],
        "n_voxels_occupied": int(len(occ_keys)),
        "occupancy_ratio": float(len(occ_keys) / (nx * ny * nz)),
        "max_points_in_a_voxel": int(counts.max()),
        "mean_points_in_a_voxel": float(counts.mean()),
        # Truncate to first 5000 voxels to keep response sane
        "occupied_voxels_truncated": [
            [int(i), int(j), int(k), int(c)]
            for i, j, k, c in zip(occ_i[:5000], occ_j[:5000], occ_k[:5000], counts[:5000])
        ],
    }


@mcp.tool()
def ground_plane_segment(
    actor_id: int,
    distance_threshold: float = 0.2,
    iterations: int = 200,
    channels: int = 64,
    range_m: float = 60.0,
    points_per_second: int = 400_000,
) -> dict[str, Any]:
    """RANSAC ground-plane fit on the lidar sweep.

    Returns plane coefficients (ax+by+cz+d=0), inlier ratio, and a BEV PNG
    showing ground (gray) vs. obstacles (orange) so Claude can sanity-check.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
    xyz = pts[:, :3]
    if len(xyz) < 100:
        raise RuntimeError("Too few points for RANSAC fit.")

    rng = np.random.default_rng(seed=0)
    best_inliers = np.zeros(len(xyz), dtype=bool)
    best_plane = None
    best_count = 0
    for _ in range(int(iterations)):
        idx = rng.choice(len(xyz), 3, replace=False)
        p1, p2, p3 = xyz[idx]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -float(n @ p1)
        dist = np.abs(xyz @ n + d)
        inliers = dist < float(distance_threshold)
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_plane = (float(n[0]), float(n[1]), float(n[2]), d)

    # Build a BEV showing ground vs. obstacles
    colors = np.where(
        best_inliers[:, None],
        np.array([130, 130, 130], dtype=np.uint8),
        np.array([240, 130, 30], dtype=np.uint8),
    )
    bev = _bev_from_points(xyz, colors=colors, range_m=range_m, resolution=0.2)
    return {
        "plane_abcd": best_plane,
        "inlier_ratio": float(best_count / len(xyz)),
        "n_ground": int(best_count),
        "n_obstacle": int(len(xyz) - best_count),
        "image": _png(bev),
    }


@mcp.tool()
def lidar_to_camera_overlay(
    actor_id: int,
    width: int = 1280,
    height: int = 720,
    fov: float = 90.0,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
) -> Image:
    """Project a lidar sweep onto an RGB camera frame, color points by depth.

    Pinhole projection assumes the camera sits at (x=1.6, z=1.7) in the
    actor frame and the lidar at (z=2.5). Useful for visual sanity check
    of lidar-to-camera calibration.
    """
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    # Capture both
    rgb = _capture_one_camera(
        parent, "sensor.camera.rgb", carla.ColorConverter.Raw,
        width, height, fov, carla.Transform(carla.Location(x=1.6, z=1.7)),
    )
    pts, _, _ = _capture_one_lidar(parent, False, channels, range_m, points_per_second)
    xyz = pts[:, :3]
    if not len(xyz):
        return _png(rgb)

    # Lidar in actor frame at z=2.5, camera at (1.6, 0, 1.7).
    # Translate lidar points into camera frame:
    cam_offset = np.array([1.6, 0.0, 1.7 - 2.5])  # camera position relative to lidar
    pts_cam = xyz - cam_offset
    # Camera convention in CARLA: x forward, y right, z up. Pinhole: project (y, -z) over x.
    fwd = pts_cam[:, 0]
    keep = fwd > 0.5
    pts_cam = pts_cam[keep]
    fwd = fwd[keep]
    if not len(pts_cam):
        return _png(rgb)
    f = width / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
    u = (f * pts_cam[:, 1] / fwd + width / 2).astype(np.int32)
    v = (-f * pts_cam[:, 2] / fwd + height / 2).astype(np.int32)
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, fwd = u[mask], v[mask], fwd[mask]

    # Color by depth (closer = warmer)
    import matplotlib.cm as cm
    norm = np.clip(fwd / range_m, 0, 1)
    colors = (cm.get_cmap("plasma")(1 - norm)[:, :3] * 255).astype(np.uint8)

    out = rgb.copy()
    for r in range(-1, 2):
        for c in range(-1, 2):
            uu = np.clip(u + c, 0, width - 1)
            vv = np.clip(v + r, 0, height - 1)
            out[vv, uu] = colors
    return _png(out)


@mcp.tool()
def iou_3d(actor_a_id: int, actor_b_id: int) -> dict[str, Any]:
    """Approximate 3D IoU between two actors' bounding boxes (AABB after yaw rotation).

    Useful as a sanity check or for tracking evaluation. Note: this is the
    axis-aligned IoU after rotating both boxes to a common frame, which is
    a conservative under-estimate for boxes with very different yaws.
    """
    w = _world()
    a = w.get_actor(actor_a_id)
    b = w.get_actor(actor_b_id)
    if a is None or b is None:
        raise ValueError("One of the actors was not found.")

    def world_bbox_extents(actor):
        bb = actor.bounding_box
        t = actor.get_transform()
        center = t.transform(bb.location)
        half = np.array([bb.extent.x, bb.extent.y, bb.extent.z])
        return np.array([center.x, center.y, center.z]), half

    ca, ha = world_bbox_extents(a)
    cb, hb = world_bbox_extents(b)
    lo_a, hi_a = ca - ha, ca + ha
    lo_b, hi_b = cb - hb, cb + hb
    inter = np.maximum(0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    inter_vol = float(np.prod(inter))
    vol_a = float(np.prod(2 * ha))
    vol_b = float(np.prod(2 * hb))
    union = vol_a + vol_b - inter_vol
    iou = inter_vol / union if union > 0 else 0.0
    return {
        "actor_a": actor_a_id,
        "actor_b": actor_b_id,
        "iou_3d_aabb": float(iou),
        "volume_a": vol_a,
        "volume_b": vol_b,
        "intersection_volume": inter_vol,
    }


@mcp.tool()
def export_dataset(
    rig_id: str,
    n_frames: int = 10,
    delta_seconds: float = 0.1,
    output_dir: str | None = None,
    label_format: str = "kitti",
) -> dict[str, Any]:
    """Produce a packaged ML dataset folder from a sensor rig.

    Combines:
      - synchronized capture (capture_synchronized) — paired sensor frames
      - calibration export (export_calibration) — intrinsics + extrinsics
      - 3D bbox labels (auto_label) — one label file per frame

    Output layout (KITTI-style):
        <output_dir>/
            calib/000000.txt
            image_2/000000.png       (front_rgb)
            velodyne/000000.bin      (lidar as KITTI raw float32)
            label_2/000000.txt       (KITTI bbox labels)
            manifest.json
    """
    rigs = _load_rigs()
    if rig_id not in rigs:
        raise ValueError(f"Unknown rig_id {rig_id!r}")
    parent_id = rigs[rig_id]["parent"]

    out_root = Path(output_dir) if output_dir else (
        Path(tempfile.gettempdir()) / "carla-mcp-datasets" / f"{rig_id}_{int(time.time())}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("calib", "image_2", "velodyne", "label_2"):
        (out_root / sub).mkdir(exist_ok=True)

    # First grab sensor data via capture_synchronized to a temp folder
    sync_res = capture_synchronized(
        rig_id=rig_id, n_frames=n_frames, delta_seconds=delta_seconds,
        output_dir=str(out_root / "_raw"),
    )
    raw_root = Path(sync_res["output_dir"])

    # Calibration once (rigs are static during a run)
    calib = export_calibration(rig_id=rig_id, format="json")

    manifest: dict[str, Any] = {
        "rig_id": rig_id,
        "n_frames": 0,
        "label_format": label_format,
        "calibration": calib,
        "frames": [],
    }
    for i, frame in enumerate(sync_res["manifest"] if False else  # hack: reuse the file
                              json.loads(Path(sync_res["manifest"]).read_text())["frames"]):
        idx = f"{i:06d}"
        # Find the front_rgb and lidar files
        rgb_src = frame["files"].get("front_rgb")
        lidar_src = next((p for k, p in frame["files"].items() if "lidar" in k), None)
        if rgb_src and Path(rgb_src).exists():
            shutil_copy(rgb_src, out_root / "image_2" / f"{idx}.png")
        if lidar_src and Path(lidar_src).exists():
            # PLY → numpy → bin (KITTI float32)
            xyz = _read_ply_xyz(lidar_src)
            inten = np.zeros(len(xyz), dtype=np.float32)
            np.hstack([xyz.astype(np.float32), inten.reshape(-1, 1)]).tofile(
                out_root / "velodyne" / f"{idx}.bin"
            )
        # Calibration per frame (KITTI repeats)
        (out_root / "calib" / f"{idx}.txt").write_text(_kitti_calib_text(calib))
        # Labels
        labels = extract_3d_bboxes(observer_actor_id=parent_id, format=label_format,
                                   max_distance=120.0)
        body = labels.get("labels") if label_format == "kitti" else json.dumps(labels.get("boxes", []))
        (out_root / "label_2" / f"{idx}.txt").write_text(body or "")
        manifest["frames"].append({"index": i, "name": idx})

    manifest["n_frames"] = len(manifest["frames"])
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "output_dir": str(out_root),
        "n_frames": manifest["n_frames"],
        "format": label_format,
        "manifest": str(out_root / "manifest.json"),
    }


def shutil_copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


def _read_ply_xyz(path: str) -> np.ndarray:
    """Minimal ASCII-PLY reader for x y z (skips header)."""
    out = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        in_body = False
        for line in f:
            if not in_body:
                if line.startswith("end_header"):
                    in_body = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 3), dtype=np.float32)


def _kitti_calib_text(calib: dict[str, Any]) -> str:
    """Build a minimal KITTI calib.txt from our JSON calibration."""
    cam = next(
        (s for s in calib["sensors"] if s.get("intrinsic")),
        None,
    )
    if cam is None:
        return "# no camera in rig\n"
    K = np.array(cam["intrinsic"]["K"])
    P2 = np.hstack([K, np.zeros((3, 1))])
    return (
        "P0: " + " ".join(f"{v:.4f}" for v in P2.flatten()) + "\n"
        "P1: " + " ".join(f"{v:.4f}" for v in P2.flatten()) + "\n"
        "P2: " + " ".join(f"{v:.4f}" for v in P2.flatten()) + "\n"
        "P3: " + " ".join(f"{v:.4f}" for v in P2.flatten()) + "\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )


# ====================================================================
# v1.0 — sensor rigs + datasets
# ====================================================================

_SENSOR_PRESETS = {
    "minimal": [
        ("front_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            {"image_size_x": 800, "image_size_y": 450, "fov": 90}),
        ("lidar", "sensor.lidar.ray_cast",
            carla.Transform(carla.Location(z=2.5)),
            {"channels": 32, "range": 50.0, "points_per_second": 100000,
             "rotation_frequency": 10.0}),
    ],
    "perception": [
        ("front_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("front_depth", "sensor.camera.depth",
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("front_semantic", "sensor.camera.semantic_segmentation",
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("lidar", "sensor.lidar.ray_cast_semantic",
            carla.Transform(carla.Location(z=2.5)),
            {"channels": 64, "range": 80.0, "points_per_second": 600000,
             "rotation_frequency": 10.0}),
        ("imu", "sensor.other.imu",
            carla.Transform(carla.Location(z=1.0)), {}),
        ("gnss", "sensor.other.gnss",
            carla.Transform(carla.Location(z=1.0)), {}),
    ],
    "full": [
        ("front_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("rear_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(x=-2.0, z=1.7), carla.Rotation(yaw=180)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("left_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(y=-1.0, z=1.7), carla.Rotation(yaw=-90)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("right_rgb", "sensor.camera.rgb",
            carla.Transform(carla.Location(y=1.0, z=1.7), carla.Rotation(yaw=90)),
            {"image_size_x": 1280, "image_size_y": 720, "fov": 90}),
        ("lidar", "sensor.lidar.ray_cast_semantic",
            carla.Transform(carla.Location(z=2.5)),
            {"channels": 64, "range": 80.0, "points_per_second": 600000,
             "rotation_frequency": 10.0}),
        ("imu", "sensor.other.imu",
            carla.Transform(carla.Location(z=1.0)), {}),
        ("gnss", "sensor.other.gnss",
            carla.Transform(carla.Location(z=1.0)), {}),
        ("radar", "sensor.other.radar",
            carla.Transform(carla.Location(x=2.0, z=1.0)),
            {"range": 60.0, "horizontal_fov": 30.0, "vertical_fov": 10.0}),
    ],
}


@mcp.tool()
def attach_sensor_rig(actor_id: int, preset: str = "minimal") -> dict[str, Any]:
    """Attach a multi-sensor rig to an actor in one call.

    Returns a `rig_id` you can pass to capture_synchronized / render_sensor_montage
    / export_calibration. Sensors are tracked and cleaned up by reset_world.

    Args:
        actor_id: Vehicle to attach to.
        preset: "minimal" (front cam + 32-ch lidar)
                | "perception" (front cam+depth+sem + 64-ch sem-lidar + IMU + GNSS)
                | "full" (4 cams + 64-ch sem-lidar + IMU + GNSS + radar)
    """
    if preset not in _SENSOR_PRESETS:
        raise ValueError(f"preset must be one of {list(_SENSOR_PRESETS)}")
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    w = _world()
    bplib = w.get_blueprint_library()

    rig_id = f"rig_{actor_id}_{preset}_{int(time.time())}"
    sensors: list[dict[str, Any]] = []
    for name, bp_id, transform, attrs in _SENSOR_PRESETS[preset]:
        bp = bplib.find(bp_id)
        for k, v in attrs.items():
            bp.set_attribute(k, str(v))
        sensor = w.spawn_actor(bp, transform, attach_to=parent)
        _track(sensor)
        sensors.append({
            "name": name,
            "id": sensor.id,
            "type": bp_id,
            "transform": [transform.location.x, transform.location.y, transform.location.z,
                          transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            "attributes": attrs,
        })

    rigs = _load_rigs()
    rigs[rig_id] = {"parent": actor_id, "preset": preset, "sensors": sensors}
    _save_rigs(rigs)
    return {"rig_id": rig_id, "n_sensors": len(sensors), "sensors": [s["name"] for s in sensors]}


@mcp.tool()
def export_calibration(rig_id: str, format: str = "json") -> dict[str, Any]:
    """Export intrinsics + extrinsics for all sensors in a rig.

    For cameras: K matrix from FOV + image dims.
    For all sensors: 4x4 transform from parent (extrinsic).
    """
    rigs = _load_rigs()
    if rig_id not in rigs:
        raise ValueError(f"Unknown rig_id {rig_id!r}. Available: {list(rigs)}")
    rig = rigs[rig_id]
    out: list[dict[str, Any]] = []
    for s in rig["sensors"]:
        attrs = s["attributes"]
        loc = s["transform"][:3]
        rot = s["transform"][3:]
        entry: dict[str, Any] = {
            "name": s["name"],
            "type": s["type"],
            "extrinsic": {
                "location_xyz": loc,
                "rotation_pyr_deg": rot,
            },
        }
        if s["type"].startswith("sensor.camera."):
            W = int(attrs.get("image_size_x", 800))
            H = int(attrs.get("image_size_y", 600))
            fov = float(attrs.get("fov", 90.0))
            f = W / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
            entry["intrinsic"] = {
                "image_size_xy": [W, H],
                "fov_deg": fov,
                "K": [[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
            }
        out.append(entry)
    if format == "json":
        return {"rig_id": rig_id, "format": "json", "sensors": out}
    if format == "kitti":
        # KITTI calib.txt-ish: P2 + Tr_velo_to_cam (camera + lidar only)
        cam = next((s for s in out if s["type"].startswith("sensor.camera.")), None)
        lid = next((s for s in out if s["type"].startswith("sensor.lidar.")), None)
        if cam is None or lid is None:
            raise RuntimeError("KITTI export needs at least one camera and one lidar in the rig.")
        K = np.array(cam["intrinsic"]["K"])
        P2 = np.hstack([K, np.zeros((3, 1))])
        return {
            "rig_id": rig_id,
            "format": "kitti",
            "P2": P2.flatten().tolist(),
            "comment": "Tr_velo_to_cam computation requires sensor-frame conventions; see ROADMAP.",
        }
    raise ValueError("format must be 'json' or 'kitti'")


@mcp.tool()
def render_sensor_montage(rig_id: str, max_cells: int = 6) -> Image:
    """Capture all camera sensors in the rig and lay them out in a grid PNG.

    Lidar / radar / IMU / GNSS are reported as overlay text. RGB / depth /
    semantic / instance / optical_flow / dvs camera sensors are all rendered.
    """
    rigs = _load_rigs()
    if rig_id not in rigs:
        raise ValueError(f"Unknown rig_id {rig_id!r}.")
    parent_id = rigs[rig_id]["parent"]
    parent = _world().get_actor(parent_id)
    if parent is None:
        raise ValueError(f"Rig parent actor {parent_id} no longer exists.")

    cells: list[tuple[str, np.ndarray]] = []
    for s in rigs[rig_id]["sensors"]:
        if not s["type"].startswith("sensor.camera."):
            continue
        if len(cells) >= max_cells:
            break
        sensor_kind = {
            "sensor.camera.rgb": "rgb",
            "sensor.camera.depth": "depth",
            "sensor.camera.semantic_segmentation": "semantic",
            "sensor.camera.instance_segmentation": "instance_segmentation",
            "sensor.camera.optical_flow": "optical_flow",
            "sensor.camera.dvs": "dvs",
        }.get(s["type"], "rgb")
        # capture_sensor lazy-spawns its own camera at cam_transform; use the rig's
        # parent and sensor kind, but at modest resolution for the montage
        png = capture_sensor(actor_id=parent_id, sensor=sensor_kind, width=480, height=270)
        rgb = np.array(PILImage.open(io.BytesIO(png.data)).convert("RGB"))
        cells.append((s["name"], rgb))

    if not cells:
        raise RuntimeError("Rig has no camera sensors to montage.")

    # Lay out 2 columns x ceil(N/2) rows
    cols = 2 if len(cells) > 1 else 1
    rows = (len(cells) + cols - 1) // cols
    cell_h, cell_w = cells[0][1].shape[:2]
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, (_, img) in enumerate(cells):
        r, c = divmod(i, cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img

    # Annotate names
    pil = PILImage.fromarray(canvas)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    for i, (name, _) in enumerate(cells):
        r, c = divmod(i, cols)
        draw.rectangle([(c * cell_w + 4, r * cell_h + 4),
                        (c * cell_w + 8 + 9 * len(name), r * cell_h + 22)],
                       fill=(0, 0, 0, 180))
        draw.text((c * cell_w + 8, r * cell_h + 6), name, fill=(255, 255, 255))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def capture_synchronized(
    rig_id: str,
    n_frames: int = 5,
    delta_seconds: float = 0.1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Drive the world in synchronous mode and capture paired sensor frames.

    Writes each frame's outputs to `output_dir` (default: %TEMP%/carla-mcp-runs/<rig_id>/),
    plus a `manifest.json` listing every produced file.

    NOTE: temporarily switches the world to synchronous_mode=True with the
    given fixed delta. Other clients (e.g. manual_control) will pause while
    this is running. We restore the previous mode before returning.
    """
    rigs = _load_rigs()
    if rig_id not in rigs:
        raise ValueError(f"Unknown rig_id {rig_id!r}.")
    parent_id = rigs[rig_id]["parent"]
    parent = _world().get_actor(parent_id)
    if parent is None:
        raise ValueError(f"Rig parent actor {parent_id} no longer exists.")

    out_root = Path(output_dir) if output_dir else (
        Path(tempfile.gettempdir()) / "carla-mcp-runs" / rig_id
    )
    out_root.mkdir(parents=True, exist_ok=True)

    w = _world()
    prev_settings = w.get_settings()
    sync_settings = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=delta_seconds,
        no_rendering_mode=prev_settings.no_rendering_mode,
    )
    w.apply_settings(sync_settings)
    tm = _get_client().get_trafficmanager()
    tm.set_synchronous_mode(True)

    manifest: dict[str, Any] = {
        "rig_id": rig_id,
        "parent_actor_id": parent_id,
        "delta_seconds": delta_seconds,
        "frames": [],
    }
    try:
        sensors_meta = rigs[rig_id]["sensors"]
        # Pre-spawn dedicated listeners for each sensor in the rig
        # (the rig's existing actors already have their own Carla sensors;
        #  we just attach a queue to each.)
        queues: dict[int, queue.Queue] = {}
        actors: dict[int, carla.Actor] = {}
        for s in sensors_meta:
            actor = w.get_actor(s["id"])
            if actor is None:
                continue
            actors[s["id"]] = actor
            q: queue.Queue = queue.Queue()
            actor.listen(q.put)
            queues[s["id"]] = q

        for f in range(n_frames):
            w.tick()
            frame_dir = out_root / f"frame_{f:04d}"
            frame_dir.mkdir(exist_ok=True)
            frame_files: dict[str, str] = {}
            for s in sensors_meta:
                if s["id"] not in queues:
                    continue
                try:
                    data = queues[s["id"]].get(timeout=2.0)
                except queue.Empty:
                    continue
                if s["type"].startswith("sensor.camera."):
                    if s["type"] == "sensor.camera.depth":
                        data.convert(carla.ColorConverter.LogarithmicDepth)
                    elif s["type"] == "sensor.camera.semantic_segmentation":
                        data.convert(carla.ColorConverter.CityScapesPalette)
                    fp = frame_dir / f"{s['name']}.png"
                    data.save_to_disk(str(fp))
                    frame_files[s["name"]] = str(fp)
                elif s["type"].startswith("sensor.lidar."):
                    fp = frame_dir / f"{s['name']}.ply"
                    data.save_to_disk(str(fp))
                    frame_files[s["name"]] = str(fp)
                else:
                    fp = frame_dir / f"{s['name']}.json"
                    fp.write_text(json.dumps({"frame": data.frame, "type": s["type"]}))
                    frame_files[s["name"]] = str(fp)
            manifest["frames"].append({
                "index": f,
                "sim_time": w.get_snapshot().timestamp.elapsed_seconds,
                "files": frame_files,
            })
        # stop listeners
        for s in sensors_meta:
            actor = actors.get(s["id"])
            if actor is not None:
                try:
                    actor.stop()
                except Exception:
                    pass
    finally:
        w.apply_settings(prev_settings)
        tm.set_synchronous_mode(False)

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "rig_id": rig_id,
        "n_frames": len(manifest["frames"]),
        "output_dir": str(out_root),
        "manifest": str(manifest_path),
    }


@mcp.tool()
def auto_label(observer_actor_id: int, format: str = "kitti", output_path: str | None = None) -> dict[str, Any]:
    """Generate ground-truth 3D bbox labels for the current frame, in the requested format,
    and write to disk if `output_path` is given. Otherwise returns the label text inline."""
    res = extract_3d_bboxes(
        observer_actor_id=observer_actor_id,
        format=format,
        max_distance=120.0,
    )
    if output_path is not None:
        body = res.get("labels") or json.dumps(res.get("boxes", []))
        Path(output_path).write_text(body, encoding="utf-8")
        return {"format": format, "n_boxes": res["n_boxes"], "written_to": output_path}
    return res


# ====================================================================
# v2.1 — perception evaluation suite (uses ObjIdx + ObjTag from semantic lidar)
# ====================================================================

@mcp.tool()
def extract_actor_points(
    observer_actor_id: int,
    target_actor_id: int,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
    resolution: float = 0.2,
) -> dict[str, Any]:
    """Capture a semantic-lidar sweep and return only points belonging to one actor.

    Uses the per-point `ObjIdx` field that CARLA's semantic lidar carries —
    no detection model required. Useful for per-object density analysis,
    occlusion checks, or producing instance-mask ground truth.

    Returns counts, centroid, extent, dominant class, and a small BEV PNG of
    just those points highlighted against the rest of the sweep.
    """
    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")
    xyz, tags, obj_idx = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    if obj_idx is None:
        raise RuntimeError("semantic lidar capture missing ObjIdx")

    mask = obj_idx == target_actor_id
    n = int(mask.sum())
    if n == 0:
        return {
            "actor_id": target_actor_id,
            "n_points": 0,
            "image": None,
            "note": "no lidar returns hit this actor (occluded, out of range, or wrong id)",
        }

    sel = xyz[mask]
    centroid = sel.mean(axis=0)
    extent = (sel.max(axis=0) - sel.min(axis=0)) / 2
    tags_sel = tags[mask] if tags is not None else None
    dom_tag = int(np.bincount(tags_sel).argmax()) if tags_sel is not None and len(tags_sel) else 0

    # BEV: target actor's points red, everything else gray
    colors = np.full((len(xyz), 3), 90, dtype=np.uint8)
    colors[mask] = (230, 60, 60)
    bev = _bev_from_points(xyz, colors=colors, range_m=range_m, resolution=resolution)
    return {
        "actor_id": target_actor_id,
        "n_points": n,
        "centroid_local": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "extent_local": [float(extent[0]), float(extent[1]), float(extent[2])],
        "dominant_class_id": dom_tag,
        "dominant_class_name": SEMANTIC_NAMES[dom_tag] if dom_tag < len(SEMANTIC_NAMES) else "?",
        "image": _png(bev),
    }


@mcp.tool()
def actor_visibility(
    observer_actor_id: int,
    target_actor_id: int | None = None,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
) -> dict[str, Any]:
    """Per-actor lidar hit count + visibility classification.

    Returns each visible actor with: point count, dominant class, and a
    visibility label:
        - "high"      ≥ 50 points
        - "medium"    10–49 points
        - "low"       1–9 points
        - "occluded"  not in observer's actor list but 0 hits

    If `target_actor_id` is given, returns just that one. Useful for filtering
    autolabel pipelines (drop labels with <5 hits = noisy supervision).
    """
    w = _world()
    parent = w.get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")
    xyz, tags, obj_idx = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    if obj_idx is None:
        raise RuntimeError("semantic lidar capture missing ObjIdx")

    counts: dict[int, int] = {}
    dom_tag: dict[int, int] = {}
    if len(obj_idx):
        unique_ids, idx_inverse = np.unique(obj_idx, return_inverse=True)
        for i, aid in enumerate(unique_ids):
            sel = idx_inverse == i
            counts[int(aid)] = int(sel.sum())
            if tags is not None:
                local_tags = tags[sel]
                dom_tag[int(aid)] = int(np.bincount(local_tags).argmax())

    def classify(n: int) -> str:
        if n >= 50:
            return "high"
        if n >= 10:
            return "medium"
        if n >= 1:
            return "low"
        return "occluded"

    if target_actor_id is not None:
        n = counts.get(target_actor_id, 0)
        d = dom_tag.get(target_actor_id, 0)
        return {
            "actor_id": target_actor_id,
            "n_points": n,
            "visibility": classify(n),
            "dominant_class": SEMANTIC_NAMES[d] if d < len(SEMANTIC_NAMES) else "?",
        }

    rows = []
    for aid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        d = dom_tag.get(aid, 0)
        rows.append({
            "actor_id": aid,
            "n_points": n,
            "visibility": classify(n),
            "dominant_class": SEMANTIC_NAMES[d] if d < len(SEMANTIC_NAMES) else "?",
        })
    return {"n_actors_visible": len(rows), "actors": rows}


@mcp.tool()
def class_conditional_bev(
    observer_actor_id: int,
    classes: list[str],
    channels: int = 64,
    range_m: float = 60.0,
    points_per_second: int = 400_000,
    resolution: float = 0.2,
) -> Image:
    """BEV showing only the specified semantic classes (everything else dimmed).

    Example: ["Car", "Pedestrian"] produces a dynamic-actor map.
              ["Road", "RoadLine", "Sidewalk"] produces a drivable-surface map.

    Class names are case-insensitive. See SEMANTIC_NAMES for the full list.
    """
    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")
    keep = _resolve_class_filter(classes)
    if keep is None or not keep:
        raise ValueError("Pass a non-empty list of class names.")
    xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    in_keep = np.isin(tags, list(keep))
    # Keep-class points get their CityScapes color; others go dim gray
    colors = np.full((len(xyz), 3), 35, dtype=np.uint8)
    if in_keep.any():
        clipped = np.clip(tags[in_keep], 0, len(CITYSCAPES_PALETTE) - 1)
        colors[in_keep] = CITYSCAPES_PALETTE[clipped]
    bev = _bev_from_points(xyz, colors=colors, range_m=range_m, resolution=resolution)
    return _png(bev)


@mcp.tool()
def evaluate_clustering(
    observer_actor_id: int,
    eps: float = 0.7,
    min_samples: int = 6,
    channels: int = 64,
    range_m: float = 60.0,
    points_per_second: int = 400_000,
    iou_threshold: float = 0.3,
    resolution: float = 0.2,
) -> dict[str, Any]:
    """Run DBSCAN on a semantic-lidar sweep and compare against `ObjIdx` ground truth.

    For every true instance and every predicted cluster, compute point-set IoU
    (|gt ∩ pred| / |gt ∪ pred|), then greedily match by descending IoU.
    Reports precision, recall, mean matched IoU, and per-actor results.
    Renders a BEV PNG: matched points green, false-positive cluster points red,
    missed ground-truth orange.

    This is the unique selling point — real ML perception evaluation entirely
    inside CARLA, with sensor-grounded ground truth and no detection model.
    """
    from sklearn.cluster import DBSCAN

    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")
    xyz, _, obj_idx = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    if obj_idx is None or len(xyz) == 0:
        raise RuntimeError("no semantic-lidar returns")

    # Drop ground returns from clustering (rough heuristic: z < -1.4 from a 2.5m sensor)
    above_ground = xyz[:, 2] > -1.4
    xyz_obj = xyz[above_ground]
    obj_idx_obj = obj_idx[above_ground]

    if len(xyz_obj) < min_samples:
        return {"matches": 0, "precision": 0.0, "recall": 0.0, "mean_iou": 0.0,
                "image": None, "note": "too few non-ground points"}

    # Predict
    pred = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(xyz_obj[:, :2]).labels_
    # Filter out: noise (-1) for predictions, and the road-bg objidx 0 for truth
    true_ids = np.unique(obj_idx_obj[obj_idx_obj > 0])
    pred_ids = np.unique(pred[pred >= 0])

    # Build IoU matrix (true × pred). Cap matrix size for speed.
    iou_mat = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
    for i, t in enumerate(true_ids):
        t_mask = obj_idx_obj == t
        for j, p in enumerate(pred_ids):
            p_mask = pred == p
            inter = int((t_mask & p_mask).sum())
            if inter == 0:
                continue
            union = int((t_mask | p_mask).sum())
            iou_mat[i, j] = inter / max(1, union)

    # Greedy matching: pick max IoU pair, remove row+col, repeat.
    matches: list[dict[str, Any]] = []
    used_t, used_p = set(), set()
    flat = [(iou_mat[i, j], i, j) for i in range(len(true_ids)) for j in range(len(pred_ids))]
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if iou < iou_threshold:
            break
        if i in used_t or j in used_p:
            continue
        used_t.add(i); used_p.add(j)
        matches.append({
            "true_actor_id": int(true_ids[i]),
            "pred_cluster_id": int(pred_ids[j]),
            "iou": float(iou),
        })

    precision = len(matches) / max(1, len(pred_ids))
    recall = len(matches) / max(1, len(true_ids))
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0

    # Visualization: green = matched, red = FP cluster, orange = missed truth
    colors = np.full((len(xyz_obj), 3), 70, dtype=np.uint8)
    matched_t_ids = {true_ids[m_i] for m_i in used_t}
    matched_p_ids = {pred_ids[m_j] for m_j in used_p}
    for t in true_ids:
        if t in matched_t_ids:
            colors[obj_idx_obj == t] = (60, 200, 90)  # green
        else:
            colors[obj_idx_obj == t] = (240, 160, 30)  # orange = missed
    for p in pred_ids:
        if p not in matched_p_ids:
            colors[pred == p] = (220, 50, 50)  # red = FP
    bev = _bev_from_points(xyz_obj, colors=colors, range_m=range_m, resolution=resolution)

    return {
        "n_true": int(len(true_ids)),
        "n_pred": int(len(pred_ids)),
        "matches": matches,
        "n_matched": len(matches),
        "precision": float(precision),
        "recall": float(recall),
        "mean_iou": mean_iou,
        "iou_threshold": float(iou_threshold),
        "image": _png(bev),
    }


@mcp.tool()
def lidar_to_camera_segmentation(
    observer_actor_id: int,
    width: int = 1280,
    height: int = 720,
    fov: float = 90.0,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
    blend_alpha: float = 0.55,
) -> Image:
    """Project semantic-lidar onto an RGB camera frame, color each pixel by class.

    Sparse but pixel-accurate semantic ground truth from sensor calibration
    alone (no model needed). Each projected point is drawn at its (u, v) in
    its CityScapes class color, blended over the RGB at `blend_alpha`.

    Useful as: cheap semantic-seg supervision, calibration sanity check,
    or a visualization for explaining lidar coverage.
    """
    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")

    rgb = _capture_one_camera(
        parent, "sensor.camera.rgb", carla.ColorConverter.Raw,
        width, height, fov, carla.Transform(carla.Location(x=1.6, z=1.7)),
    )
    xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    if not len(xyz):
        return _png(rgb)

    # Lidar→camera projection (camera at x=1.6, z=1.7; lidar at z=2.5)
    cam_offset = np.array([1.6, 0.0, 1.7 - 2.5])
    pts_cam = xyz - cam_offset
    fwd = pts_cam[:, 0]
    keep = fwd > 0.5
    pts_cam, fwd, tags = pts_cam[keep], fwd[keep], tags[keep]
    if not len(pts_cam):
        return _png(rgb)

    f = width / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
    u = (f * pts_cam[:, 1] / fwd + width / 2).astype(np.int32)
    v = (-f * pts_cam[:, 2] / fwd + height / 2).astype(np.int32)
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, tags = u[mask], v[mask], tags[mask]
    cls_idx = np.clip(tags, 0, len(CITYSCAPES_PALETTE) - 1)
    point_colors = CITYSCAPES_PALETTE[cls_idx]

    out = rgb.copy().astype(np.float32)
    a = float(blend_alpha)
    for r in range(-1, 2):
        for c in range(-1, 2):
            uu = np.clip(u + c, 0, width - 1)
            vv = np.clip(v + r, 0, height - 1)
            out[vv, uu] = (1.0 - a) * out[vv, uu] + a * point_colors
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return _png(out_u8)


@mcp.tool()
def check_sensor_consistency(
    observer_actor_id: int,
    width: int = 800,
    height: int = 450,
    fov: float = 90.0,
    channels: int = 64,
    range_m: float = 80.0,
    points_per_second: int = 600_000,
) -> dict[str, Any]:
    """Cross-validate semantic camera vs semantic lidar.

    Captures a semantic camera image (each pixel's RED channel encodes the
    CARLA class id) and a semantic lidar sweep, projects each lidar point
    into the camera, and compares the lidar's class to the camera's class
    at the same pixel. Reports per-class agreement % plus an overlay PNG
    that highlights disagreements in red.

    A sub-50% agreement on common classes usually means the lidar mounting
    transform drifted from the camera's, or one of the sensors fell behind
    in async mode.
    """
    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")

    # Capture semantic camera in RAW so the red channel is the class id.
    cam_classid_bgra = _capture_one_camera(
        parent, "sensor.camera.semantic_segmentation",
        carla.ColorConverter.Raw, width, height, fov,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
    )
    # _capture_one_camera returns RGB after BGRA→RGB swap → red = class_id is now red[..., 0].
    cam_classes = cam_classid_bgra[..., 0].astype(np.int32)

    xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    if not len(xyz):
        return {"agreement": None, "note": "no lidar points captured"}

    cam_offset = np.array([1.6, 0.0, 1.7 - 2.5])
    pts_cam = xyz - cam_offset
    fwd = pts_cam[:, 0]
    keep = fwd > 0.5
    pts_cam, fwd, tags = pts_cam[keep], fwd[keep], tags[keep]
    if not len(pts_cam):
        return {"agreement": None, "note": "all lidar points behind the camera"}

    f = width / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
    u = (f * pts_cam[:, 1] / fwd + width / 2).astype(np.int32)
    v = (-f * pts_cam[:, 2] / fwd + height / 2).astype(np.int32)
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, tags = u[in_frame], v[in_frame], tags[in_frame]
    cam_at_pt = cam_classes[v, u]
    agree = (cam_at_pt == tags)

    # Per-class breakdown
    per_class: dict[str, dict[str, int]] = {}
    for cls_id in np.unique(tags):
        sel = tags == cls_id
        name = SEMANTIC_NAMES[int(cls_id)] if cls_id < len(SEMANTIC_NAMES) else f"id_{cls_id}"
        per_class[name] = {
            "n_points": int(sel.sum()),
            "n_agree": int(agree[sel].sum()),
            "agreement_pct": float(100.0 * agree[sel].sum() / max(1, sel.sum())),
        }

    # Overlay PNG: cam (colorized via palette) with disagreements highlighted red
    cam_rgb = CITYSCAPES_PALETTE[np.clip(cam_classes, 0, len(CITYSCAPES_PALETTE) - 1)]
    cam_rgb = cam_rgb.astype(np.uint8)
    disagree = ~agree
    if disagree.any():
        for r in range(-1, 2):
            for c in range(-1, 2):
                uu = np.clip(u[disagree] + c, 0, width - 1)
                vv = np.clip(v[disagree] + r, 0, height - 1)
                cam_rgb[vv, uu] = (240, 50, 50)

    return {
        "n_points_in_frame": int(len(u)),
        "agreement_overall_pct": float(100.0 * agree.sum() / max(1, len(agree))),
        "per_class": per_class,
        "image": _png(cam_rgb),
    }


@mcp.tool()
def semantic_voxelize(
    observer_actor_id: int,
    voxel_size: float = 0.5,
    range_m: float = 50.0,
    z_range: list[float] | None = None,
    classes: list[str] | None = None,
    channels: int = 64,
    points_per_second: int = 600_000,
) -> dict[str, Any]:
    """Voxelize a semantic-lidar sweep; each occupied voxel keeps the dominant class.

    Direct ground truth for occupancy networks (OccNet, BEVFusion, …).
    Optionally filter to a subset of class names before voxelizing.
    """
    parent = _world().get_actor(observer_actor_id)
    if parent is None:
        raise ValueError(f"Observer actor {observer_actor_id} not found.")
    xyz, tags, _ = _capture_one_lidar(parent, True, channels, range_m, points_per_second)
    keep = _resolve_class_filter(classes)
    if keep is not None and keep:
        sel = np.isin(tags, list(keep))
        xyz, tags = xyz[sel], tags[sel]

    z_lo, z_hi = (z_range or [-2.0, 4.0])
    in_box = (
        (np.abs(xyz[:, 0]) < range_m)
        & (np.abs(xyz[:, 1]) < range_m)
        & (xyz[:, 2] > z_lo)
        & (xyz[:, 2] < z_hi)
    )
    xyz = xyz[in_box]
    tags = tags[in_box]
    if not len(xyz):
        return {"voxels": 0, "shape": [0, 0, 0]}

    nx = int(2 * range_m / voxel_size)
    ny = nx
    nz = int((z_hi - z_lo) / voxel_size)
    ix = ((xyz[:, 0] + range_m) / voxel_size).astype(np.int32)
    iy = ((xyz[:, 1] + range_m) / voxel_size).astype(np.int32)
    iz = ((xyz[:, 2] - z_lo) / voxel_size).astype(np.int32)
    keys = ix * (ny * nz) + iy * nz + iz

    # Per-voxel dominant class via groupby
    order = np.argsort(keys)
    keys_sorted, tags_sorted = keys[order], tags[order]
    boundaries = np.concatenate(([0], np.where(np.diff(keys_sorted) != 0)[0] + 1, [len(keys_sorted)]))
    occupied = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        k = int(keys_sorted[s])
        cls_counts = np.bincount(tags_sorted[s:e])
        dom = int(cls_counts.argmax())
        occupied.append((k, int(e - s), dom))

    return {
        "voxel_size": voxel_size,
        "shape": [nx, ny, nz],
        "n_voxels_occupied": len(occupied),
        "occupancy_ratio": float(len(occupied) / (nx * ny * nz)),
        "class_filter": classes,
        "occupied_voxels_truncated": [
            [int(k // (ny * nz)), int((k % (ny * nz)) // nz), int(k % nz),
             int(cnt), int(dom_cls), SEMANTIC_NAMES[dom_cls] if dom_cls < len(SEMANTIC_NAMES) else "?"]
            for k, cnt, dom_cls in occupied[:5000]
        ],
    }


# ====================================================================
# v2.0 — visualization & scenarios
# ====================================================================

@mcp.tool()
def render_trajectory(
    actor_id: int,
    duration_s: float = 10.0,
    sample_hz: float = 5.0,
    range_m: float = 100.0,
) -> Image:
    """Sample an actor's pose for `duration_s` seconds and overlay the path on a top-down map.

    BLOCKING: this tool sleeps for `duration_s` seconds before returning.
    """
    duration_s = max(1.0, min(float(duration_s), 60.0))
    sample_hz = max(0.5, min(float(sample_hz), 20.0))
    parent = _world().get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")

    n = int(duration_s * sample_hz)
    dt = 1.0 / sample_hz
    path = []
    for _ in range(n):
        t = parent.get_transform()
        path.append((t.location.x, t.location.y, t.rotation.yaw))
        time.sleep(dt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    w = _world()
    cx = path[len(path) // 2][0] if path else 0.0
    cy = path[len(path) // 2][1] if path else 0.0
    # roads
    wps = w.get_map().generate_waypoints(2.0)
    xs = [wp.transform.location.x for wp in wps]
    ys = [wp.transform.location.y for wp in wps]
    ax.scatter(xs, ys, s=1, c="#cccccc", alpha=0.5)
    # path
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, "-", c="#e63946", lw=2)
        ax.scatter(px[0], py[0], c="#264653", s=80, label="start", zorder=5)
        ax.scatter(px[-1], py[-1], c="#2a9d8f", s=80, label="end", zorder=5)
    ax.set_xlim(cx - range_m, cx + range_m)
    ax.set_ylim(cy - range_m, cy + range_m)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"Trajectory of actor {actor_id} — {duration_s:.0f}s @ {sample_hz} Hz")
    ax.grid(True, alpha=0.25, linestyle=":")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def spawn_adversarial(
    behavior: str = "cut_in",
    target_actor_id: int | None = None,
    distance_m: float = 15.0,
) -> dict[str, Any]:
    """Spawn a vehicle or walker with pre-configured adversarial behavior.

    behaviors:
      - "cut_in"        : autopilot vehicle that ignores leading-distance and lane-changes aggressively
      - "sudden_brake"  : autopilot vehicle that decelerates randomly
      - "jaywalker"     : pedestrian walking across the road in front of the target
    """
    w = _world()
    target = w.get_actor(target_actor_id) if target_actor_id else None
    spawn_t = None
    if target is not None:
        t = target.get_transform()
        fwd = t.get_forward_vector()
        spawn_t = carla.Transform(
            carla.Location(
                x=t.location.x + fwd.x * distance_m,
                y=t.location.y + fwd.y * distance_m,
                z=t.location.z + 0.5,
            ),
            carla.Rotation(yaw=t.rotation.yaw + (90 if behavior == "jaywalker" else 0)),
        )

    if behavior in ("cut_in", "sudden_brake"):
        bp = random.choice(list(w.get_blueprint_library().filter("vehicle.tesla.model3")))
        # Retry with several distances + map spawn fallbacks if the spot is occupied.
        candidates: list[carla.Transform] = []
        if target is not None:
            t = target.get_transform()
            fwd = t.get_forward_vector()
            for d in (distance_m, distance_m * 0.7, distance_m * 1.4,
                      distance_m * 1.8, distance_m * 0.5):
                candidates.append(carla.Transform(
                    carla.Location(
                        x=t.location.x + fwd.x * d,
                        y=t.location.y + fwd.y * d,
                        z=t.location.z + 0.5,
                    ),
                    carla.Rotation(yaw=t.rotation.yaw),
                ))
        spawn_points = w.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        candidates.extend(spawn_points[:8])  # up to 8 random fallbacks
        vehicle = None
        for sp in candidates:
            vehicle = w.try_spawn_actor(bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Could not find a clear spawn position for adversarial vehicle.")
        _track(vehicle)
        tm = _get_client().get_trafficmanager()
        vehicle.set_autopilot(True, tm.get_port())
        if behavior == "cut_in":
            tm.distance_to_leading_vehicle(vehicle, 0.5)
            tm.auto_lane_change(vehicle, True)
            tm.vehicle_percentage_speed_difference(vehicle, -30.0)  # 30% faster
            tm.ignore_lights_percentage(vehicle, 50.0)
        elif behavior == "sudden_brake":
            tm.distance_to_leading_vehicle(vehicle, 0.5)
            tm.vehicle_percentage_speed_difference(vehicle, 80.0)  # 80% slower
        return {"behavior": behavior, "actor_id": vehicle.id}

    if behavior == "jaywalker":
        walker_bps = w.get_blueprint_library().filter("walker.pedestrian.*")
        sp = spawn_t or carla.Transform(w.get_random_location_from_navigation())
        walker = w.try_spawn_actor(random.choice(walker_bps), sp)
        if walker is None:
            raise RuntimeError("Could not spawn jaywalker at the given location.")
        _track(walker)
        controller_bp = w.get_blueprint_library().find("controller.ai.walker")
        controller = w.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            (w.tick() if w.get_settings().synchronous_mode else w.wait_for_tick())
            controller.start()
            # walk perpendicular to vehicle heading
            target_loc = sp.location
            if target is not None:
                # cross perpendicular to target's forward direction
                fwd = target.get_transform().get_forward_vector()
                target_loc = carla.Location(
                    x=sp.location.x - fwd.y * 8.0,
                    y=sp.location.y + fwd.x * 8.0,
                    z=sp.location.z,
                )
            controller.go_to_location(target_loc)
            controller.set_max_speed(2.5)  # running pace
            _track(controller)
        return {"behavior": "jaywalker", "actor_id": walker.id}

    raise ValueError(f"Unknown behavior {behavior!r}. Use cut_in | sudden_brake | jaywalker.")


@mcp.tool()
def compile_scenario(spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize a structured scenario spec by chaining existing tools.

    Spec schema (all keys optional except where noted):
      {
        "map":           "Town03",                       # str (required to switch town)
        "weather":       {"preset":"HardRainNoon", ...}, # passed to set_weather
        "ego":           {"model":"vehicle.tesla.model3",
                          "spawn_index": 0,
                          "autopilot": true},
        "traffic":       {"n_vehicles": 30, "n_pedestrians": 5},
        "adversarials":  [{"behavior":"cut_in", "target":"ego"}],
        "spectator":     "ego" | {"actor_id": 42} | {"x":0,"y":0,"z":40},
        "wait_s":        2.0,
      }

    Returns a list of step results, including the ego's actor_id and any spawned ids.
    """
    steps: list[dict[str, Any]] = []
    ego_id: int | None = None

    if "map" in spec:
        steps.append({"load_map": load_map(spec["map"])})
    if "weather" in spec:
        steps.append({"set_weather": set_weather(**spec["weather"])})
    if "ego" in spec:
        e = spec["ego"]
        res = spawn_vehicle(
            model=e.get("model", "vehicle.tesla.model3"),
            spawn_point_index=e.get("spawn_index"),
            autopilot=e.get("autopilot", False),
        )
        ego_id = res["actor_id"]
        steps.append({"spawn_ego": res})
    if "traffic" in spec:
        steps.append({"spawn_traffic": spawn_traffic(**spec["traffic"])})
    for adv in spec.get("adversarials", []) or []:
        a_args = dict(adv)
        if a_args.get("target") == "ego":
            a_args["target_actor_id"] = ego_id
            a_args.pop("target", None)
        steps.append({"spawn_adversarial": spawn_adversarial(**a_args)})
    if "spectator" in spec:
        sp = spec["spectator"]
        if sp == "ego" and ego_id is not None:
            steps.append({"set_spectator": set_spectator(actor_id=ego_id)})
        elif isinstance(sp, dict):
            steps.append({"set_spectator": set_spectator(**sp)})
    if spec.get("wait_s"):
        steps.append({"wait": wait(seconds=float(spec["wait_s"]))})

    return {"ego_id": ego_id, "steps": steps}


@mcp.tool()
def scenario_sweep(
    base_spec: dict[str, Any],
    vary: dict[str, list[Any]],
    capture_after_s: float = 3.0,
) -> dict[str, Any]:
    """Run `compile_scenario(base_spec)` multiple times, varying parameters.

    Args:
        base_spec: Same shape as compile_scenario's spec.
        vary: dotted-key map of values to sweep, e.g.
              {"weather.preset": ["ClearNoon", "HardRainNoon", "MidRainSunset"]}
              or {"traffic.n_vehicles": [10, 30, 60]}.
              Currently sweeps each key independently (Cartesian over keys is left
              as a v2.5 enhancement).
        capture_after_s: After materializing each scenario, wait this many seconds,
              then snapshot world_status() into the result.

    Returns a list of (params, world_status) per run. Between runs, all spawned
    actors from this server's tracking set are destroyed.
    """
    runs: list[dict[str, Any]] = []
    for key, values in vary.items():
        for v in values:
            spec = json.loads(json.dumps(base_spec))  # deep copy
            cur = spec
            parts = key.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
            compiled = compile_scenario(spec)
            time.sleep(float(capture_after_s))
            ws = world_status()
            runs.append({
                "varied": {key: v},
                "ego_id": compiled.get("ego_id"),
                "world_status": {
                    "map": ws["map"],
                    "actors": ws["actors"],
                    "weather": ws["weather"],
                },
            })
            reset_world(also_clear_tracked=True)
    return {"n_runs": len(runs), "runs": runs}


@mcp.tool()
def failure_snapshot(
    actor_id: int,
    watch_seconds: float = 30.0,
    sensor: str = "rgb",
) -> dict[str, Any]:
    """Watch an actor for collisions / lane invasions for up to `watch_seconds`,
    return the first event + a frame captured at the moment of impact.

    BLOCKING up to watch_seconds. Spawns a transient collision sensor and a
    lane-invasion sensor on the actor; both are destroyed before return.
    """
    w = _world()
    parent = w.get_actor(actor_id)
    if parent is None:
        raise ValueError(f"Actor {actor_id} not found.")
    bplib = w.get_blueprint_library()

    events: list[dict[str, Any]] = []
    coll_bp = bplib.find("sensor.other.collision")
    lane_bp = bplib.find("sensor.other.lane_invasion")
    coll = w.spawn_actor(coll_bp, carla.Transform(), attach_to=parent)
    lane = w.spawn_actor(lane_bp, carla.Transform(), attach_to=parent)

    def on_collision(event):
        events.append({
            "kind": "collision",
            "frame": event.frame,
            "other": event.other_actor.type_id if event.other_actor else None,
            "intensity": (event.normal_impulse.x ** 2
                          + event.normal_impulse.y ** 2
                          + event.normal_impulse.z ** 2) ** 0.5,
        })

    def on_lane(event):
        events.append({
            "kind": "lane_invasion",
            "frame": event.frame,
            "markings": [str(m.type) for m in event.crossed_lane_markings],
        })

    coll.listen(on_collision)
    lane.listen(on_lane)
    deadline = time.time() + max(1.0, min(float(watch_seconds), 120.0))
    try:
        while time.time() < deadline and not events:
            time.sleep(0.2)
    finally:
        coll.stop()
        lane.stop()
        coll.destroy()
        lane.destroy()

    if not events:
        return {"events": [], "image": None, "note": "no collision/lane invasion observed"}

    snap = capture_sensor(actor_id=actor_id, sensor=sensor, width=800, height=450)
    return {"events": events, "first_event": events[0], "image": snap}


@mcp.tool()
def run_openscenario(
    file: str,
    scenario_runner_root: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run an OpenSCENARIO 1.x file via CARLA's `scenario_runner` (must be cloned separately).

    Args:
        file: Path to the .xosc file.
        scenario_runner_root: Path to a clone of carla-simulator/scenario_runner.
            If None, uses the env var SCENARIO_RUNNER_ROOT.
        timeout_s: Kill the runner after this many seconds.

    Returns the runner's final status and stdout/stderr (truncated).
    """
    import subprocess

    root = scenario_runner_root or os.environ.get("SCENARIO_RUNNER_ROOT")
    if not root or not Path(root).exists():
        raise RuntimeError(
            "Provide scenario_runner_root or set SCENARIO_RUNNER_ROOT to a "
            "clone of https://github.com/carla-simulator/scenario_runner. "
            "ScenarioRunner is intentionally NOT bundled with carla-mcp to "
            "avoid a heavy dep."
        )
    runner = Path(root) / "scenario_runner.py"
    if not runner.exists():
        raise RuntimeError(f"scenario_runner.py not found at {runner}")
    cmd = [sys.executable, str(runner), "--openscenario", file, "--reloadWorld"]
    proc = subprocess.run(
        cmd, cwd=root, capture_output=True, text=True, timeout=float(timeout_s),
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _build_http_app(token: str):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not token:
                return await call_next(request)
            header = request.headers.get("authorization", "")
            expected = f"Bearer {token}"
            if header != expected:
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="carla-mcp"'},
                )
            return await call_next(request)

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuth)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="carla-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.environ.get("CARLA_MCP_TRANSPORT", "stdio"),
        help="stdio for Claude Desktop / Claude Code; http for claude.ai connectors.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CARLA_MCP_HTTP_HOST", "127.0.0.1"),
        help="HTTP bind address (default 127.0.0.1; set to 0.0.0.0 to expose).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CARLA_MCP_HTTP_PORT", "8765")),
        help="HTTP port (default 8765).",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
        return

    import uvicorn

    token = os.environ.get("CARLA_MCP_TOKEN", "").strip()
    app = _build_http_app(token)

    if not token:
        print(
            "WARNING: CARLA_MCP_TOKEN is not set. HTTP server has NO authentication.\n"
            "         Anyone who reaches the URL can drive your CARLA simulator.\n"
            "         Set CARLA_MCP_TOKEN to a long random string for any non-trivial use.",
            file=sys.stderr,
        )
    else:
        print(
            "carla-mcp HTTP requires header: Authorization: Bearer <CARLA_MCP_TOKEN>",
            file=sys.stderr,
        )
    print(
        f"carla-mcp listening at http://{args.host}:{args.port}/mcp",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
