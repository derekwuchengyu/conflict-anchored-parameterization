"""
config -> .xosc driver.

Stage A (retrieval-scenarios/convert2yaml.convert_to_yaml) runs IN-PROCESS so our
monkeypatched sampler is active, with cwd temporarily switched into retrieval-scenarios
(its output paths are relative). Stage B (hcis_scenario_generation/main.py) runs as a
SUBPROCESS with a per-config base config so each config's .xosc lands in its own dir
(the Scenario_name is config-independent, so configs would otherwise overwrite).

Nothing in the existing repos is modified.
"""
from __future__ import annotations
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import yaml as _yaml

from . import paths
from . import config as C
from . import sampling

paths.add_import_paths()

_carla_maps: dict[str, object] = {}


@contextlib.contextmanager
def _chdir(d: Path):
    prev = os.getcwd()
    os.chdir(str(d))
    try:
        yield
    finally:
        os.chdir(prev)


def _get_map(dataset: str):
    """Build & cache the carla.Map for a dataset (needed by Stage A route planning)."""
    if dataset not in _carla_maps:
        import carla
        d = paths.dataset_of(dataset)
        xodr_content = Path(d["xodr"]).read_text()
        _carla_maps[dataset] = carla.Map(d["xodr_name"], xodr_content)
    return _carla_maps[dataset]


# convert_to_yaml adjusts start_frame internally (to the ego's real first frame), so
# we don't reconstruct the name — we capture the exact path it writes via export.
_captured: dict[str, object] = {}
_orig_export = None


def _install_export_capture():
    global _orig_export
    import convert2yaml
    if _orig_export is None:
        _orig_export = convert2yaml.export_config_to_yaml

    def wrapped(config, output_path="config_example.yaml"):
        # export_config_to_yaml is reused for spec/stop files (dict configs); capture
        # only the main scenario YAML (dataclass config written under yaml/).
        name = getattr(config, "Scenario_name", None)
        op = str(output_path).replace("\\", "/")
        if name is not None and "/yaml/" in f"/{op}" and "output_scenario" not in op and "yaml" not in _captured:
            _captured["yaml"] = output_path
            _captured["name"] = name
        return _orig_export(config, output_path)

    convert2yaml.export_config_to_yaml = wrapped


def _uninstall_export_capture():
    global _orig_export
    if _orig_export is not None:
        import convert2yaml
        convert2yaml.export_config_to_yaml = _orig_export
        _orig_export = None


# ─────────────────────────────────────────────────────────────────────────────
# strategy post-hook: flatten the speed event for strategy A (constant velocity)
# ─────────────────────────────────────────────────────────────────────────────
_orig_build = None


def _install_speed_hook():
    """Adjust the speed event per the forced speed model (config): 'const' = constant start
    speed over the whole duration; 'real' = ramp timed to reach the forced anchor (critical)
    frame — so uniform@pet vs uniform@min_dis differ in the exported xosc."""
    mode = C.get_forced_speed()
    if mode is None:
        return
    import convert2yaml
    global _orig_build
    if _orig_build is None:
        _orig_build = convert2yaml.build_agent_from_trajectory
    # speed event frame: the decoupled speed frame when set, else the CP anchor
    forced = C.get_forced_speed_frame()
    if forced is None:
        forced = C.get_forced_anchor_frame()
    speed_val = C.get_forced_speed_value()

    def wrapped(*a, **k):
        agent = _orig_build(*a, **k)
        start_frame = int(a[3]) if len(a) > 3 else 0
        fps = int(k.get("frame_rate", 30))
        for act in getattr(agent, "Acts", []) or []:
            evs = getattr(act, "Events", []) or []
            pos_dur = next((getattr(e, "Dynamic_duration", None) for e in evs
                            if getattr(e, "Type", None) == "position"), None)
            for ev in evs:
                if getattr(ev, "Type", None) != "speed":
                    continue
                if mode == "const":
                    ev.End = getattr(agent, "Start_speed", ev.End)   # constant velocity
                    if pos_dur is not None:
                        ev.Dynamic_duration = pos_dur
                elif mode == "real" and forced is not None:
                    ev.Dynamic_duration = max(0.1, (forced - start_frame) / fps)  # reach crit frame
                    if speed_val is not None:
                        ev.End = speed_val   # target = the agent's REAL speed at the critical frame
        return agent

    convert2yaml.build_agent_from_trajectory = wrapped


def _uninstall_speed_hook():
    global _orig_build
    if _orig_build is not None:
        import convert2yaml
        convert2yaml.build_agent_from_trajectory = _orig_build
        _orig_build = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage A
# ─────────────────────────────────────────────────────────────────────────────
def _anchor_frame_for(dataset, ego_id, param_agent, start_frame, end_frame, anchor):
    """The absolute frame for the active anchor (pet / min_dist / traj_cross), computed from
    the (ego, actor) trajectories — overrides convert_to_yaml's internal frame selection."""
    from . import parampath as PP
    petf, mdf, tcf = PP.anchor_frames(dataset, int(ego_id), int(param_agent),
                                      int(start_frame), int(end_frame))
    f = {"pet": petf, "min_dist": mdf, "traj_cross": tcf, "min_speed": None}.get(anchor, petf)
    return f if f is not None else (petf if petf is not None else mdf)


def _anchor_speed_for(dataset, param_agent, anchor_frame):
    """Real speed (km/h) of the param agent AT the anchor (critical) frame — used as the speed
    event's target End speed so anchored methods differ by END SPEED, not only by duration
    (uniform@pet vs uniform@min_dis stay distinct even when both durations hit the 0.1s floor)."""
    if anchor_frame is None:
        return None
    from . import parampath as PP
    try:
        sub = PP._traj(dataset).xs(int(param_agent), level="track_id").sort_index()
    except Exception:
        return None
    if sub.empty or "velocity" not in sub.columns:
        return None
    idx = int(sub.index.get_indexer([int(anchor_frame)], method="nearest")[0])
    if idx < 0:
        return None
    return float(sub.iloc[idx]["velocity"]) * 3.6


def stage_a(dataset, ego_id, agent_ids, replayer_ids, start_frame, end_frame, label,
            sample_cfg: C.SampleConfig | None = None,
            strategy: C.StrategyConfig | None = None,
            force_anchor: bool = False,
            speed_model: str | None = None,
            speed_anchor: str | None = None) -> tuple[str, Path]:
    """Run convert_to_yaml with the sampler override. Returns (name, yaml_path).
    force_anchor=True injects the active config's anchor frame (so traj_cross / min_dist take
    effect in the real pipeline); A/B/C strategies leave it False to keep legacy behavior.
    speed_anchor ("pet"/"min_dist"/"traj_cross") decouples the SPEED event's critical frame
    from the sampling (CP position) anchor; None = coupled (speed frame = CP anchor frame)."""
    import convert2yaml

    if strategy is not None:
        C.set_active_strategy(strategy)
    elif sample_cfg is not None:
        C.set_active_sample_config(sample_cfg)

    active = C.get_active_sample_config()
    if active.is_full_replay:
        raise NotImplementedError(
            "full-replay strategy not yet wired through convert_to_yaml "
            "(challenge agent would need to move to replayer_ids); it is the "
            "geometric upper bound (path_dev ~ 0) and can be added later.")

    carla_map = _get_map(dataset)
    dinfo = paths.dataset_of(dataset)
    data_id = dinfo["data_id"]

    if force_anchor:
        try:
            C.set_forced_anchor_frame(_anchor_frame_for(
                dataset, ego_id, list(agent_ids)[0], start_frame, end_frame, active.anchor))
        except Exception:
            C.set_forced_anchor_frame(None)

    mode = speed_model
    if mode is None and strategy is not None and strategy.constant_speed:
        mode = "const"
    C.set_forced_speed(mode)
    # target End speed = real speed at the SPEED frame (only for anchored 'real' methods; B/C have
    # no forced anchor -> value stays None -> keep the pipeline's default End = window max speed).
    # speed_anchor decouples that frame from the CP anchor; None keeps them coupled.
    if mode == "real":
        sframe = C.get_forced_anchor_frame()
        if speed_anchor is not None:
            try:
                sframe = _anchor_frame_for(dataset, ego_id, list(agent_ids)[0],
                                           start_frame, end_frame, speed_anchor)
            except Exception:
                pass
        C.set_forced_speed_frame(sframe)
        C.set_forced_speed_value(_anchor_speed_for(dataset, list(agent_ids)[0], sframe))
    else:
        C.set_forced_speed_frame(None)
        C.set_forced_speed_value(None)

    _captured.clear()
    with _chdir(paths.RETRIEVAL):
        sampling.install()
        _install_speed_hook()
        _install_export_capture()
        try:
            convert2yaml.convert_to_yaml(
                carla_map, dataset, data_id, int(ego_id),
                [int(a) for a in agent_ids], [int(r) for r in replayer_ids],
                int(start_frame), int(end_frame), int(label),
                debug_mode=True,
            )
        finally:
            _uninstall_export_capture()
            _uninstall_speed_hook()
            sampling.uninstall()
            C.set_forced_anchor_frame(None)
            C.set_forced_speed(None)
            C.set_forced_speed_value(None)
            C.set_forced_speed_frame(None)

    if "yaml" not in _captured:
        raise RuntimeError("Stage A did not export any config "
                           "(convert_to_yaml failed / demoted the agent to replay)")
    name = str(_captured["name"])
    # captured path is relative to RETRIEVAL (e.g. ./yaml/debug/NAME.yaml)
    yaml_path = (paths.RETRIEVAL / str(_captured["yaml"])).resolve()
    if not yaml_path.exists():
        raise RuntimeError(f"Stage A exported {name} but {yaml_path} is missing")
    return name, yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage B
# ─────────────────────────────────────────────────────────────────────────────
def stage_b(dataset, name, yaml_path: Path, out_dir: Path, timeout: int = 300) -> Path | None:
    """Run Stage B (subprocess) with save_paths redirected to out_dir. Returns .xosc path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = "sr_inD.yaml" if dataset == "inD" else "sr_HetroD.yaml"
    base_cfg = _yaml.safe_load((paths.SCENGEN / "config" / "base" / base_name).read_text())
    base_cfg["save_paths"] = [str(out_dir) + "/"]
    tmp_base = out_dir / "_base.yaml"
    tmp_base.write_text(_yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True))

    cmd = [sys.executable, "main.py", "-b", str(tmp_base), "-c", str(yaml_path)]
    proc = subprocess.run(cmd, cwd=str(paths.SCENGEN),
                          capture_output=True, text=True, timeout=timeout)
    xosc = out_dir / f"{name}.xosc"
    if xosc.exists():
        return xosc
    tail = (proc.stdout or "")[-800:] + "\n--- stderr ---\n" + (proc.stderr or "")[-800:]
    raise RuntimeError(f"Stage B produced no {xosc.name} (rc={proc.returncode}).\n{tail}")


# ─────────────────────────────────────────────────────────────────────────────
# One-shot
# ─────────────────────────────────────────────────────────────────────────────
def generate(dataset, ego_id, agent_ids, replayer_ids, start_frame, end_frame, label,
             out_dir: Path,
             sample_cfg: C.SampleConfig | None = None,
             strategy: C.StrategyConfig | None = None,
             force_anchor: bool = False,
             speed_model: str | None = None,
             speed_anchor: str | None = None) -> Path | None:
    """config -> .xosc. Returns the path to the generated concrete scenario."""
    name, yaml_path = stage_a(dataset, ego_id, agent_ids, replayer_ids,
                              start_frame, end_frame, label, sample_cfg=sample_cfg,
                              strategy=strategy, force_anchor=force_anchor,
                              speed_model=speed_model, speed_anchor=speed_anchor)
    return stage_b(dataset, name, yaml_path, out_dir)
