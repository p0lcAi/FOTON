from __future__ import annotations
from typing import Optional, Dict
import os

USE_WANDB_DEFAULT = True  # global default; can be overridden per run via config/env

def _read_bool(val: object, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1","true","yes","on"}
    return bool(val)

def maybe_init_wandb(config: Optional[Dict] = None):
    """
    Safe W&B init:
    - Honors (in priority order): config["WANDB_ENABLE"] > env WANDB_ENABLE > USE_WANDB_DEFAULT
    - Respects env or config for project/entity/name/group/job_type/tags/notes
    - Supports modes: online | offline | disabled (via config or env WANDB_MODE)
    - Falls back to offline if online init fails; if that also fails, disables logging
    Returns the run object or None. You can always proceed with training.
    """
    cfg = config or {}

    # Resolve enable flag
    use_wandb = _read_bool(
        cfg.get("WANDB_ENABLE", os.getenv("WANDB_ENABLE", None)),
        default=USE_WANDB_DEFAULT
    )
    if not use_wandb:
        return None

    try:
        import wandb  # import only when needed
    except Exception as e:
        print(f"[W&B] Import failed: {e}. Disabling logging.")
        return None

    # Mode resolution
    mode = (str(cfg.get("WANDB_MODE", os.getenv("WANDB_MODE", "online"))).lower())
    if mode in {"offline", "disabled"}:
        os.environ["WANDB_MODE"] = mode  # respected by wandb.init

    # Collect standard kwargs (env overrides are common in CI)
    kwargs = dict(
        project = cfg.get("WANDB_PROJECT", os.getenv("WANDB_PROJECT", "default")),
        name    = cfg.get("EXP_NAME",     os.getenv("WANDB_NAME",    "run")),
        config  = cfg
    )

    # Optional fields (only set if present)
    entity   = cfg.get("WANDB_ENTITY",   os.getenv("WANDB_ENTITY"))
    group    = cfg.get("WANDB_GROUP",    os.getenv("WANDB_GROUP"))
    job_type = cfg.get("WANDB_JOB_TYPE", os.getenv("WANDB_JOB_TYPE"))
    tags     = cfg.get("WANDB_TAGS")     # expect list[str]
    notes    = cfg.get("WANDB_NOTES")

    if entity:   kwargs["entity"]   = entity
    if group:    kwargs["group"]    = group
    if job_type: kwargs["job_type"] = job_type
    if tags:     kwargs["tags"]     = tags
    if notes:    kwargs["notes"]    = notes

    # Optional: allow anonymous logging (handy for workshops)
    # - Set config["WANDB_ANONYMOUS"]="allow" OR env WANDB_ANONYMOUS=allow
    anonymous = (cfg.get("WANDB_ANONYMOUS", os.getenv("WANDB_ANONYMOUS")))
    if anonymous:
        kwargs["anonymous"] = str(anonymous)

    # Try online/offline init
    try:
        run = wandb.init(**kwargs)
        return run
    except Exception as e:
        print(f"[W&B] Online init failed: {e}. Falling back to offline.")
        try:
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(**kwargs)
            return run
        except Exception as e2:
            print(f"[W&B] Offline init also failed: {e2}. Disabling logging.")
            return None


def maybe_log(metrics: Dict[str, float]):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception as e:
        print(f"[W&B] log failed: {e}")


def maybe_finish():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"[W&B] finish failed: {e}")
