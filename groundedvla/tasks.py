"""Selected tasks (plan Section 3.1). Instructions are read from the BDDL files at runtime."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    key: str
    suite: str
    bddl_name: str
    max_steps: int


TASKS = {
    "T1": TaskSpec("T1", "libero_spatial",
                   "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", 220),
    "T2": TaskSpec("T2", "libero_spatial",
                   "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", 220),
    "T3": TaskSpec("T3", "libero_goal", "put_the_bowl_on_the_stove", 300),
    # Same scene as T3, different destination: only the instruction tells them apart.
    "T4": TaskSpec("T4", "libero_goal", "put_the_bowl_on_the_plate", 300),
    # Scope extension (Nov 13): drawers and a bottle. Lighter evaluation (configs/eval_ood.yaml).
    "T5": TaskSpec("T5", "libero_spatial",
                   "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", 220),
    "T6": TaskSpec("T6", "libero_goal", "open_the_top_drawer_and_put_the_bowl_inside", 300),
    "T7": TaskSpec("T7", "libero_goal", "put_the_wine_bottle_on_top_of_the_cabinet", 300),
}

# Episode limits per suite (OpenVLA / LIBERO evaluation convention).
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                   "libero_10": 520, "libero_90": 400}

NUM_SETTLE_STEPS = 10   # no-op steps at episode start (A2)
CONTROL_HZ = 20


def resolve_task(name: str) -> TaskSpec:
    """'T1'..'T4', or any LIBERO task as '<suite>:<task_id>' (e.g. 'libero_spatial:4') for quick checks."""
    if name in TASKS:
        return TASKS[name]
    if ":" not in name:
        raise ValueError(f"unknown task {name!r}; use one of {list(TASKS)} or '<suite>:<task_id>'")
    import groundedvla.envs.compat  # noqa: F401  (GL backend before LIBERO imports)
    from libero.libero import benchmark

    suite_name, task_id = name.split(":", 1)
    task = benchmark.get_benchmark_dict()[suite_name]().get_task(int(task_id))
    return TaskSpec(name, suite_name, task.name, SUITE_MAX_STEPS.get(suite_name, 500))
