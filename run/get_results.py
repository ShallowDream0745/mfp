import wandb
import pandas as pd
from pathlib import Path
from typing import List, Dict


def get_column_name(key: str) -> str:
    """
    Convert wandb metric key to CSV column name.
    """
    mapping = {
        "eval/episode_success": "success",
        "eval/episode_reward": "reward",
    }
    if key in mapping:
        return mapping[key]
    # For other keys, use the last part after '/'
    return key.split("/")[-1]


def get_unique_filepath(filepath: Path) -> Path:
    """
    Get a unique filepath by appending -2, -3, etc. if the file exists.

    Args:
        filepath: Original filepath

    Returns:
        Unique filepath (either original or with suffix)
    """
    if not filepath.exists():
        return filepath

    counter = 2
    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent

    while True:
        new_name = f"{stem}-{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def get_result(
    entity: str = "shallowdream0745-thu",
    project: str = "flow",
    base_dir: str = "result_plot/data",
    env_type: str = "myosuite",
    alg: str = "mfp",
):
    """
    Fetch task results from wandb and save to CSV files.

    Args:
        entity: Wandb entity name
        project: Wandb project name
        base_dir: Base directory for saving CSV files
        env_type: Environment type - "myosuite" or "dm_control" (default: "myosuite")
        alg: Algorithm type - "mfp" or "boom" (default: "mfp")
    """
    # Set keys based on env_type
    if env_type == "myosuite":
        selected_keys = ["eval/episode_success", "eval/episode_reward"]
    elif env_type == "dm_control":
        selected_keys = ["eval/episode_reward"]
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'myosuite' or 'dm_control'")

    # Set update_flow filter based on alg
    update_filter = (alg == "mfp")

    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{entity}/{project}")

    # Process runs
    print(f"Processing runs with env_type={env_type}, update_flow={update_filter}...")
    process_runs(
        runs,
        selected_keys=selected_keys,
        env_type_filter=env_type,
        update_filter=update_filter,
        output_dir=Path(base_dir) / env_type / alg,
    )


def process_runs(
    runs: List,
    selected_keys: List[str],
    env_type_filter: str,
    update_filter: bool,
    output_dir: Path,
):
    """
    Process runs and save to CSV files.

    Args:
        runs: List of wandb runs
        selected_keys: List of metric keys to fetch
        env_type_filter: Filter by env_type config ("myosuite" or "dm_control")
        update_filter: Filter by update_flow config (True/False)
        output_dir: Output directory for CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by task name
    task_runs: Dict[str, List] = {}

    for run in runs:
        # Check if run matches filter criteria
        config = run.config

        # Check env_type
        env_type = config.get("env_type")
        if env_type != env_type_filter:
            continue

        # Check seed (> 10)
        seed = config.get("seed")
        if seed is None or seed <= 10:
            continue

        # Check update_flow
        update_flow = config.get("update_flow")
        if update_flow != update_filter:
            continue

        # Get task name
        task = config.get("task", "")

        # additional filter
        if "-q1" in run.name or "-new" in run.name:
            continue

        # Group by task name
        if task not in task_runs:
            task_runs[task] = []
        task_runs[task].append(run)

    # Process each task
    for task_name, runs_list in task_runs.items():
        print(f"  Processing task: {task_name} ({len(runs_list)} runs)")

        for run in runs_list:
            try:
                # Fetch history with selected keys
                history = run.history(keys=selected_keys + ["_step"])

                # Extract data
                seed = run.config.get("seed", 0)
                all_data = []
                for _, row in history.iterrows():
                    data_row = {"step": row["_step"]}
                    for key in selected_keys:
                        col_name = get_column_name(key)
                        data_row[col_name] = row.get(key, None)
                    all_data.append(data_row)

                if not all_data:
                    print(f"    Warning: No data found for run {run.id}")
                    continue

                # Create DataFrame and save
                df = pd.DataFrame(all_data)

                # Reorder columns: step, selected_keys columns
                columns = ["step"] + [get_column_name(key) for key in selected_keys]
                df = df[columns]

                # Sort by step
                df = df.sort_values("step")

                # Save to CSV with format {name}_{seed}.csv
                # Handle duplicate filenames by adding -2, -3, etc.
                base_path = output_dir / f"{task_name}_{seed}.csv"
                output_path = get_unique_filepath(base_path)
                df.to_csv(output_path, index=False)
                print(f"    Saved to {output_path} ({len(df)} rows)")

            except Exception as e:
                print(f"    Warning: Failed to fetch data for run {run.id}: {e}")
                continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch task results from wandb")
    parser.add_argument(
        "--entity",
        default="shallowdream0745-thu",
        help="Wandb entity name",
    )
    parser.add_argument(
        "--project",
        default="flow",
        help="Wandb project name",
    )
    parser.add_argument(
        "--base-dir",
        default="result_plot/data",
        help="Base directory for saving CSV files",
    )
    parser.add_argument(
        "--env-type",
        default="myosuite",
        choices=["myosuite", "dm_control"],
        help="Environment type - 'myosuite' or 'dm_control' (default: myosuite)",
    )
    parser.add_argument(
        "--alg",
        default="boom",
        choices=["mfp", "boom"],
        help="Algorithm type - 'mfp' or 'boom' (default: mfp)",
    )

    args = parser.parse_args()

    get_result(
        entity=args.entity,
        project=args.project,
        base_dir=args.base_dir,
        env_type=args.env_type,
        alg=args.alg,
    )
