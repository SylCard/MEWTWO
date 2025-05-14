"""
activation_checkpointing.py
───────────────────────────
Greedy activation-checkpoint planner (μ-TWO Algorithm B heuristic)
that works out-of-the-box with sensible defaults.

Defaults
========
• --profile   =  dynamic_profiling.csv      (in cwd)
• --mem-limit =  80 % of CUDA device RAM or 12000 MiB if no CUDA

Output
======
checkpoint_plan.csv  – same columns as input + updated 'to_recompute'
"""

from __future__ import annotations
import argparse, pandas as pd, textwrap, os, torch


# ────────────────────────────────────────────────────────────────────
#  Default helpers
# ────────────────────────────────────────────────────────────────────
def default_profile_path() -> str:
    return "dynamic_profiling.csv"


def default_mem_limit_mb() -> int:
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory  # bytes
        return int(total / 1024**2 * 0.80)                        # 80 %
    return 12000  # fallback 12 GiB


# ────────────────────────────────────────────────────────────────────
#  Data loading / preprocessing
# ────────────────────────────────────────────────────────────────────
def load_profile(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Profile CSV '{csv_path}' not found. "
            "Run fx_profiler.py first or pass --profile <file>."
        )
    df = pd.read_csv(csv_path)
    df = df[df["gtype"] == "forward"].copy()

    needed = ["rank", "run_time_ms", "peak_mem_bytes"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {csv_path}")

    df.sort_values("rank", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # memory_size = change in peak_mem after this op
    prev = df["peak_mem_bytes"].shift().fillna(method="bfill")
    df["memory_size_bytes"] = (df["peak_mem_bytes"] - prev).clip(lower=0).astype(int)

    # ensure column exists
    if "to_recompute" not in df.columns:
        df["to_recompute"] = ""

    return df


# ────────────────────────────────────────────────────────────────────
#  Greedy checkpoint algorithm
# ────────────────────────────────────────────────────────────────────
def apply_checkpoint_policy(
    df: pd.DataFrame, mem_limit_bytes: int
) -> tuple[pd.DataFrame, int, float, int]:
    peak_bytes = df["peak_mem_bytes"].max()
    df["recompute_ratio"] = df["memory_size_bytes"] / df["run_time_ms"].clip(lower=1e-3)

    saved = 0
    extra_ms = 0.0

    for idx, row in df.sort_values("recompute_ratio", ascending=False).iterrows():
        if peak_bytes - saved <= mem_limit_bytes:
            break
        if row["memory_size_bytes"] == 0:
            continue
        df.at[idx, "to_recompute"] = "yes"
        saved += row["memory_size_bytes"]
        extra_ms += row["run_time_ms"]

    final_peak = peak_bytes - saved
    return df, saved, extra_ms, final_peak


# ────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Decide which activations to checkpoint (drop & recompute) so the
            peak forward memory does not exceed --mem-limit (MiB).

            All arguments are OPTIONAL; sensible defaults are provided."""
        ),
    )
    parser.add_argument(
        "--profile",
        default=default_profile_path(),
        help="Dynamic profile CSV (default: dynamic_profiling.csv)",
    )
    parser.add_argument(
        "--mem-limit",
        type=int,
        default=default_mem_limit_mb(),
        help="Target peak memory in MiB "
        f"(default: 80%% of GPU RAM or {default_mem_limit_mb()} MiB)",
    )
    parser.add_argument(
        "--out",
        default="checkpoint_plan.csv",
        help="Output CSV file (default: checkpoint_plan.csv)",
    )
    args = parser.parse_args()

    df = load_profile(args.profile)
    limit_bytes = args.mem_limit * 1024**2

    df, saved, overhead_ms, final_peak = apply_checkpoint_policy(df, limit_bytes)
    df.to_csv(args.out, index=False)

    print("\n──────── Activation-Checkpoint Summary ────────")
    print(f"Profile file           : {args.profile}")
    print(f"Requested peak limit   : {args.mem_limit} MiB")
    print(f"Observed original peak : {df['peak_mem_bytes'].max()/2**20:.1f} MiB")
    print(f"Memory saved by ckpt   : {saved/2**20:.1f} MiB")
    print(f"Peak after scheduling  : {final_peak/2**20:.1f} MiB")
    print(f"Extra recompute time   : {overhead_ms:.2f} ms / iteration")
    print(f"Plan written to        : {args.out}\n")


if __name__ == "__main__":
    main()
