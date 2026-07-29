import re
import glob
import argparse
import pandas as pd


FILENAME_RE = re.compile(
    r"^5o_nano_(?P<fov>\d+)_(?:online_p(?P<threshold>\d+)|offline)_seed_(?P<seed>\d+)_mp3d_result\.csv$"
)


def parse_filename(path):
    name = path.split("/")[-1]
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {name}")

    fov = int(m.group("fov"))
    seed = int(m.group("seed"))
    threshold_digits = m.group("threshold")
    if threshold_digits is None:
        mode = "offline"
        threshold = None
    else:
        mode = "online"
        threshold = float("0." + threshold_digits)

    return fov, mode, threshold, seed


def compute_spl(df):
    success = df["true_success"].astype(bool)
    actual = df["total_distance_traversed"]
    geo = df["geodesic_distance"]

    denom = pd.concat([actual, geo], axis=1).max(axis=1)
    ratio = geo / denom
    # both actual and geodesic are 0 (spawned at goal) -> perfect ratio
    ratio = ratio.where(denom != 0, 1.0)

    spl = success.astype(float) * ratio.where(success, 0.0).fillna(0.0)
    return spl.mean()


def process_file(path):
    fov, mode, threshold, seed = parse_filename(path)

    df = pd.read_csv(path)

    # Some result files contain duplicate rows per episode (rerun appended
    # instead of overwritten); keep one row per episode, preferring rows
    # with a non-null goal.
    if df["episode_number"].duplicated().any():
        df = df.sort_values(by="goal", na_position="last")
        df = df.drop_duplicates(subset="episode_number", keep="first")

    success_rate = df["true_success"].astype(bool).mean()
    spl = compute_spl(df)

    return {
        "seed": seed,
        "fov": fov,
        "mode": mode,
        "threshold": threshold,
        "num_episodes": len(df),
        "success_rate": success_rate,
        "spl": spl,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate CPnav result CSVs into a summary CSV")
    parser.add_argument("--results_dir", type=str, default="logs/results")
    parser.add_argument("--out", type=str, default="logs/final_results.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.results_dir}/*.csv"))
    rows = [process_file(f) for f in files]

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by=["mode", "fov", "threshold", "seed"], na_position="first")
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
