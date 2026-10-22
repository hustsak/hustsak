"""
GitHub Realistic Contribution Generator
=======================================
Generates a realistic active contribution history for the past 6 months
(approx 180 days) matching AI, Data Science & Computer Vision research work.
"""

import os
import random
import subprocess
from datetime import datetime, timedelta

RESEARCH_TOPICS = [
    "optimize YOLOv11 inference pipeline",
    "preprocess multimodal sensor telemetry",
    "fine-tune vision transformer backbone",
    "benchmark object tracking latency",
    "clean and normalize dataset features",
    "update exploratory data analysis notebooks",
    "tune hyperparameter learning rate schedules",
    "refactor neural model evaluation metrics",
    "implement feature extraction pipelines",
    "test distributed model checkpointing",
    "enhance gradient clipping stability",
    "evaluate cross-validation performance",
    "add data augmentation transformations",
    "profile GPU memory allocation during training",
    "update research experiment benchmarks",
]


def run_cmd(cmd_list, env=None):
    res = subprocess.run(cmd_list, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running: {' '.join(cmd_list)}\n{res.stderr}")
        return False
    return True


def generate_contributions(
    days=365,
    author_name="Viraksak Koy",
    author_email="viraksakkoy8@gmail.com",
    log_file=".github/activity.log",
):
    print(f"Generating dense high-activity green plots for past {days} days (1 full year)...")
    base_date = datetime.now()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    commit_count = 0
    # Walk forward from past date (365 days ago) to present
    for day_offset in range(days, -1, -1):
        target_day = base_date - timedelta(days=day_offset)
        is_weekend = target_day.weekday() >= 5

        # 95% active on weekdays, 85% active on weekends
        active_prob = 0.85 if is_weekend else 0.95
        if random.random() > active_prob:
            continue

        # 2 to 8 commits per day to generate diverse shades of green (levels 1-4)
        num_commits = random.choices(
            [2, 3, 4, 5, 6, 7, 8],
            weights=[0.15, 0.25, 0.25, 0.15, 0.10, 0.06, 0.04],
        )[0]

        for _ in range(num_commits):
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            commit_time = target_day.replace(
                hour=hour, minute=minute, second=second, microsecond=0
            )

            iso_date = commit_time.strftime("%Y-%m-%dT%H:%M:%S+07:00")
            topic = random.choice(RESEARCH_TOPICS)
            message = f"research: {topic} [log #{commit_count + 1}]"

            # Append to log file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{iso_date}] {message}\n")

            env = os.environ.copy()
            env["GIT_AUTHOR_NAME"] = author_name
            env["GIT_AUTHOR_EMAIL"] = author_email
            env["GIT_COMMITTER_NAME"] = author_name
            env["GIT_COMMITTER_EMAIL"] = author_email
            env["GIT_AUTHOR_DATE"] = iso_date
            env["GIT_COMMITTER_DATE"] = iso_date

            if not run_cmd(["git", "add", log_file]):
                return
            if not run_cmd(["git", "commit", "-m", message], env=env):
                return

            commit_count += 1

            if commit_count % 100 == 0:
                print(f"  ... created {commit_count} commits")

    print(f"\n[DONE] Successfully created {commit_count} high-density commits across the full year!")


def generate_daily_contributions(
    count=3,
    author_name="Viraksak Koy",
    author_email="viraksakkoy8@gmail.com",
    log_file=".github/activity.log",
):
    print(f"Generating {count} daily active green contribution commits...")
    base_date = datetime.now()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    for i in range(count):
        iso_date = base_date.strftime("%Y-%m-%dT%H:%M:%S+07:00")
        topic = random.choice(RESEARCH_TOPICS)
        message = f"research: {topic} [daily pulse #{i + 1}]"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{iso_date}] {message}\n")

        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = author_name
        env["GIT_AUTHOR_EMAIL"] = author_email
        env["GIT_COMMITTER_NAME"] = author_name
        env["GIT_COMMITTER_EMAIL"] = author_email
        env["GIT_AUTHOR_DATE"] = iso_date
        env["GIT_COMMITTER_DATE"] = iso_date

        run_cmd(["git", "add", log_file])
        run_cmd(["git", "commit", "-m", f"{message} [skip ci]"], env=env)

    print(f"[DONE] Successfully added {count} active green contribution commits!")


if __name__ == "__main__":
    import sys

    if "--daily" in sys.argv:
        idx = sys.argv.index("--daily")
        num = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 3
        generate_daily_contributions(count=num)
    else:
        generate_contributions(days=365)
