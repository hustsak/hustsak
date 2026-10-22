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


def run_cmd(cmd, env=None):
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running: {cmd}\n{res.stderr}")
        return False
    return True


def generate_contributions(
    days=180,
    author_name="Viraksak Koy",
    author_email="viraksakkoy8@gmail.com",
    log_file=".github/activity.log",
):
    print(f"Generating realistic activity for past {days} days...")
    base_date = datetime.now()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    commit_count = 0
    # Walk forward from past date to present
    for day_offset in range(days, -1, -1):
        target_day = base_date - timedelta(days=day_offset)
        is_weekend = target_day.weekday() >= 5

        # Realistic probabilities:
        # Weekdays: 75% active
        # Weekends: 40% active
        active_prob = 0.40 if is_weekend else 0.75
        if random.random() > active_prob:
            continue

        # 1 to 4 commits on an active day
        num_commits = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.15, 0.10])[0]

        for c_idx in range(num_commits):
            hour = random.randint(9, 22)
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

            if not run_cmd(f'git add "{log_file}"'):
                return
            if not run_cmd(f'git commit -m "{message}"', env=env):
                return

            commit_count += 1

    print(f"\n[DONE] Successfully created {commit_count} historical commits!")


if __name__ == "__main__":
    generate_contributions(days=180)
