# app/utils/scheduler.py
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List


def calculate_next_review(score_pct: float, current_streak: int = 0) -> Dict[str, Any]:
    """
    Spaced repetition scheduling algorithm (based on SuperMemo SM-2 principles).
    Determines next review date and interval based on quiz performance.
    """
    now = datetime.now()

    if score_pct < 60.0:
        # Struggling: review within 24 hours
        interval_days = 1
        new_streak = 0
        status = "Needs Urgent Review"
    elif score_pct < 75.0:
        # Borderline: review in 2-3 days
        interval_days = 2
        new_streak = max(1, current_streak)
        status = "Reinforce Soon"
    elif score_pct < 90.0:
        # Good: review in 5-7 days
        interval_days = 5 + (current_streak * 2)
        new_streak = current_streak + 1
        status = "Progressing Well"
    else:
        # Mastered: review in 10-14 days
        interval_days = 10 + (current_streak * 3)
        new_streak = current_streak + 1
        status = "Mastered"

    next_date = now + timedelta(days=interval_days)

    return {
        "interval_days": interval_days,
        "new_streak": new_streak,
        "status": status,
        "next_review_date": next_date.strftime("%Y-%m-%d"),
        "timestamp": int(next_date.timestamp()),
    }


def get_due_topics(user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scans user profile and returns topics that are due for revision today or overdue.
    """
    due_topics = []
    now_date = datetime.now().date()

    for topic, data in user_profile.items():
        if not isinstance(data, dict):
            continue

        review_info = data.get("review_schedule", {})
        next_review_str = review_info.get("next_review_date")

        is_due = False
        if next_review_str:
            try:
                target_date = datetime.strptime(next_review_str, "%Y-%m-%d").date()
                if target_date <= now_date:
                    is_due = True
            except ValueError:
                is_due = True
        else:
            # If never scheduled and score < 70, mark as due
            avg_score = data.get("avg_score", 0)
            if isinstance(avg_score, (int, float)) and avg_score < 70 and data.get("attempts", 0) > 0:
                is_due = True

        if is_due:
            due_topics.append({
                "topic": topic,
                "avg_score": data.get("avg_score", 0),
                "attempts": data.get("attempts", 0),
                "revisions_count": len(data.get("revisions", [])),
                "status": review_info.get("status", "Review Required"),
            })

    return due_topics
