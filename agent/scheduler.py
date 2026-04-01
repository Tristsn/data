"""
定时任务调度 — 每日总结与性格更新
"""

from apscheduler.schedulers.background import BackgroundScheduler
from agent import EmotionalAgent


# 全局Agent实例池（实际项目中按用户ID管理）
_agents = {}


def get_agent(user_id: str) -> EmotionalAgent:
    if user_id not in _agents:
        _agents[user_id] = EmotionalAgent(user_id=user_id)
    return _agents[user_id]


def run_daily_summary():
    """每日凌晨2点执行所有活跃用户的总结"""
    print("[Scheduler] Running daily summary...")
    for user_id, agent in _agents.items():
        try:
            agent.daily_summary()
            print(f"  [OK] {user_id}")
        except Exception as e:
            print(f"  [FAIL] {user_id}: {e}")


def start_scheduler():
    """启动定时任务"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_daily_summary, "cron", hour=2, minute=0)
    scheduler.start()
    print("[Scheduler] Started. Daily summary at 02:00.")
    return scheduler


if __name__ == "__main__":
    # 测试：手动触发
    agent = get_agent("test_user")
    agent.session_dialogues = [
        "用户: 今天工作好累",
        "助手: 辛苦啦，要不要休息一下呀~",
        "用户: 嗯，谢谢你",
        "助手: 不客气呢，我一直在的~",
    ]
    agent.daily_summary()
