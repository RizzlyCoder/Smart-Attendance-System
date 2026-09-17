from src.db import Database


def test_add_and_get_user(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    uid = db.add_user("Alice")
    assert isinstance(uid, int)
    row = db.get_user_by_name("Alice")
    assert row == (uid, "Alice")


def test_list_users_sorted(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    db.add_user("Charlie")
    db.add_user("Alice")
    db.add_user("Bob")
    names = [name for _, name in db.list_users()]
    assert names == ["Alice", "Bob", "Charlie"]


def test_mark_attendance_once_per_day(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    uid = db.add_user("Alice")
    first = db.mark_attendance(uid, "2026-09-10", "09:00:00", 40.0, "happy")
    second = db.mark_attendance(uid, "2026-09-10", "09:15:00", 42.0, "neutral")
    assert first is True
    assert second is False  # duplicate for same date rejected

    records = db.get_attendance()
    assert len(records) == 1
    assert records[0][1] == "Alice"


def test_get_attendance_date_filtering(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    uid = db.add_user("Alice")
    db.mark_attendance(uid, "2026-09-01", "09:00:00", 30.0, "happy")
    db.mark_attendance(uid, "2026-09-05", "09:00:00", 30.0, "sad")

    all_records = db.get_attendance()
    filtered = db.get_attendance(start_date="2026-09-03")
    assert len(all_records) == 2
    assert len(filtered) == 1
    assert filtered[0][2] == "2026-09-05"
