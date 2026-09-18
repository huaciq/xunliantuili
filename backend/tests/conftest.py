import os
from pathlib import Path

TEST_DB = Path(__file__).parent / "test.db"
TEST_STORAGE = Path(__file__).parent / "test-storage"
if TEST_DB.exists():
    TEST_DB.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["APP_SECRET_KEY"] = "test-secret-key"
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = "admin@example.com"
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = "AdminPass123!"
os.environ["AUTO_CREATE_TABLES"] = "true"
os.environ["STORAGE_ROOT"] = str(TEST_STORAGE)
os.environ["FAKE_RUN_DURATION_SECONDS"] = "1"
os.environ["SCHEDULER_POLL_SECONDS"] = "0.05"
