"""仅合成临时目录：覆盖完整快照、WAL、一致性、失败关闭和非破坏恢复。"""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import stat
import tarfile
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "deploy/clockworkkoi_upgrade_guard.py"
_SPEC = importlib.util.spec_from_file_location("clockworkkoi_upgrade_guard", _SCRIPT)
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


@pytest.fixture
def state(tmp_path):
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    config = buckets / "config.yaml"
    config.write_text("password: private-synthetic-value\n", encoding="utf-8")
    env = {"OMBRE_BUCKETS_DIR": str(buckets), "OMBRE_VAULT_DIR": str(buckets),
           "OMBRE_CONFIG_PATH": str(config), "OMBRE_CODE_DIR": str(buckets / "_app")}
    return buckets, env


def _populate(buckets):
    values = {"dynamic/私人标题.md": "完整原话", "_sources/source.txt": "原始证据",
              "_media/image.bin": "image", ".dashboard_mcp_tokens.json": "oauth-secret",
              "_app/src/VERSION": "2.16.10", "_app/src/server.py": "print('legacy')"}
    for name, content in values.items():
        path = buckets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (buckets / "_app/src/server.py").chmod(0o700)
    return values


def test_empty_vault_does_not_create_backup(state):
    buckets, env = state
    assert guard.backup("3.6.14", offline=True, environ=env) == "empty"
    assert not (buckets / guard.BACKUP_DIR / "before-3.6.14").exists()
    _populate(buckets)
    (buckets / "_app/src/VERSION").write_text("3.6.14")
    assert guard.backup("3.6.14", offline=True, environ=env) == "empty"


def test_complete_private_snapshot_and_idempotent_restart(state, tmp_path):
    buckets, env = state
    values = _populate(buckets)
    assert guard.backup("3.6.14", offline=True, environ=env) == "created"
    snapshot = buckets / guard.BACKUP_DIR / "before-3.6.14"
    manifest = guard.verify_snapshot(snapshot)
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o500
    assert stat.S_IMODE((snapshot / "snapshot.tar").stat().st_mode) == 0o400
    assert not any(guard.BACKUP_DIR in entry["name"] for entry in manifest["entries"])
    (buckets / "_app/src/VERSION").write_text("3.6.14")
    assert guard.backup("3.6.14", offline=True, environ=env) == "verified"
    recovered = tmp_path / "recovered"
    guard.restore("before-3.6.14", recovered, offline=True, environ=env)
    for name, value in values.items():
        assert (recovered / "root-0" / name).read_text(encoding="utf-8") == value
    assert (recovered / "root-0/config.yaml").read_text() == (buckets / "config.yaml").read_text()
    assert stat.S_IMODE((recovered / "root-0/_app/src/server.py").stat().st_mode) == 0o700
    assert (buckets / "_app/src/VERSION").read_text() == "3.6.14"
    mapping = json.loads((recovered / "RESTORE_MAP.json").read_text())
    guard.verify_snapshot(buckets / guard.BACKUP_DIR / mapping["current_state_backup"])


def test_wal_commit_is_included_without_mutating_source(state, tmp_path):
    buckets, env = state
    _populate(buckets)
    db = buckets / "index.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE data (value TEXT)")
    connection.execute("INSERT INTO data VALUES ('committed-in-wal')")
    connection.commit()
    original = {p.name: p.read_bytes() for p in buckets.glob("index.sqlite3*")}
    try:
        assert guard.backup("3.6.14", offline=True, environ=env) == "created"
        assert {p.name: p.read_bytes() for p in buckets.glob("index.sqlite3*")} == original
        guard.restore("before-3.6.14", tmp_path / "recovered", offline=True, environ=env)
        with sqlite3.connect(tmp_path / "recovered/root-0/index.sqlite3") as recovered:
            assert recovered.execute("SELECT value FROM data").fetchall() == [("committed-in-wal",)]
    finally:
        connection.close()


def test_external_paths_are_included_once(state, tmp_path):
    buckets, env = state
    _populate(buckets)
    external = tmp_path / "external"
    external.mkdir()
    (external / "memory.md").write_text("additional memory")
    env["OMBRE_VAULT_DIR"] = str(external)
    env["OMBRE_CONFIG_PATH"] = str(external / "config.yaml")
    (external / "config.yaml").write_text("external config")
    guard.backup("3.6.14", offline=True, environ=env)
    manifest = guard.verify_snapshot(buckets / guard.BACKUP_DIR / "before-3.6.14")
    assert set(manifest["roots"]) == {str(buckets), str(external)}
    assert sum(entry["name"].endswith("/memory.md") for entry in manifest["entries"]) == 1


def test_low_disk_space_stops_before_completed_backup(state, monkeypatch):
    buckets, env = state
    _populate(buckets)
    monkeypatch.setattr(guard.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(guard.GuardError, match="空间不足"):
        guard.backup("3.6.14", offline=True, environ=env)
    assert not (buckets / guard.BACKUP_DIR / "before-3.6.14").exists()


def test_source_change_during_copy_blocks_upgrade(state, monkeypatch):
    buckets, env = state
    _populate(buckets)
    original = guard._inventory
    calls = 0

    def changed(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            (buckets / "dynamic/私人标题.md").write_text("unexpected writer")
        return original(*args)

    monkeypatch.setattr(guard, "_inventory", changed)
    with pytest.raises(guard.GuardError, match="数据变化"):
        guard.backup("3.6.14", offline=True, environ=env)
    assert not (buckets / guard.BACKUP_DIR / "before-3.6.14").exists()


def test_corruption_and_missing_baseline_block_startup(state):
    buckets, env = state
    _populate(buckets)
    (buckets / "_app/src/VERSION").write_text("3.6.14")
    with pytest.raises(guard.GuardError, match="缺少升级前快照"):
        guard.backup("3.6.14", offline=True, environ=env)
    (buckets / "_app/src/VERSION").write_text("2.16.10")
    guard.backup("3.6.14", offline=True, environ=env)
    archive = buckets / guard.BACKUP_DIR / "before-3.6.14/snapshot.tar"
    archive.chmod(0o600)
    with archive.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(guard.GuardError, match="校验失败"):
        guard.backup("3.6.14", offline=True, environ=env)


def test_symlink_offline_and_restore_destination_rejected(state, tmp_path):
    buckets, env = state
    _populate(buckets)
    with pytest.raises(guard.GuardError, match="offline"):
        guard.backup("3.6.14", environ=env)
    (buckets / "linked").symlink_to(tmp_path / "outside")
    with pytest.raises(guard.GuardError, match="符号链接"):
        guard.backup("3.6.14", offline=True, environ=env)
    (buckets / "linked").unlink()
    guard.backup("3.6.14", offline=True, environ=env)
    with pytest.raises(guard.GuardError, match="全新目录"):
        guard.restore("before-3.6.14", buckets / "restored", offline=True, environ=env)


def test_cli_failure_never_prints_private_path_or_secret(state, monkeypatch, capsys):
    buckets, env = state
    _populate(buckets)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    def fail(*args, **kwargs):
        raise OSError("private-synthetic-value 私人标题.md")
    monkeypatch.setattr(guard, "_create_snapshot", fail)
    assert guard.main(["backup", "--target-version", "3.6.14", "--offline"]) == 1
    output = capsys.readouterr()
    assert "private-synthetic-value" not in output.err
    assert "私人标题" not in output.err


def test_restore_to_dedicated_persistent_slot_and_reject_symlink(state, tmp_path):
    buckets, env = state
    _populate(buckets)
    guard.backup("3.6.15", offline=True, environ=env)
    backup_root = buckets / guard.BACKUP_DIR
    destination = backup_root / "restores" / "rollback-copy"
    preserved = guard.restore("before-3.6.15", destination, offline=True, environ=env)
    assert (destination / "root-0/dynamic/私人标题.md").read_text() == "完整原话"
    manifest = guard.verify_snapshot(backup_root / preserved)
    assert not any("restores" in entry["name"] for entry in manifest["entries"])
    (tmp_path / "alias").symlink_to(backup_root / "restores", target_is_directory=True)
    with pytest.raises(guard.GuardError, match="全新目录"):
        guard.restore("before-3.6.15", tmp_path / "alias" / "bad-copy", offline=True, environ=env)


def test_corrupt_current_database_is_preserved_without_blocking_restore(state, tmp_path):
    buckets, env = state
    _populate(buckets)
    database = buckets / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE remembered(value TEXT)")
    guard.backup("3.6.15", offline=True, environ=env)
    corrupted = b"SQLite format 3\x00broken-current-data"
    database.write_bytes(corrupted)
    restored = tmp_path / "restored"
    preserved = guard.restore("before-3.6.15", restored, offline=True, environ=env)
    assert database.read_bytes() == corrupted
    with sqlite3.connect(restored / "root-0/index.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    manifest = guard.verify_snapshot(buckets / guard.BACKUP_DIR / preserved, check_sqlite=False)
    assert manifest["sqlite_integrity_checked"] is False


def test_archive_traversal_is_rejected_even_if_checksum_is_recomputed(tmp_path):
    backup_root = tmp_path / "backups"
    snapshot = backup_root / "malicious"
    snapshot.mkdir(parents=True)
    archive_path = snapshot / "snapshot.tar"
    content = b"do not write this"
    with tarfile.open(archive_path, "w:") as archive:
        member = tarfile.TarInfo("../escaped")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    manifest = {"format": 1, "roots": [], "archive_sha256": guard._file_digest(archive_path),
                "entries": [{"name": "../escaped", "kind": "file", "size": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()}]}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(guard.GuardError, match="非法路径"):
        guard.verify_snapshot(snapshot)
    assert not (backup_root / "escaped").exists()
