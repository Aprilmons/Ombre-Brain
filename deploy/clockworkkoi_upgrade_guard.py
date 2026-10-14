#!/usr/bin/env python3
"""ClockworkKoi 大版本升级前的离线完整快照与恢复副本工具。

只能在旧实例已停止、卷由本进程独占时运行。复制前后逐文件校验用于发现
意外写入，不代替停服。必须在 entrypoint 创建配置或播种代码之前调用。
本工具不导入应用代码、不改数据库、不读取环境中的密码值。

backup --target-version 3.6.15 --offline
verify --backup-id before-3.6.15
restore --backup-id before-3.6.15 --destination /另一个全新目录 --offline

restore 先留存当前状态，再生成恢复副本与 RESTORE_MAP.json；绝不自动覆盖
在线目录。停服后用旧镜像，按映射切换各路径，才是实际回滚。环境变量中的
凭据仍由部署平台保管，快照不会复制它们。同卷快照不是异地灾备。
只有一个持久卷时，可将恢复目标指定为该卷中
.clockworkkoi-upgrade-backups/restores/一个全新标识；此目录不参与备份。
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import sys
import tarfile
import tempfile
import uuid


BACKUP_DIR = ".clockworkkoi-upgrade-backups"
RESERVE_BYTES = 64 * 1024 * 1024


class GuardError(RuntimeError):
    """只携带可安全公开的错误，不包含私人路径或数据。"""


def _digest(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _file_digest(path):
    with path.open("rb") as stream:
        return _digest(stream)


def configured_paths(environ=None):
    env = os.environ if environ is None else environ
    image = Path(env.get("OMBRE_IMAGE_ROOT", "/app")).resolve()
    buckets = Path(env.get("OMBRE_BUCKETS_DIR", str(image / "buckets"))).resolve()
    config = Path(env.get("OMBRE_CONFIG_PATH", str(image / "config.yaml"))).resolve()
    code = Path(env.get("OMBRE_CODE_DIR", str(config.parent / "_app"))).resolve()
    candidates = [buckets, Path(env.get("OMBRE_VAULT_DIR", str(buckets))).resolve(),
                  config, code]
    # 父目录已经完整纳入时，不再重复保存其中的配置和活动代码。
    roots = []
    for candidate in sorted(set(candidates), key=lambda path: (len(path.parts), str(path))):
        if candidate == Path("/"):
            raise GuardError("路径配置过宽，无法安全备份。")
        if not any(candidate == parent or parent in candidate.parents for parent in roots):
            roots.append(candidate)
    return buckets, roots, config, code


def _safe_name(name):
    path = PurePosixPath(name)
    return not path.is_absolute() and bool(path.parts) and all(
        part not in ("", ".", "..") for part in path.parts
    ) and "\\" not in name


def _inventory(roots, excluded):
    records = []

    def visit(path, name):
        if path == excluded:
            return
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise GuardError("备份范围含符号链接或特殊文件，需先明确其持久化位置。")
        record = {"name": name, "kind": "dir" if path.is_dir() else "file",
                  "mode": stat.S_IMODE(info.st_mode), "mtime_ns": info.st_mtime_ns,
                  "ctime_ns": info.st_ctime_ns, "inode": info.st_ino,
                  "size": info.st_size if path.is_file() else 0}
        if path.is_file():
            with path.open("rb") as stream:
                record["sqlite"] = (stream.read(16) == b"SQLite format 3\x00"
                                    or path.suffix.lower() in (".db", ".sqlite", ".sqlite3"))
                stream.seek(0)
                record["sha256"] = _digest(stream)
            after = path.stat()
            if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode) != (
                info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode
            ):
                raise GuardError("备份期间检测到文件变化；请停止所有写入后重试。")
        records.append(record)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child, name + "/" + child.name)

    for index, root in enumerate(roots):
        if root.exists() or root.is_symlink():
            visit(root, f"root-{index}")
    return records


def _source_path(roots, name):
    parts = PurePosixPath(name).parts
    return roots[int(parts[0].split("-")[1])].joinpath(*parts[1:])


def _has_existing_state(roots, config, excluded):
    for root in roots:
        if root.is_file() and root != config:
            return True
        if root.is_dir():
            for current, directories, files in os.walk(root):
                directories[:] = [name for name in directories if Path(current) / name != excluded]
                if any(Path(current) / name != config for name in files):
                    return True
                if any((Path(current) / name).is_symlink() for name in directories):
                    return True
    return False


def _space_check(directory, records):
    # 归档与 SQLite 核验副本最多同时占两份；预留 tar 元数据和余量。
    required = 2 * sum(item["size"] + 1024 for item in records) + RESERVE_BYTES
    if shutil.disk_usage(directory).free < required:
        raise GuardError("剩余空间不足以保存并验证完整快照，已阻止升级。")


def _sync(path):
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _extract_members(archive, destination, names=None, preserve_mode=False):
    # 不使用 extractall；只允许普通目录与文件，不信任归档中的链接和权限。
    for member in archive.getmembers():
        if names is not None and member.name not in names:
            continue
        if not _safe_name(member.name) or not (member.isdir() or member.isfile()):
            raise GuardError("快照包含非法路径或文件类型。")
        output = destination.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with output.open("xb") as target, archive.extractfile(member) as source:
                shutil.copyfileobj(source, target)
            output.chmod((member.mode & 0o777) if preserve_mode else 0o600)
    if preserve_mode:
        for member in reversed(archive.getmembers()):
            if member.isdir():
                destination.joinpath(*PurePosixPath(member.name).parts).chmod(member.mode & 0o777)


def verify_snapshot(snapshot, expected_roots=None, check_sqlite=True):
    try:
        if snapshot.is_symlink() or any((snapshot / name).is_symlink() for name in ("manifest.json", "snapshot.tar")):
            raise GuardError("快照路径不安全。")
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        archive_path = snapshot / "snapshot.tar"
        if manifest.get("format") != 1 or _file_digest(archive_path) != manifest["archive_sha256"]:
            raise GuardError("快照校验失败，拒绝继续。")
        if expected_roots is not None and manifest["roots"] != [str(path) for path in expected_roots]:
            raise GuardError("持久化路径与原快照不一致，需人工确认后再升级。")
        expected = {item["name"]: item for item in manifest["entries"]}
        if len(expected) != len(manifest["entries"]):
            raise GuardError("快照清单存在重复路径。")
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            if len(members) != len(expected) or {item.name for item in members} != set(expected):
                raise GuardError("快照文件清单不完整。")
            for member in members:
                record = expected[member.name]
                if not _safe_name(member.name) or not (member.isdir() or member.isfile()):
                    raise GuardError("快照包含非法路径或文件类型。")
                if member.isdir() != (record["kind"] == "dir"):
                    raise GuardError("快照文件类型不一致。")
                if member.isfile():
                    with archive.extractfile(member) as source:
                        if member.size != record["size"] or _digest(source) != record["sha256"]:
                            raise GuardError("快照文件内容校验失败。")
            databases = [item["name"] for item in manifest["entries"] if item.get("sqlite")]
            if databases and check_sqlite:
                with tempfile.TemporaryDirectory(prefix=".verify-", dir=snapshot.parent) as temporary:
                    directory = Path(temporary)
                    names = {name + suffix for name in databases for suffix in ("", "-wal", "-shm", "-journal")}
                    _extract_members(archive, directory, names)
                    for name in databases:
                        uri = (directory / name).as_uri() + "?mode=ro"
                        with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
                            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                                raise GuardError("SQLite 快照完整性校验失败。")
        return manifest
    except GuardError:
        raise
    except Exception:
        raise GuardError("无法验证快照完整性，已阻止继续。") from None


def _create_snapshot(roots, backup_root, identifier, target_version, check_sqlite=True):
    final = backup_root / identifier
    if final.exists():
        raise GuardError("快照名称已存在，拒绝覆盖。")
    records = _inventory(roots, backup_root)
    _space_check(backup_root, records)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=backup_root))
    try:
        archive_path = pending / "snapshot.tar"
        with tarfile.open(archive_path, "w:", format=tarfile.PAX_FORMAT) as archive:
            for record in records:
                info = tarfile.TarInfo(record["name"])
                info.mode = record["mode"]
                info.mtime = record["mtime_ns"] / 1_000_000_000
                if record["kind"] == "dir":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                else:
                    info.size = record["size"]
                    with _source_path(roots, record["name"]).open("rb") as source:
                        archive.addfile(info, source)
        if records != _inventory(roots, backup_root):
            raise GuardError("备份期间检测到数据变化；请停止所有写入后重试。")
        manifest = {"format": 1, "target_version": target_version,
                    "roots": [str(path) for path in roots], "entries": records,
                    "sqlite_integrity_checked": check_sqlite,
                    "archive_sha256": _file_digest(archive_path)}
        manifest_path = pending / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in (archive_path, manifest_path):
            path.chmod(0o400)
            _sync(path)
        verify_snapshot(pending, roots, check_sqlite=check_sqlite)
        _sync_dir(pending)
        pending.chmod(0o500)
        pending.rename(final)
        _sync_dir(backup_root)
        return final
    except Exception:
        # 失败的部分副本保留在私有目录，绝不伪装为已完成备份。
        raise


@contextlib.contextmanager
def _lock(backup_root):
    backup_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if backup_root.is_symlink() or not backup_root.is_dir():
        raise GuardError("备份目录不安全。")
    backup_root.chmod(0o700)
    with (backup_root / ".lock").open("a+b") as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GuardError("另一项备份或恢复操作正在进行。") from None
        yield


def backup(target_version, offline=False, environ=None):
    if not offline:
        raise GuardError("备份前必须停止旧实例并确认卷独占，使用 --offline。")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", target_version):
        raise GuardError("目标版本格式无效。")
    buckets, roots, config, code = configured_paths(environ)
    backup_root = buckets / BACKUP_DIR
    existing = backup_root / ("before-" + target_version)
    fresh_marker = backup_root / ("fresh-" + target_version + ".json")
    with _lock(backup_root):
        if existing.exists():
            verify_snapshot(existing, roots)
            return "verified"
        fresh_state = {"format": 1, "target_version": target_version,
                       "roots": [str(path) for path in roots]}
        if fresh_marker.exists():
            if fresh_marker.is_symlink() or json.loads(fresh_marker.read_text(encoding="utf-8")) != fresh_state:
                raise GuardError("空库初始化标记校验失败。")
            return "empty"
        if not _has_existing_state(roots, config, backup_root):
            # 只记“本版本从空卷开始”，不制作无意义备份；后续启动不能误报缺基线。
            with fresh_marker.open("x", encoding="utf-8") as marker:
                json.dump(fresh_state, marker)
            fresh_marker.chmod(0o400)
            _sync(fresh_marker)
            _sync_dir(backup_root)
            return "empty"
        for version_file in (code / "src/VERSION", code / "VERSION"):
            if version_file.is_file():
                version = version_file.read_text(encoding="utf-8").strip()
                if re.match(r"^[3-9]\d*\.", version):
                    raise GuardError("已有 3.x 运行数据但缺少升级前快照，不能补造迁移前备份。")
        _create_snapshot(roots, backup_root, existing.name, target_version)
        return "created"


def restore(backup_id, destination, offline=False, environ=None):
    if not offline:
        raise GuardError("恢复前必须停止实例并确认卷独占，使用 --offline。")
    if not re.fullmatch(r"[A-Za-z0-9.+-]+", backup_id):
        raise GuardError("快照名称无效。")
    buckets, roots, _, _ = configured_paths(environ)
    backup_root = buckets / BACKUP_DIR
    destination = Path(destination).absolute()
    # 唯一允许的卷内恢复位置是已经被备份排除的专用 restores/<新标识>。
    dedicated_slot = destination.parent == backup_root / "restores"
    within_live_root = any(root == destination or root in destination.parents for root in roots)
    if (destination != destination.resolve() or destination.exists() or destination.is_symlink()
            or (within_live_root and not dedicated_slot)):
        raise GuardError("恢复目标必须是专用 restores 子目录或活动数据目录之外的全新目录。")
    with _lock(backup_root):
        manifest = verify_snapshot(backup_root / backup_id, roots)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _space_check(destination.parent, manifest["entries"])
        preserved_id = "before-restore-" + uuid.uuid4().hex
        # 当前数据库可能正因损坏而需要回滚；逐字节保留并验证即可，不妨碍恢复健康基线。
        _create_snapshot(roots, backup_root, preserved_id, "restore-preservation", check_sqlite=False)
        destination.mkdir(mode=0o700)
        try:
            with tarfile.open(backup_root / backup_id / "snapshot.tar", "r:") as archive:
                _extract_members(archive, destination, preserve_mode=True)
            for record in manifest["entries"]:
                if record["kind"] == "file":
                    restored = destination.joinpath(*PurePosixPath(record["name"]).parts)
                    if restored.stat().st_size != record["size"] or _file_digest(restored) != record["sha256"]:
                        raise GuardError("恢复副本内容校验失败。")
                    _sync(restored)
            mapping = {"status": "complete", "current_state_backup": preserved_id, "original_backup": backup_id,
                       "roots": {f"root-{index}": path for index, path in enumerate(manifest["roots"])}}
            map_path = destination / "RESTORE_MAP.json"
            map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            map_path.chmod(0o600)
            _sync(map_path)
            _sync_dir(destination)
        except Exception:
            raise GuardError("恢复副本生成中断；当前数据及完整快照均已保留，勿使用不完整副本。") from None
        return preserved_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("backup")
    create.add_argument("--target-version", required=True)
    create.add_argument("--offline", action="store_true")
    check = commands.add_parser("verify")
    check.add_argument("--backup-id", required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--backup-id", required=True)
    recover.add_argument("--destination", required=True)
    recover.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == "backup":
            result = backup(args.target_version, args.offline)
            print("[upgrade-guard] " + {"empty": "已确认本版本从空库初始化，无需旧版迁移备份。", "created": "完整快照已保存并验证。", "verified": "已存在的升级前快照验证通过。"}[result])
        elif args.command == "verify":
            if not re.fullmatch(r"[A-Za-z0-9.+-]+", args.backup_id):
                raise GuardError("快照名称无效。")
            buckets, roots, _, _ = configured_paths()
            with _lock(buckets / BACKUP_DIR):
                verify_snapshot(buckets / BACKUP_DIR / args.backup_id, roots)
            print("[upgrade-guard] 完整快照验证通过。")
        else:
            restore(args.backup_id, args.destination, args.offline)
            print("[upgrade-guard] 当前状态已另存，恢复副本已生成；活动数据未被覆盖。")
        return 0
    except GuardError as error:
        print("[upgrade-guard] FATAL: " + str(error), file=sys.stderr)
    except Exception:
        print("[upgrade-guard] FATAL: 备份或恢复失败，已阻止继续；详情未输出以保护隐私。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
