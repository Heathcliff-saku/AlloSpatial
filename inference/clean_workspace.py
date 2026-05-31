#!/usr/bin/env python3
"""清理 demo/workspace 下的场景缓存,保留评测结果等其他目录。

判定规则:仅当顶层条目是目录且其中包含 `cognitive_map_output` 子目录时,
视为场景缓存并删除。其他一切(如 results / vsibench_tiny / mindcube_tiny /
vsibench_debiased / local_baseline_results 等)一律保留。

默认 dry-run,加 --execute 才会真正删除。
  python3 demo/clean_workspace.py --execute --yes  
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_WORKSPACE = Path(__file__).resolve().parent / "workspace"
CACHE_MARKER = "cognitive_map_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"待清理的 workspace 目录 (默认: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--marker",
        default=CACHE_MARKER,
        help=f"判定为场景缓存的标志子目录名 (默认: {CACHE_MARKER})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除 (不带此参数为 dry-run)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互确认 (仅在 --execute 下生效)",
    )
    parser.add_argument(
        "--show-kept",
        action="store_true",
        help="额外打印被保留的条目名",
    )
    return parser.parse_args()


def is_scene_cache(path: Path, marker: str) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    return (path / marker).is_dir()


def main() -> int:
    args = parse_args()
    workspace: Path = args.workspace

    if not workspace.is_dir():
        print(f"[ERROR] workspace 不存在或不是目录: {workspace}", file=sys.stderr)
        return 2

    entries = sorted(workspace.iterdir(), key=lambda p: p.name)
    to_delete = [p for p in entries if is_scene_cache(p, args.marker)]
    kept = [p for p in entries if p not in to_delete]

    print(f"workspace : {workspace}")
    print(f"判定规则  : 顶层目录中存在 `{args.marker}/` -> 场景缓存")
    print(f"扫描总数  : {len(entries)}")
    print(f"待删除项  : {len(to_delete)}")
    print(f"保留项    : {len(kept)}")
    if args.show_kept:
        for p in kept:
            print(f"  KEEP  {p.name}")
    print(f"模式      : {'EXECUTE (会真正删除)' if args.execute else 'DRY-RUN (仅打印)'}")

    if not to_delete:
        print("没有需要清理的内容。")
        return 0

    if args.execute and not args.yes:
        ans = input(f"确认要删除 {len(to_delete)} 个场景缓存目录?输入 yes 继续: ").strip().lower()
        if ans != "yes":
            print("已取消。")
            return 1

    deleted, failed = 0, 0
    for i, path in enumerate(to_delete, 1):
        if args.execute:
            try:
                shutil.rmtree(path)
                deleted += 1
            except OSError as e:
                failed += 1
                print(f"[FAIL] {path.name}: {e}", file=sys.stderr)
        if i % 500 == 0 or i == len(to_delete):
            tag = "deleted" if args.execute else "would-delete"
            print(f"  [{i}/{len(to_delete)}] {tag} ...")

    if args.execute:
        print(f"完成。已删除 {deleted} 个,失败 {failed} 个。")
        return 0 if failed == 0 else 1
    print("Dry-run 结束。加 --execute 真正执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
