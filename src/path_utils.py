# src/path_utils.py
from __future__ import annotations
from pathlib import Path
import os

def find_repo_root(start: str | Path | None = None) -> Path:
    """
    실행 위치(cwd)와 무관하게 repo root를 찾는다.
    기준(하나라도 만족하면 root 후보):
      - configs/ 디렉토리가 존재
      - README.md 존재
      - .git 존재(리뷰어 zip 배포면 없을 수 있어 보조 조건)
    """
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start).resolve()

    # 파일이면 parent부터 시작
    cur = start_path if start_path.is_dir() else start_path.parent

    for p in [cur] + list(cur.parents):
        if (p / "configs").is_dir() and (p / "README.md").exists():
            return p
        if (p / "configs").is_dir() and (p / "LICENSE").exists():
            return p
        if (p / ".git").exists() and (p / "configs").is_dir():
            return p

    # fallback: cwd 기준으로라도 찾기
    cur = Path.cwd().resolve()
    for p in [cur] + list(cur.parents):
        if (p / "configs").is_dir():
            return p

    # 최후: 현재 위치
    return Path.cwd().resolve()

def resolve_repo_path(path_str: str | None, repo_root: Path) -> Path | None:
    """
    - 절대경로면 그대로
    - 상대경로면 repo_root 기준으로 결합
    - 환경변수/틸드 확장 지원
    """
    if path_str is None:
        return None
    s = str(path_str).strip()
    if s == "":
        return None
    s = os.path.expanduser(os.path.expandvars(s))
    p = Path(s)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()

def resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in ("auto", ""):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return s
