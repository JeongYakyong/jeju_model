# -*- coding: utf-8 -*-
"""사이트 접속 잠금 — 진입점(app.py)에서 화면 전체를 막는 게이트 + 관리자 설정 UI.

관리자메뉴 잠금(``common.ops_gate``)과는 **목적이 다른 별개 기능**이다:

  - ops_gate  : 이미 들어온 사람이 '수집·예측 실행' 버튼을 못 누르게 막는다.
  - site_lock : 사이트에 **들어오는 것 자체**를 막는다. 잠기면 아무 메뉴도 안 그려진다.

해제는 **세션(브라우저 탭) 단위** — 한 번 풀면 그 세션 동안 유지되고, 다른 브라우저나
다른 사람에게는 전파되지 않는다.

설정은 저장소 루트의 ``site_lock.json`` 한 파일(``P.SITE_LOCK``)에 담는다.
이 파일은 .gitignore 대상이라 ``git pull`` 배포로 지워지거나 덮어써지지 않는다 —
서버 재시작·재배포에도 설정이 그대로 남는다.

비밀번호는 평문이 아니라 **PBKDF2-HMAC-SHA256 해시**로 저장한다. 파일을 열어봐도
비밀번호 자체는 알 수 없다. 대신 잊어버리면 되돌릴 수 없으므로 복구는 이렇게 한다:

    site_lock.json 의 "enabled" 를 false 로 고쳐 저장  →  사이트가 열린다
    →  관리자 메뉴에서 새 비밀번호를 지정하고 다시 켠다.

첫 실행 때 파일이 없으면 기존 게이트가 쓰던 ``.streamlit/secrets.toml`` 의 password 로
자동 시드한다 — 그래서 바꾼 뒤에도 지금까지 쓰던 비밀번호가 그대로 통한다.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import hmac
import json
import os
import secrets as pysecrets
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent   # pages/ 의 한 단계 위 = 저장소 루트
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

CONFIG_FILE = Path(P.SITE_LOCK)
SESSION_KEY = "site_unlocked"       # 세션 단위 해제 플래그 (브라우저마다 따로)
PBKDF2_ITERATIONS = 200_000


# ---------------------------------------------------------------- 설정 파일 입출력
def _hash_password(password: str, salt_hex: str, iterations: int) -> str:
    """비밀번호 → PBKDF2 해시(hex). 같은 salt·iterations 면 항상 같은 값이 나온다."""
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), iterations)
    return digest.hex()


def _password_fields(password: str) -> dict:
    salt_hex = pysecrets.token_hex(16)
    return {"algo": "pbkdf2_sha256", "iterations": PBKDF2_ITERATIONS, "salt": salt_hex,
            "hash": _hash_password(password, salt_hex, PBKDF2_ITERATIONS),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _write_config(config: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    try:                       # 소유자만 읽게 — Windows 에서는 무시된다
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def _seed_config() -> dict:
    """설정 파일이 없을 때의 최초 1회 생성 — 기존 secrets.toml 비밀번호를 물려받는다."""
    try:
        legacy_password = str(st.secrets["password"])
    except Exception:          # noqa: BLE001 — secrets.toml 이 없는 설치도 있다
        legacy_password = ""
    config = {"enabled": bool(legacy_password)}
    if legacy_password:
        config.update(_password_fields(legacy_password))
    _write_config(config)
    return config


def load_config() -> dict:
    """현재 설정. 파일이 없으면 만들고, 깨져 있으면 잠금 꺼짐으로 본다(잠겨서 못 들어가는 것 방지)."""
    if not CONFIG_FILE.exists():
        return _seed_config()
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return config if isinstance(config, dict) else {"enabled": False}
    except Exception:          # noqa: BLE001
        return {"enabled": False}


def has_password(config: dict) -> bool:
    return bool(config.get("hash") and config.get("salt"))


def is_locked(config: dict) -> bool:
    """실제로 잠글지 여부.

    비밀번호가 없으면 켜져 있어도 잠그지 않는다 — 손으로 JSON 을 고쳐 enabled 만 켜 두면
    아무도 못 들어오는 상태가 되므로, 그 사고를 구조적으로 막는다.
    """
    return bool(config.get("enabled")) and has_password(config)


def verify(password: str, config: dict) -> bool:
    if not has_password(config):
        return False
    expected = str(config["hash"])
    actual = _hash_password(password, str(config["salt"]),
                            int(config.get("iterations", PBKDF2_ITERATIONS)))
    return hmac.compare_digest(expected, actual)   # 타이밍 공격 방지


# ---------------------------------------------------------------- 게이트 (진입점에서 호출)
def gate() -> bool:
    """잠김이면 비밀번호 화면만 그리고 False, 통과면 True.

    app.py 맨 위에서:  ``if not site_lock.gate(): st.stop()``
    """
    config = load_config()
    if not is_locked(config):
        return True
    if st.session_state.get(SESSION_KEY):
        return True

    st.title(" ")
    with st.form("site_lock_form"):
        password = st.text_input("비밀번호를 입력하세요", type="password")
        submitted = st.form_submit_button("접속")
    if submitted:
        if verify(password, config):
            st.session_state[SESSION_KEY] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False


# ---------------------------------------------------------------- 관리자 설정 UI
def render_admin_section() -> None:
    """관리자 메뉴 안에서 잠금 켜기/끄기·비밀번호 변경. 호출부는 ops_gate 통과 뒤여야 한다."""
    config = load_config()
    locked, pw_set = is_locked(config), has_password(config)

    st.caption(
        "사이트에 **들어오는 것 자체**를 비밀번호로 막습니다(관리자메뉴 잠금과는 별개). "
        f"설정은 `{CONFIG_FILE.name}` 에 저장되며 git 에 올라가지 않으므로 "
        "`git pull` 배포·서버 재시작 뒤에도 유지됩니다.")

    col_state, col_pw = st.columns(2)
    col_state.metric("현재 상태", "🔒 잠김" if locked else "🔓 열림")
    col_pw.metric("비밀번호", "설정됨" if pw_set else "미설정",
                  help=config.get("updated_at", "아직 지정한 적 없습니다"))

    with st.form("site_lock_admin_form"):
        enabled = st.checkbox("사이트 접속 잠금 사용", value=bool(config.get("enabled")))
        new_password = st.text_input(
            "새 비밀번호", type="password",
            placeholder="비워 두면 기존 비밀번호를 그대로 유지합니다")
        confirm_password = st.text_input("새 비밀번호 확인", type="password")
        saved = st.form_submit_button("저장", type="primary")

    if not saved:
        if pw_set:
            st.caption("비밀번호를 잊었다면 `site_lock.json` 의 `\"enabled\"` 를 "
                       "`false` 로 고치면 사이트가 열립니다. 그 뒤 여기서 다시 지정하세요.")
        return

    if new_password and new_password != confirm_password:
        st.error("두 비밀번호가 서로 다릅니다. 저장하지 않았습니다.")
        return
    if enabled and not new_password and not pw_set:
        st.error("비밀번호를 한 번도 설정하지 않았습니다. "
                 "잠금을 켜려면 먼저 새 비밀번호를 입력하세요.")
        return

    updated = dict(config)
    updated["enabled"] = bool(enabled)
    if new_password:
        updated.update(_password_fields(new_password))
    _write_config(updated)

    st.success(("잠금을 켰습니다 — 새 접속자는 비밀번호를 입력해야 합니다."
                if enabled else "잠금을 껐습니다 — 비밀번호 없이 접속할 수 있습니다.")
               + (" 비밀번호도 변경했습니다." if new_password else ""))
    # 방금 잠금을 켠(또는 비밀번호를 바꾼) 사람은 이미 관리자 인증을 통과했으므로
    # 이 세션은 그대로 통과시킨다 — 설정하자마자 자기가 튕겨 나가지 않게.
    st.session_state[SESSION_KEY] = True
