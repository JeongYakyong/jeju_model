# -*- coding: utf-8 -*-
"""jeju_model 진입점 — 비밀번호 게이트(6시간 토큰) + 페이지 내비게이션.

torch 등 무거운 import 금지 — 추론·수집은 전부 subprocess 로 돈다(관리자 메뉴·run_pipeline.py).
게이트는 Model_api_added app.py 의 check_password/_token_valid/_write_token 이식:
비밀번호는 .streamlit/secrets.toml 의 password, 성공 시 파일 토큰(P.AUTH_TOKEN)을 남겨
6시간 동안은 새로고침·재접속해도 다시 묻지 않는다.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다
from pages import common as C

st.set_page_config(page_title="제주 순부하 예측 대시보드", page_icon="🍊",
                   layout="wide", initial_sidebar_state="expanded")

# ── 비밀번호 게이트 (6시간 파일 토큰) ────────────────────────────────────
AUTH_TTL_SECONDS = 6 * 3600


def _token_valid() -> bool:
    try:
        return (os.path.exists(P.AUTH_TOKEN)
                and (time.time() - os.path.getmtime(P.AUTH_TOKEN)) < AUTH_TTL_SECONDS)
    except Exception:
        return False


def _write_token():
    try:
        with open(P.AUTH_TOKEN, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    if _token_valid():
        st.session_state["authenticated"] = True
        return True
    st.title(" ")
    password = st.text_input("비밀번호를 입력하세요", type="password")
    if password:
        if password == st.secrets["password"]:
            st.session_state["authenticated"] = True
            _write_token()
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False


if not check_password():
    st.stop()

# ── 본문 ─────────────────────────────────────────────────────────────────
C.inject_style()
# st.navigation 을 쓰면 Streamlit 의 pages/ 폴더 자동탐색이 꺼진다 — pages/ 안의
# 헬퍼 모듈(common·chart_warn 등)이 사이드바에 페이지로 새지 않는다.
page = st.navigation([st.Page("pages/page_main.py", title="제주 순부하 예측", icon="🍊")])
page.run()
