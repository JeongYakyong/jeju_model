# -*- coding: utf-8 -*-
"""jeju_model 진입점 — 사이트 접속 잠금 게이트 + 페이지 내비게이션.

torch 등 무거운 import 금지 — 추론·수집은 전부 subprocess 로 돈다(관리자 메뉴·run_pipeline.py).
게이트 로직은 pages/site_lock.py 한 곳에 있다 — 켜기/끄기와 비밀번호는 관리자 메뉴에서
바꾸고 site_lock.json(git 제외)에 저장된다. 해제는 세션(브라우저) 단위.
(2026-09-07: secrets.toml 비밀번호 + .auth_token 6시간 파일토큰 방식을 대체.
 옛 방식은 토큰이 서버 파일이라 한 사람이 풀면 다른 브라우저·다른 사람도 6시간 통과됐다.)
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pages import common as C
from pages import site_lock

st.set_page_config(page_title="제주 순부하 예측 대시보드", page_icon="🍊",
                   layout="wide", initial_sidebar_state="expanded")

# ── 사이트 접속 잠금 (관리자 메뉴에서 켜기/끄기) ─────────────────────────
if not site_lock.gate():
    st.stop()

# ── 본문 ─────────────────────────────────────────────────────────────────
C.inject_style()
# st.navigation 을 쓰면 Streamlit 의 pages/ 폴더 자동탐색이 꺼진다 — pages/ 안의
# 헬퍼 모듈(common·chart_warn 등)이 사이드바에 페이지로 새지 않는다.
page = st.navigation([st.Page("pages/page_main.py", title="제주 순부하 예측", icon="🍊")])
page.run()
