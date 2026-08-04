"""pages — Streamlit 화면 계층 전부 (조회·차트 헬퍼 + 페이지 렌더 + 화면 보조 모듈).

app.py 가 `st.navigation` 으로 진입점을 명시하므로 Streamlit 의 `pages/` 폴더
자동탐색은 꺼진다(navigation.py 가 `uses_pages_directory = False` 로 세팅) —
이 폴더에 헬퍼 모듈이 함께 있어도 사이드바에 페이지로 새지 않는다.

Streamlit 은 페이지를 **스크립트로** 실행하므로 sys.path[0] 은 저장소 루트다.
따라서 이 폴더 안에서 서로를 부를 때는 항상 패키지 import 를 쓴다:

    from pages import common as C
    from pages import chart_warn

무거운 계산(추론·수집)은 여기 두지 않는다 — common.run_script() 로 subprocess 실행.
"""
