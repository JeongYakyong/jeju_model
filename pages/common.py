# -*- coding: utf-8 -*-
"""jeju_model 공용 레이어 — DB 조회·KPX 실측 보강·데이터 현황·관리자 실행·차트/UI 헬퍼.

화면 계층(pages/)의 토대. 페이지 모듈은 `from pages import common as C` 로 부른다.
원본 forecastmodel/08_streamlit/common.py 에서 제주에 필요한 것만 남긴 트림판.
(land/도시가스/가스단가 계열 삭제, 경로는 새 폴더 구조의 project_paths 를 따른다.)
"""
from pathlib import Path
import importlib
import os
import subprocess
import sys
import sqlite3

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent   # pages/ 의 한 단계 위 = 저장소 루트
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

# .env 를 명시 경로로 로드 — cwd 가 어디든(streamlit/cron) OPS_PASSWORD·GEMINI_API_KEY·
# KPX 라이브 보강 키를 안정적으로 읽는다.
from dotenv import load_dotenv
load_dotenv(P.ENV_FILE)

# region 시그니처는 원본과 동일하게 유지(코드 이식 최소화) — 이 프로젝트는 jeju 뿐이다.
DB = {
    "jeju": Path(P.DB_JEJU),
}

CACHE_TTL = 600


# ---------------------------------------------------------------- 테마 (light/dark 겸용)
def theme_type() -> str:
    """현재 활성 테마 'light' | 'dark' — 설정 메뉴에서 바꾸면 rerun 때 st.context 로 감지."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 — 구버전/비-streamlit 실행은 라이트로 강등
        return "light"


# 차트 시리즈 팔레트 — 색·선 규약: 실측 = solid, 예측 = dot.
# dataviz 6-checks 검증기 통과값 (2026-07-17): 동시 표시되는 5색(demand·renew·net_load·rad·wind)이
# 라이트(#ffffff)·다크(#0e1117) 표면 모두에서 CVD 분리·명시야 분리·명도 밴드 PASS.
# smp 는 별도 차트에만 등장(풍력과 동시 표시 없음). reference = 발표값·참조선(중립 회색).
_CHART_PALETTES = {
    "light": {"demand": "#2a78d6", "renew": "#008300", "net_load": "#4a3aa7",
              "rad": "#eda100", "wind": "#1baf7a", "smp": "#e34948", "temp": "#e34948",
              "reference": "#898781", "alert": "#d03b3b"},
    "dark":  {"demand": "#3987e5", "renew": "#008300", "net_load": "#9085e9",
              "rad": "#c98500", "wind": "#199e70", "smp": "#e66767", "temp": "#e66767",
              "reference": "#898781", "alert": "#e66767"},
}
# 페이지 모듈은 C.COLOR["demand"] 처럼 읽는다 — inject_style() 이 매 rerun 활성 테마 값으로 갱신.
COLOR = dict(_CHART_PALETTES["light"])


# ---------------------------------------------------------------- 조회 레이어
@st.cache_data(ttl=CACHE_TTL)
def query(region: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    con = sqlite3.connect(str(DB[region]))
    try:
        df = pd.read_sql_query(sql, con, params=params, parse_dates=["timestamp"])
    finally:
        con.close()
    return df


@st.cache_data(ttl=CACHE_TTL)
def has_table(region: str, table: str) -> bool:
    """테이블 존재 여부 — 보조 테이블이 없는 DB도 안전하게 처리."""
    con = sqlite3.connect(str(DB[region]))
    try:
        r = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone()
        return r is not None
    finally:
        con.close()


# ---------------------------------------------------------------- 관리자 실행 (예측 체인·수집)
# 같은 인터프리터(sys.executable = 이 streamlit 의 venv)로 실행해 로컬·서버 동일 동작.
# 수집기의 load_dotenv() 는 파일 위치에서 상위 폴더로 올라가며 .env 를 찾으므로
# (collectors/ → 루트) cwd 와 무관하게 루트 .env 가 잡힌다.
COLLECT_HISTORICAL = Path(P.COLLECT_HISTORICAL)   # 실측 수집기 (collect_historical.py)
COLLECT_FORECAST = Path(P.COLLECT_FORECAST)       # 기상 예보 수집기 (--region jeju)
SERVE_CHAIN = Path(P.SERVE_CHAIN)                 # 수요+신재생 체인 → est_horizon_jeju
SERVE_SMP = Path(P.SERVE_SMP)                     # SMP → est_smp_horizon_jeju
OPS_PASSWORD = os.getenv("OPS_PASSWORD", "8888")  # .env 의 OPS_PASSWORD 로 지정(미설정 시 fallback)


def run_script(script: Path, args: list[str], timeout: int = 3600,
               cwd: Path | None = None) -> tuple[int, str]:
    """script 를 현재 인터프리터로 실행 → (returncode, stdout+stderr 합본).

    blocking — 호출부에서 st.spinner 로 감싼다.  cwd 미지정이면 ROOT.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=str(cwd or ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"[timeout {timeout}s] {e}"
    except Exception as e:  # noqa: BLE001
        return 1, f"[실행 실패] {e}"


def ops_gate() -> bool:
    """관리자메뉴 전체 잠금.  해제 상태면 True(본문 렌더 허용), 잠김이면 잠금화면만 그리고 False.

    호출부 맨 위에서:  ``if not C.ops_gate(): return``  — 해제 전에는 어떤 버튼도 그려지지 않는다.
    세션 단위(브라우저별)로 해제되며, 비밀번호는 OPS_PASSWORD(환경변수 권장).
    """
    if st.session_state.get("ops_unlocked"):
        c1, c2 = st.columns([5, 1])
        c1.success("🔓 관리자메뉴 — 잠금 해제됨")
        if c2.button("🔒 잠그기", key="ops_lock"):
            st.session_state["ops_unlocked"] = False
            st.rerun()
        return True
    st.markdown("### 🔒 관리자메뉴 — 잠김")
    st.caption("이 메뉴는 예측 실행과 데이터 수집을 **직접 수행**합니다. 비밀번호 입력 후 사용하세요.")
    with st.form("ops_gate_form"):
        pw = st.text_input("비밀번호", type="password")
        ok = st.form_submit_button("잠금 해제")
    if ok:
        if pw == OPS_PASSWORD:
            st.session_state["ops_unlocked"] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀립니다.")
    return False


# ---------------------------------------------------------------- 지평 아카이브 (basetime × horizon)
# 예측은 모두 est_horizon_jeju(tall: base 발행시각 × horizon_d 지평 × timestamp 목표시각)에서
# 읽는다. 구 forecast 테이블(timestamp 단일키 "최신 스냅샷")은 지평이 뭉개져 예측 소스로 안 쓴다.
def _hz_select(region: str, table: str, cols: list[str],
               mode: str, value, start: str, end: str,
               base_hour: str | None = None) -> pd.DataFrame:
    """지평 아카이브(base × horizon_d × timestamp)에서 예측 시계열을 뽑는 공용 SQL — 세 정리축.

    mode='latest' : 목표시각마다 '가장 최근 발행본'(가장 짧은 지평) — 운영 best(과거=사실상 익일).
    mode='asof'   : value=발행시각(base) 고정 → 그 발행본이 내다본 전 구간.
    mode='fixed'  : value=지평(정수 k) 고정 → 모든 목표를 '정확히 k일 전 발행'으로.
    base_hour('21'/'03')를 주면 그 발표만 — fixed 에서 12z/18z 가 같은 지평으로 겹치는
    중복을 막는다 (검증 페이지 발표 필터).  반환: timestamp + base + horizon_d + cols.
    """
    collist = ", ".join(cols)
    bh_sql = " AND substr(base, 12, 2) = ?" if base_hour else ""
    bh_par = (base_hour,) if base_hour else ()
    if mode == "asof":
        base = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        return query(region, f"SELECT timestamp, base, horizon_d, {collist} FROM {table} "
                             "WHERE base=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
                     (base, start, end))
    if mode == "fixed":
        return query(region, f"SELECT timestamp, base, horizon_d, {collist} FROM {table} "
                             f"WHERE horizon_d=? AND timestamp BETWEEN ? AND ?{bh_sql} "
                             "ORDER BY timestamp",
                     (int(value), start, end) + bh_par)
    # latest: 목표시각마다 최단 지평 우선, 동률이면 최신 발행(base DESC).
    # 12z/18z 이원화(2026-07-18)로 같은 시각·같은 horizon_d 인 두 발행본이 공존할 수
    # 있어(예: 내일 = 오늘 12z hd1 + 오늘 18z hd1) 단순 MIN(horizon_d) JOIN 은 중복행을
    # 낸다 — ROW_NUMBER 로 표시용 1건만 고른다.  DB 원본은 (base,timestamp) 로 양쪽 다
    # 보존되며(검증 페이지가 발표별로 각각 평가), 여기는 '표시 계층' 선택일 뿐이다.
    return query(region, f"SELECT timestamp, base, horizon_d, {collist} FROM ("
                         f"SELECT e.*, ROW_NUMBER() OVER (PARTITION BY e.timestamp "
                         "ORDER BY e.horizon_d ASC, e.base DESC) rn "
                         f"FROM {table} e WHERE e.timestamp BETWEEN ? AND ?) "
                         "WHERE rn = 1 ORDER BY timestamp",
                 (start, end))


# ---------------------------------------------------------------- 최근 실측 DB 채움 (갭필 후 저장)
# 원칙: DB에 현재 시각까지 데이터가 있으면 DB에서 읽고, 없으면(뒤처졌으면)
# 그 부족분만 KPX 라이브 수집 → historical 에 저장.  매번 전체를 다시 긁지 않는다.
# jeju = chejusukub 한 창구(수급+신재생 한 번에).
_LIVE_FETCH = {
    "jeju": ("real_demand_jeju", "kpx_asos", ("fetch_kpx_jeju",)),
}


@st.cache_data(ttl=300, show_spinner="최신 실측을 받아 DB에 채우는 중...")
def ensure_recent(region: str, day_str: str) -> int:
    """그 날 historical 이 '현재 가용 시각'까지 비었으면 KPX 부족분만 수집→DB 저장.

    DB가 이미 충분하면 수집하지 않고 0 반환(= DB에서 읽으면 됨).  5분 캐시로 호출 빈도 제한.
    미래일은 즉시 0.  표시 전용 라이브가 아니라 historical 에 partial_upsert 로 영속 저장한다.
    """
    demand_col, module, fn_names = _LIVE_FETCH[region]
    now = pd.Timestamp.now()
    day = pd.Timestamp(day_str).normalize()
    if day > now.normalize():
        return 0  # 미래 — 실측 없음
    # 그 날 채워져야 할 마지막 시각(오늘이면 직전 정시, 과거일이면 23시)
    last_needed = (now.floor("h") - pd.Timedelta(hours=1)) if day == now.normalize() \
        else day + pd.Timedelta(hours=23)
    with sqlite3.connect(str(DB[region])) as con:
        row = con.execute(
            f"SELECT MAX(timestamp) FROM historical WHERE timestamp BETWEEN ? AND ? "
            f"AND {demand_col} IS NOT NULL",
            (f"{day_str} 00:00:00", f"{day_str} 23:00:00")).fetchone()
    have = pd.Timestamp(row[0]) if row and row[0] else None
    if have is not None and have >= last_needed:
        return 0  # DB 충분 — 라이브 수집 안 함 (DB에서 읽는다)

    # 부족 → KPX 그 날치만 수집해 historical 에 partial upsert (영속)
    if P.DIR_COLLECTORS not in sys.path:
        sys.path.insert(0, P.DIR_COLLECTORS)
    try:
        mod = importlib.import_module(module)
        from kma_kimg import partial_upsert
    except Exception:
        return 0
    parts = []
    for name in fn_names:
        try:
            d = getattr(mod, name)(day_str, day_str, progress=False)
            if d is not None and not d.empty:
                if "timestamp" in d.columns:
                    d = d.set_index("timestamp")
                d.index = pd.to_datetime(d.index)
                parts.append(d)
        except Exception:
            pass
    if not parts:
        return 0
    wide = pd.concat(parts, axis=1)
    wide.index = wide.index.strftime("%Y-%m-%d %H:%M:%S")
    wide.index.name = "timestamp"
    n = partial_upsert("historical", wide, DB[region])
    query.clear()   # DB 갱신 → 조회 캐시 무효화
    return n


def clear_live_caches():
    ensure_recent.clear()
    query.clear()


# ---------------------------------------------------------------- 제주 지평 아카이브·비교
# 수요·신재생·순 부하 예측 = est_horizon_jeju, SMP = est_smp_horizon_jeju.
# 발표본(base) = 전일 밤 21:00(12z, horizon_d 1~5) + 당일 새벽 03:00(18z, horizon_d 0~2,
# 2026-07-18 도입).  SMP 는 12z 전용(horizon_d 1~2).
JEJU_HZ_TABLE = "est_horizon_jeju"
JEJU_SMP_TABLE = "est_smp_horizon_jeju"
JEJU_HZ_MAX = 5   # 운영 지평 상한 (2026-07-17 D+7→D+5 축소; 아카이브의 과거 D+6~7 행은 표시만 제외)
JEJU_EST_COLS = ["est_demand_jeju", "est_solar_gen_jeju", "est_wind_gen_jeju", "est_net_load_jeju",
                 "est_solar_util_jeju", "est_wind_util_jeju"]
JEJU_SMP_COLS = ["est_smp", "smp_neg_proba", "smp_danger"]

# ── 용어 SSOT (사용자 확정 2026-07-17): 화면에 D+N 표기 금지 —
# 지평은 당일/익일/모레/N일후, 발표는 배지("새벽 발표"/"전일 밤 발표")로 말한다.
BASE_HOUR_12Z = "21"   # 12z = 전일 밤 21시(KST) 발표
BASE_HOUR_18Z = "03"   # 18z = 당일 새벽 03시(KST) 발표 (당일예보, 2026-07-18 도입)


def hz_label(h: int) -> str:
    """지평 라벨 — 0=당일, 1=익일, 2=모레, n=n일후 (화면 D+N 금지 방침)."""
    return {0: "당일", 1: "익일", 2: "모레"}.get(int(h), f"{int(h)}일후")


def base_badge(base) -> str:
    """발표 배지 — base 시각으로 '새벽 발표'(18z=03시) / '전일 밤 발표'(12z=21시) 구분.

    발표 '주기'의 이름이다 (예측 검증 화면의 발표 필터 BASETIME_OPTS 와 같은 용어).
    "언제 만든 예측인가"에는 답하지 못하므로 그 용도로는 base_stamp 를 쓴다.
    """
    if base is None or (isinstance(base, float) and pd.isna(base)):
        return "—"
    b = pd.Timestamp(base)
    return "새벽 발표" if b.hour == 3 else "전일 밤 발표"


def base_stamp(base, ref_day=None) -> str:
    """발표 시각 표기 — '8/25 21시 발표 (어제)'.

    base_badge 가 주기 이름만 주는 것과 달리 **언제 만든 예측인지**를 실제 시각으로 답한다.
    ref_day(선택일)를 주면 그 날 기준 며칠 전 발표인지 괄호로 붙는다 — 같은 발표라도
    익일 예측이면 '(어제)', 5일후 예측이면 '(5일 전)'이 되어 리드타임이 드러나고,
    수집이 며칠 밀려 오래된 발표를 보고 있으면 그 자리에서 눈에 띈다.
    (구 base_badge 표기는 지평과 무관하게 늘 '전일 밤 발표'라 이 둘을 구분하지 못했다.)
    """
    if base is None or (isinstance(base, float) and pd.isna(base)):
        return "—"
    b = pd.Timestamp(base)
    stamp = f"{b.month}/{b.day} {b.hour}시 발표"
    if ref_day is None:
        return stamp
    days_ahead = (pd.Timestamp(ref_day).normalize() - b.normalize()).days
    if days_ahead < 0:          # 발표보다 이전 날짜 — 경과를 말할 수 없다
        return stamp
    ago = {0: "오늘", 1: "어제"}.get(days_ahead, f"{days_ahead}일 전")
    return f"{stamp} ({ago})"


@st.cache_data(ttl=CACHE_TTL)
def jeju_date_range() -> tuple[str, str]:
    """제주 예측 표시 가능 목표시각 범위 — est_horizon_jeju 기준."""
    df = query("jeju", f"SELECT MIN(timestamp) lo, MAX(timestamp) hi FROM {JEJU_HZ_TABLE} "
                       "WHERE est_demand_jeju IS NOT NULL")
    return str(df.loc[0, "lo"])[:10], str(df.loc[0, "hi"])[:10]


@st.cache_data(ttl=CACHE_TTL)
def jeju_horizon_range() -> tuple[int, int]:
    """제주 예측 지평 범위 — est_horizon_jeju 기준, 상한은 운영 지평 JEJU_HZ_MAX 로 클램프.

    하한도 1 로 클램프: 18z 당일예보(horizon_d=0, 2026-07-18 도입)가 아카이브에 있어도
    이 범위를 쓰는 위젯(익일 이후 지평 선택)에 0 이 새어들지 않게 한다 — 당일예보 지평은
    jeju_horizon_options('03') 로 따로 얻는다.
    """
    df = query("jeju", f"SELECT MIN(horizon_d) lo, MAX(horizon_d) hi FROM {JEJU_HZ_TABLE}")
    return max(int(df.loc[0, "lo"]), 1), min(int(df.loc[0, "hi"]), JEJU_HZ_MAX)


@st.cache_data(ttl=CACHE_TTL)
def jeju_horizon_options(base_hour: str) -> list[int]:
    """발표별 실보유 지평 목록 — base_hour '21'(전일 밤 12z) / '03'(새벽 18z 당일예보).

    est_horizon_jeju 에 실제로 있는 horizon_d 만 (JEJU_HZ_MAX 클램프) — 검증 페이지의
    발표 필터 선택지가 된다.  18z 도입 직후처럼 아직 행이 없으면 빈 목록.
    """
    df = query("jeju", f"SELECT DISTINCT horizon_d h FROM {JEJU_HZ_TABLE} "
                       "WHERE substr(base, 12, 2) = ? ORDER BY horizon_d", (base_hour,))
    return [int(h) for h in df["h"] if 0 <= int(h) <= JEJU_HZ_MAX]


def jeju_range_compare(start_day: pd.Timestamp, end_day: pd.Timestamp,
                       use_live: bool = True, mode: str = "latest", value=None,
                       base_hour: str | None = None) -> pd.DataFrame:
    """[start_day 00시, end_day 23시] 제주 MW 비교(수요·신재생·net_load) + 실측.

    mode/value = 예측 정리축(latest/asof/fixed). 종합 화면은 latest 로 선택일부터 D+1~D+k 창을 본다
    (목표시각마다 가장 최근 발행본 — 발행본 구멍에 강건, 미래 구간은 자연히 긴 지평으로 채워짐).
    신재생 예측 = 태양광+풍력 발전 예측 합. net_load 실측 = 수요−신재생(예측과 같은 기준 재구성).
    KPX 제주 하루전 수요예측(jeju_est_demand_da)은 비교용 참조로 그대로 싣는다.
    실측 보강 — 최근 구간이 DB에 비면 chejusukub 에서 부족분만 채운다(ensure_recent).
    SMP는 별도 jeju_smp_frame 으로 다룬다.
    """
    s = start_day.strftime("%Y-%m-%d 00:00:00")
    e = end_day.strftime("%Y-%m-%d 23:00:00")
    # DB가 최근 구간(오늘 포함 3일 이내)을 못 따라왔으면 그 부족분만 채워 최신화한 뒤 DB에서 읽는다.
    if use_live:
        today = pd.Timestamp.now().normalize()
        for day in pd.date_range(start_day.normalize(), end_day.normalize(), freq="D"):
            if 0 <= (today - day).days <= 3:
                ensure_recent("jeju", day.strftime("%Y-%m-%d"))
    base = pd.DataFrame({"timestamp": pd.date_range(s, e, freq="h")})
    est = _hz_select("jeju", JEJU_HZ_TABLE, JEJU_EST_COLS, mode, value, s, e, base_hour)
    act = query("jeju", "SELECT timestamp, real_demand_jeju, real_renew_gen_jeju, "
                        "real_solar_gen_jeju, real_wind_gen_jeju, jeju_est_demand_da "
                        "FROM historical WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp", (s, e))
    df = (base.merge(est, on="timestamp", how="left")
              .merge(act, on="timestamp", how="left"))
    df["est_renew_gen_jeju"] = df[["est_solar_gen_jeju", "est_wind_gen_jeju"]].sum(axis=1, min_count=1)
    df["real_net_load_jeju"] = df["real_demand_jeju"] - df["real_renew_gen_jeju"]
    return df


@st.cache_data(ttl=CACHE_TTL)
def jeju_smp_frame(start_day: pd.Timestamp, end_day: pd.Timestamp,
                   mode: str = "asof", value=None) -> pd.DataFrame:
    """[start_day 00시, end_day 23시] 제주 SMP 비교 — 예측(est_smp_horizon_jeju) + 하루전 SMP.

    참조선 = 하루전 SMP(smp_jeju_da). smp_danger = 음수가격 위험(0/1).
    종합 화면은 latest 로 선택일부터 48시간 창을 본다(SMP는 D+1·D+2만 발행).
    """
    s = start_day.strftime("%Y-%m-%d 00:00:00")
    e = end_day.strftime("%Y-%m-%d 23:00:00")
    base = pd.DataFrame({"timestamp": pd.date_range(s, e, freq="h")})
    est = _hz_select("jeju", JEJU_SMP_TABLE, JEJU_SMP_COLS, mode, value, s, e)[
        ["timestamp"] + JEJU_SMP_COLS]
    act = query("jeju", "SELECT timestamp, smp_jeju_da FROM historical "
                        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp", (s, e))
    return (base.merge(est, on="timestamp", how="left")
                .merge(act, on="timestamp", how="left"))


# 검증탭 4개 모델 — (표시 라벨, est 컬럼, 실측 컬럼, 지표종류, 설비용량 컬럼).  SMP는 검증탭에서 제외.
#   수요    = MAPE          (분모=실측수요, 크고 안정).
#   순 부하 = nMAE          (분모=평균 |net_load| — 설비용량이 없고 한낮 0 근처라 MAPE는 튐).
#   태양광·풍력 = capmae    (MAE ÷ 설비용량 — 신재생 예측의 표준 nMAE. 저발전 시간 분모 0 문제 회피).
JEJU_ACC_SPECS = [
    ("수요", "est_demand_jeju", "real_demand_jeju", "mape", None),
    ("순 부하", "est_net_load_jeju", "real_net_load_jeju", "nmae", None),
    ("태양광", "est_solar_gen_jeju", "real_solar_gen_jeju", "capmae", "real_solar_capacity_jeju"),
    ("풍력", "est_wind_gen_jeju", "real_wind_gen_jeju", "capmae", "real_wind_capacity_jeju"),
]
JEJU_ACC_MODELS = [s[0] for s in JEJU_ACC_SPECS]   # ["수요","순 부하","태양광","풍력"]


def _jeju_acc_join(start: str | None, end: str | None,
                   base_hour: str | None = None) -> pd.DataFrame:
    """검증용 est_horizon_jeju × historical 조인 (검증기간=실측 대상 timestamp). 파생 net_load·설비용량 포함.

    base_hour('21'=전일 밤 12z / '03'=새벽 18z)를 주면 그 발표의 est 만 —
    12z/18z 이원화(2026-07-18) 후 두 발표를 섞으면 같은 지평에 서로 다른 리드타임이
    합산되므로 검증은 발표별로 나눠 본다.  None = 전체(레거시).
    """
    where, params = [], []
    if start:
        where.append("e.timestamp >= ?"); params.append(f"{start} 00:00:00")
    if end:
        where.append("e.timestamp <= ?"); params.append(f"{end} 23:59:59")
    if base_hour:
        where.append("substr(e.base, 12, 2) = ?"); params.append(base_hour)
    wsql = ("WHERE " + " AND ".join(where)) if where else ""
    df = query("jeju", f"""
        SELECT e.timestamp, e.base, e.horizon_d AS hz,
               e.est_demand_jeju, e.est_net_load_jeju, e.est_solar_gen_jeju, e.est_wind_gen_jeju,
               h.real_demand_jeju, h.real_renew_gen_jeju, h.real_solar_gen_jeju, h.real_wind_gen_jeju,
               h.real_solar_capacity_jeju, h.real_wind_capacity_jeju
        FROM {JEJU_HZ_TABLE} e JOIN historical h ON e.timestamp = h.timestamp {wsql}
        ORDER BY e.timestamp
    """, tuple(params))
    if not df.empty:
        df["real_net_load_jeju"] = df["real_demand_jeju"] - df["real_renew_gen_jeju"]
    return df


def error_metrics(est: pd.Series, act: pd.Series) -> dict | None:
    """겹치는 시간만으로 MAPE(%)·MAE·bias(%). 겹침 없으면 None."""
    m = pd.concat([est, act], axis=1, keys=["e", "a"]).dropna()
    m = m[m["a"].abs() > 1e-6]
    if m.empty:
        return None
    err = m["e"] - m["a"]
    return {"mape": float((err.abs() / m["a"].abs()).mean() * 100),
            "nmae": float(err.abs().mean() / m["a"].abs().mean() * 100),  # 분모 작은 시간대에 강건
            "mae": float(err.abs().mean()),
            "bias": float(err.sum() / m["a"].sum() * 100),
            "n": len(m)}


def _acc_metric(g: pd.DataFrame, ec: str, ac: str, kind: str, cap_col):
    """(값%, 표본). capmae = MAE ÷ 설비용량(설비용량 기준 nMAE), 그 외 = error_metrics 의 mape/nmae.

    error_metrics 가 |실측|>1e-6 인 시간만 쓰므로 태양광 MAE 는 사실상 낮만 — 그 낮 MAE 를 설비용량으로 나눈다.
    """
    m = error_metrics(g[ec], g[ac])
    if not m:
        return None, 0
    if kind == "capmae":
        cap = g[cap_col].dropna().mean()
        if not cap or cap <= 0:
            return None, m["n"]
        return m["mae"] / cap * 100, m["n"]
    return m[kind], m["n"]


@st.cache_data(ttl=CACHE_TTL)
def jeju_horizon_accuracy(start: str | None = None, end: str | None = None,
                          base_hour: str | None = None) -> pd.DataFrame:
    """제주 4개 모델의 지평별 정확도 — 검증기간(실측 대상일 = 타깃 timestamp) 안에서 집계.

    수요 = MAPE, 순 부하 = nMAE, 태양광·풍력 = 설비용량 기준 nMAE(MAE÷설비용량), 단위 %.
    est_horizon_jeju × historical 실측. 미래 대상일·실측 없는 칸은 자동 제외.
    base_hour '03'(새벽 18z)이면 당일(0)부터, '21'/None 이면 익일(1)부터.
    반환: index=지평 라벨(당일/익일/모레/N일후), 컬럼=hz(정수)+모델별 지표(%)+표본(시간).
    """
    mw = _jeju_acc_join(start, end, base_hour)
    if mw.empty:
        return pd.DataFrame()
    lo = 0 if base_hour == BASE_HOUR_18Z else 1
    rows = []
    for hz in range(lo, min(int(mw["hz"].max()), JEJU_HZ_MAX) + 1):
        g = mw[mw["hz"] == hz]
        rec, n = {"지평": hz_label(hz), "hz": hz}, 0
        for label, ec, ac, kind, cap_col in JEJU_ACC_SPECS:
            v, nn = _acc_metric(g, ec, ac, kind, cap_col)
            rec[label] = round(v, 1) if v is not None else None
            n = max(n, nn)
        rec["표본"] = n
        # 실측 평균 리드타임(발표→대상 시각) — 검증탭 '리드타임(h) 축'용.  12z/18z 를
        # 한 축에서 비교할 때 지평 정수 대신 실제 신선도를 보여준다.
        lead = (g["timestamp"] - pd.to_datetime(g["base"])).dt.total_seconds() / 3600
        rec["리드(h)"] = round(float(lead.mean()), 1) if len(lead) else None
        rows.append(rec)
    return pd.DataFrame(rows).set_index("지평")


@st.cache_data(ttl=CACHE_TTL)
def jeju_daily_accuracy(start: str, end: str | None, k: int,
                        base_hour: str | None = None) -> pd.DataFrame:
    """선택 지평 k에서 검증기간의 '일별' 정확도 추이 — index=날짜, 컬럼=모델별 지표(%).

    수요 = MAPE, 순 부하 = nMAE, 태양광·풍력 = 설비용량 기준 nMAE. 하루 표본 < 8이면 그 날 그 모델은 NaN(낮만인 태양광 보호).
    맑은 날 정확하다가 특정일 급등 = 그 날 기상 급변(비·구름) 신호로 읽는다.
    base_hour = 발표 필터('21' 전일 밤 / '03' 새벽) — jeju_horizon_accuracy 와 동일.
    """
    mw = _jeju_acc_join(start, end, base_hour)
    if mw.empty:
        return pd.DataFrame()
    mw = mw[mw["hz"] == k]
    if mw.empty:
        return pd.DataFrame()
    grp = mw.groupby(mw["timestamp"].dt.date)
    out = {}
    for label, ec, ac, kind, cap_col in JEJU_ACC_SPECS:
        def _daily(g, ec=ec, ac=ac, kind=kind, cap_col=cap_col):
            v, nn = _acc_metric(g, ec, ac, kind, cap_col)
            return v if (v is not None and nn >= 8) else float("nan")
        out[label] = grp.apply(_daily, include_groups=False)
    res = pd.DataFrame(out)
    res.index = pd.to_datetime(res.index)
    return res


@st.cache_data(ttl=CACHE_TTL)
def jeju_renew_capacity() -> tuple[float, float]:
    """제주 태양광·풍력 설비용량(MW) — historical 최신값. 합산 신재생 이용률 계산용."""
    df = query("jeju", "SELECT real_solar_capacity_jeju s, real_wind_capacity_jeju w FROM historical "
                       "WHERE real_solar_capacity_jeju IS NOT NULL ORDER BY timestamp DESC LIMIT 1")
    if df.empty:
        return float("nan"), float("nan")
    return float(df.loc[0, "s"]), float(df.loc[0, "w"])


# 데이터 현황 히트맵 항목 — (라벨, 테이블, 컬럼, 장지평여부).
# 장지평(D+3 이상) 수집 가능=예보(green) / 불가=실측·근일(blue). SMP는 D+1·D+2뿐이라 장지평 불가(blue).
JEJU_COVERAGE_ITEMS = [
    ("수요 예측", "est_horizon_jeju", "est_demand_jeju", True),
    ("순 부하 예측", "est_horizon_jeju", "est_net_load_jeju", True),
    ("태양광 예측", "est_horizon_jeju", "est_solar_gen_jeju", True),
    ("풍력 예측", "est_horizon_jeju", "est_wind_gen_jeju", True),
    ("기상 예보", "forecast_horizon", "temp_west", True),
    ("SMP 예측", "est_smp_horizon_jeju", "est_smp", False),
    ("하루전 SMP", "historical", "smp_jeju_da", False),
    ("수요 실측", "historical", "real_demand_jeju", False),
    ("신재생 실측", "historical", "real_renew_gen_jeju", False),
]


@st.cache_data(ttl=CACHE_TTL)
def jeju_coverage_daily(days_back: int = 30, days_fwd: int = 7) -> pd.DataFrame:
    """제주 주요 항목의 일별 적재율(0~1) — index=항목, columns=날짜(MM-DD). 데이터 현황 히트맵용.

    하루 = 24시간 기준 채워진 비율. 예측 tall 테이블(est_horizon_*)은 목표시각 distinct 기준
    (어느 발행본이든 그 시각을 덮으면 셈) — 발행본별이 아니라 '그 시각 예측이 있나'를 본다.
    """
    today = pd.Timestamp.now().normalize()
    lo, hi = today - pd.Timedelta(days=days_back), today + pd.Timedelta(days=days_fwd)
    s, e = lo.strftime("%Y-%m-%d 00:00:00"), hi.strftime("%Y-%m-%d 23:00:00")
    dates = pd.date_range(lo, hi, freq="D")
    out = {}
    for label, table, col, _lh in JEJU_COVERAGE_ITEMS:
        if not has_table("jeju", table) or col not in table_columns("jeju", table):
            out[label] = [0.0] * len(dates)
            continue
        df = query("jeju", f"SELECT substr(timestamp, 1, 10) d, COUNT(DISTINCT timestamp) n "
                           f"FROM {table} WHERE {col} IS NOT NULL AND timestamp BETWEEN ? AND ? "
                           "GROUP BY d", (s, e))
        m = dict(zip(df["d"], df["n"]))
        out[label] = [min(1.0, m.get(d.strftime("%Y-%m-%d"), 0) / 24) for d in dates]
    return pd.DataFrame(out, index=[d.strftime("%m-%d") for d in dates]).T


# ---------------------------------------------------------------- 데이터 현황
# 화면 표기용 테이블 이름 — 값(실제 테이블명)은 그대로 두고 표시만 바꾼다.
TABLE_LABEL = {
    "historical": "실측 기록",
    "forecast_horizon": "기상 예보 아카이브",
    "est_horizon_jeju": "예측 아카이브",
    "est_smp_horizon_jeju": "SMP 예측 아카이브",
}

COVERAGE = {
    "jeju": [
        ("est_horizon_jeju", "수요 예측", "est_demand_jeju"),
        ("est_horizon_jeju", "순 부하 예측", "est_net_load_jeju"),
        ("est_horizon_jeju", "태양광 이용률 예측", "est_solar_util_jeju"),
        ("est_horizon_jeju", "풍력 이용률 예측", "est_wind_util_jeju"),
        ("est_smp_horizon_jeju", "SMP 예측", "est_smp"),
        ("est_smp_horizon_jeju", "SMP 음수가격 경보", "smp_danger"),
        ("forecast_horizon", "기상 예보(서부 기온)", "temp_west"),
        ("historical", "수요 실측(KPX)", "real_demand_jeju"),
        ("historical", "신재생 실측(KPX)", "real_renew_gen_jeju"),
        ("historical", "순 부하 실측", "real_net_load_jeju"),
        ("historical", "실시간 SMP", "smp_jeju_rt"),
    ],
}


@st.cache_data(ttl=CACHE_TTL)
def table_columns(region: str, table: str) -> list[str]:
    con = sqlite3.connect(str(DB[region]))
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


@st.cache_data(ttl=CACHE_TTL)
def table_range(region: str, table: str) -> tuple[str, str]:
    df = query(region, f"SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM {table}")
    return str(df.loc[0, "lo"]), str(df.loc[0, "hi"])


@st.cache_data(ttl=CACHE_TTL)
def coverage_heat(region: str, table: str, start: str, end: str) -> pd.DataFrame:
    """6시간 블록별 컬럼 적재율(0~1) — index=컬럼(DB 순서), columns=블록 시작 시각.

    기간 전체를 시간 격자로 reindex — 행 자체가 없는 구간도 0%로 드러난다.
    """
    df = query(region, f"SELECT * FROM {table} WHERE timestamp BETWEEN ? AND ? "
                       "ORDER BY timestamp", (start, end))
    grid = pd.date_range(start, end, freq="h")
    if df.empty:
        return pd.DataFrame(0.0, index=[c for c in table_columns(region, table)
                                        if c != "timestamp"],
                            columns=pd.date_range(start, end, freq="6h"))
    df = df.set_index("timestamp").reindex(grid)
    return df.notna().resample("6h").mean().T


@st.cache_data(ttl=CACHE_TTL)
def coverage_table(region: str) -> pd.DataFrame:
    rows, now = [], pd.Timestamp.now()
    con = sqlite3.connect(str(DB[region]))
    try:
        tables = {t for t, _, _ in COVERAGE[region]}
        have = {t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")} for t in tables}
        for table, label, col in COVERAGE[region]:
            if col not in have[table]:
                rows.append([TABLE_LABEL.get(table, table), label, "—", "—", 0, None]); continue
            lo, hi, n = con.execute(
                f"SELECT MIN(timestamp), MAX(timestamp), COUNT({col}) "
                f"FROM {table} WHERE {col} IS NOT NULL").fetchone()
            lag = round((now - pd.Timestamp(hi)).total_seconds() / 3600, 1) if hi else None
            rows.append([TABLE_LABEL.get(table, table), label,
                         (lo or "—")[:16], (hi or "—")[:16], n, lag])
    finally:
        con.close()
    return pd.DataFrame(rows, columns=["저장소", "항목", "시작", "마지막 저장", "행수", "경과(시간)"])


# ---------------------------------------------------------------- 차트 헬퍼
# 전 차트 공용 템플릿 — 라이트/다크 한 쌍을 등록하고, inject_style() 이 매 rerun 활성 테마를
# pio 기본값으로 지정한다 (make_fig 외 go.Figure 직접 생성에도 일괄 적용).
def _make_template(font_color, grid, line, tick, legend_color, hover_bg, hover_ink):
    return go.layout.Template(layout=go.Layout(
        font=dict(family="Pretendard, 'Segoe UI', sans-serif", size=13, color=font_color),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=grid, linecolor=line, zerolinecolor=line,
                   tickfont=dict(size=11.5, color=tick)),
        yaxis=dict(gridcolor=grid, linecolor=line, zerolinecolor=line,
                   tickfont=dict(size=11.5, color=tick),
                   title=dict(font=dict(size=12, color=tick))),
        legend=dict(font=dict(size=12, color=legend_color)),
        hoverlabel=dict(bgcolor=hover_bg, bordercolor="rgba(0,0,0,0)",
                        font=dict(family="Pretendard, 'Segoe UI', sans-serif",
                                  size=12.5, color=hover_ink)),
    ))


pio.templates["briefing_light"] = _make_template(
    font_color="#334155", grid="#eef2f7", line="#e2e8f0", tick="#64748b",
    legend_color="#475569", hover_bg="#0f172a", hover_ink="#f1f5f9")
pio.templates["briefing_dark"] = _make_template(
    font_color="#c3cad6", grid="#232b3b", line="#333c4f", tick="#8b95a8",
    legend_color="#a8b3c5", hover_bg="#f1f5f9", hover_ink="#0f172a")
pio.templates.default = "briefing_light"


def make_fig(height: int = 420, ytitle: str = "MW") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=-0.15), yaxis_title=ytitle)
    return fig


# 시간당 점을 둥근 곡선으로 잇는 정도(0~1.3). 시각화 전용 — 직선 이음의 각진 느낌을 부드럽게.
# 클수록 더 둥글지만 급변 구간(태양광 일출 등)에서 곡선이 0 아래로 살짝 휠 수 있어 보수적으로 0.6.
LINE_SMOOTHING = 0.6


def add_actual(fig: go.Figure, ts, y, name: str, color: str, **kw):
    fig.add_trace(go.Scatter(x=ts, y=y, name=name,
                             line=dict(color=color, width=2, shape="spline",
                                       smoothing=LINE_SMOOTHING), **kw))


def add_forecast(fig: go.Figure, ts, y, name: str, color: str, **kw):
    fig.add_trace(go.Scatter(x=ts, y=y, name=name,
                             line=dict(color=color, dash="dot", width=2, shape="spline",
                                       smoothing=LINE_SMOOTHING), **kw))


def hz_hover(df: pd.DataFrame):
    """예측 트레이스용 (customdata, hovertemplate) — 커서에 발표 배지·지평 라벨 표시.

    df는 jeju_range_compare/_hz_select 산출(컬럼 base·horizon_d 포함)을 가정.
    표기 = "{발표일} {새벽|밤} 발표 · {당일/익일/모레/N일후}" (화면 D+N 금지 방침).
    """
    b = df["base"] if "base" in df.columns else [None] * len(df)
    h = df["horizon_d"] if "horizon_d" in df.columns else [None] * len(df)
    cd = []
    for bb, hh in zip(b, h):
        if pd.isna(bb) or pd.isna(hh):
            cd.append(["—"])
        else:
            ts = pd.Timestamp(bb)
            word = "새벽" if ts.hour == 3 else "밤"
            cd.append([f"{ts:%m-%d} {word} 발표 · {hz_label(hh)}"])
    tmpl = ("%{x|%m-%d %H시} · %{y:,.0f} MW<br>%{customdata[0]}"
            "<extra>%{fullData.name}</extra>")
    return cd, tmpl


# ---------------------------------------------------------------- UI 레이어
# 디자인 토큰 — 라이트/다크 한 쌍. 캔버스·사이드바 기본색은 .streamlit/config.toml(라이트 커스텀
# 테마)이 담당하고, 설정 메뉴에서 Dark 를 고르면 st.context 감지로 아래 다크 토큰이 적용된다.
_UI_TOKENS = {
    "light": dict(CARD="#ffffff", BORDER="#cbd5e1", INK="#0f172a", SUB="#64748b",
                  ACCENT="#059669", BRIEF_BG="#f4faf7", BRIEF_BORDER="#d7e6dd",
                  TAB_BORDER="#94a3b8", TAB_ACTIVE="#0f172a", TAB_ACTIVE_TEXT="#ffffff",
                  SHADOW="0 1px 2px rgba(15,23,42,.05)"),
    "dark":  dict(CARD="#1b2130", BORDER="#333c4f", INK="#e8edf5", SUB="#94a3b8",
                  ACCENT="#34d399", BRIEF_BG="#12261e", BRIEF_BORDER="#1e4034",
                  TAB_BORDER="#475569", TAB_ACTIVE="#e2e8f0", TAB_ACTIVE_TEXT="#0f172a",
                  SHADOW="none"),
}

_CSS_TEMPLATE = """
<style>
/* 폰트 — 커스텀 테마 밖(내장 Dark)에서도 Pretendard 유지 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

/* ---- 지표 카드 ---- */
[data-testid="stMetric"]{
  background:__CARD__; border:1px solid __BORDER__; border-radius:14px;
  padding:.85rem 1.05rem .8rem; box-shadow:__SHADOW__; }
[data-testid="stMetricLabel"] p{
  font-size:.78rem; font-weight:700; color:__SUB__; letter-spacing:.01em; }
[data-testid="stMetricValue"]{
  font-family:'IBM Plex Mono','Pretendard',monospace; font-size:1.5rem;
  font-weight:600; color:__INK__; letter-spacing:-.03em; }
[data-testid="stMetricDelta"]{ font-size:.78rem; }

/* ---- 차트·지도 임베드를 카드로 ---- */
[data-testid="stPlotlyChart"]{
  background:__CARD__; border:1px solid __BORDER__; border-radius:14px;
  padding:.6rem .7rem .3rem; box-shadow:__SHADOW__; }
[data-testid="stIFrame"], iframe[title="st.iframe"]{
  border:1px solid __BORDER__; border-radius:14px;
  box-shadow:__SHADOW__; }

/* ---- AI 브리핑 카드 — 연한 톤 + 좌측 강조선, 글머리별 줄바꿈 ---- */
.brief-card{
  background:__BRIEF_BG__; border:1px solid __BRIEF_BORDER__; border-left:3px solid __ACCENT__;
  border-radius:12px; padding:.75rem .95rem; margin:.15rem 0 .5rem;
  box-shadow:__SHADOW__; }
.brief-card .bi{
  position:relative; padding-left:1.05rem; margin:.34rem 0; line-height:1.62;
  color:__INK__; font-size:.93rem; }
.brief-card .bi:before{
  content:"•"; position:absolute; left:.1rem; top:-.02rem;
  color:__ACCENT__; font-weight:700; }
.brief-card .bi:first-child{ margin-top:.05rem; }
.brief-card .bi:last-child{ margin-bottom:.05rem; }

/* ---- 탭 — 알약 버튼(선택 가능해 보이게, 지도 패널 mchip 과 동일 문법) ---- */
[data-testid="stTabs"] [role="tablist"]{ gap:6px; border-bottom:none; }
[data-testid="stTabs"] [role="tab"]{
  background:__CARD__; border:1px solid __TAB_BORDER__; border-radius:999px;
  padding:.15rem 1.05rem; margin-bottom:6px; transition:border-color .12s; }
[data-testid="stTabs"] [role="tab"]:hover{ border-color:__SUB__; }
[data-testid="stTabs"] [role="tab"] p{ font-size:.9rem; font-weight:700; color:__SUB__; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  background:__TAB_ACTIVE__; border-color:__TAB_ACTIVE__; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p{ color:__TAB_ACTIVE_TEXT__; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ display:none; }
[data-testid="stTabs"] [data-baseweb="tab-border"]{ display:none; }

/* ---- 작은 지표(보조 수치): container(key=*_metric_sm) 로 감싸 적용 ---- */
[class*="metric_sm"] [data-testid="stMetricValue"]{ font-size:1.05rem; }
[class*="metric_sm"] [data-testid="stMetric"]{ padding:.7rem .9rem; }

/* ---- 슬라이더 라벨 — ? 도움말 아이콘을 오른쪽 끝이 아니라 라벨 글자 바로 옆에 ---- */
[data-testid="stSlider"] [data-testid="stWidgetLabel"]{ width:fit-content !important; }

/* ---- 검증 '표시 구간' pill — 줄바꿈 없이 한 줄 균등 배치 ---- */
.st-key-fm_win_box [data-testid="stElementContainer"],
.st-key-fm_win_box [data-testid="stButtonGroup"]{ width:100% !important; }
.st-key-fm_win_box [data-baseweb="button-group"]{
  display:flex !important; flex-wrap:nowrap !important; gap:4px; width:100%; }
.st-key-fm_win_box [data-baseweb="button-group"] button{
  flex:1 1 0; min-width:0; padding:.15rem .3rem;
  justify-content:center; white-space:nowrap; }

/* ---- 버튼·캡션 ---- */
.stButton button p{ font-weight:700; font-size:.88rem; }
[data-testid="stCaptionContainer"]{ color:__SUB__; }

/* ---- date_input: 박스 안 날짜 중앙 정렬 (전 페이지 공통) ---- */
.stDateInput input{ text-align:center; }

/* ---- 사이드바: radio 를 내비게이션 메뉴처럼 (사이드바는 양 테마 모두 다크 잉크) ---- */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
  font-size:.7rem; font-weight:800; letter-spacing:.16em; color:#94a3b8;
  text-transform:uppercase; }
section[data-testid="stSidebar"] [role="radiogroup"]{ gap:3px; }
section[data-testid="stSidebar"] [role="radiogroup"] label{
  width:100%; margin:0; padding:.5rem .8rem; border-radius:10px;
  transition:background .12s; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:rgba(148,163,184,.14); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:rgba(52,211,153,.16); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{
  color:#6ee7b7; font-weight:800; }
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{
  display:none; }   /* 라디오 동그라미 숨김 — 메뉴처럼 보이게 */

/* ---- 페이지 헤더 ---- */
.bf-head{ margin:0 0 .9rem; }
.bf-eyebrow{ font-size:.7rem; font-weight:800; letter-spacing:.2em; color:__ACCENT__; }
.bf-titlerow{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin:.15rem 0 .3rem; }
.bf-title{ font-size:1.85rem; font-weight:800; letter-spacing:-.02em;
  color:__INK__; line-height:1.15; }
.bf-chain{ display:inline-flex; align-items:center; gap:.45rem; background:__CARD__;
  border:1px solid __BORDER__; border-radius:999px; padding:.38rem .9rem;
  box-shadow:__SHADOW__; }
.bf-step{ display:inline-flex; align-items:center; gap:.34rem; font-size:.8rem;
  font-weight:700; color:__INK__; white-space:nowrap; }
.bf-step i{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.bf-arrow{ color:__SUB__; font-size:.78rem; }
.bf-sub{ font-size:.88rem; color:__SUB__; }
</style>"""


def _build_css(mode: str) -> str:
    css = _CSS_TEMPLATE
    for key, value in _UI_TOKENS[mode].items():
        css = css.replace(f"__{key}__", value)
    return css


def inject_style():
    """전역 CSS + 테마 동기화 — app.py(엔트리)에서 매 rerun마다 호출(전 페이지 공통).

    활성 테마(light/dark)에 맞춰 ① CSS 토큰 ② 차트 팔레트(COLOR) ③ plotly 기본 템플릿을
    한 번에 갱신한다 — 페이지 코드는 C.COLOR / make_fig 를 그대로 쓰면 된다.
    """
    mode = theme_type()
    COLOR.clear()
    COLOR.update(_CHART_PALETTES[mode])
    pio.templates.default = f"briefing_{mode}"
    st.markdown(_build_css(mode), unsafe_allow_html=True)


def day_navigator(prefix: str, ndays: tuple[int, int, int] | None = None,
                  refresh: bool = True):
    """표준 날짜 컨트롤 — ◀ 어제 | 날짜 | 내일 ▶ | (새로고침) | (표시 기간) | 캡션.

    탭/메뉴마다 독립 배치(prefix 별 session 키). ndays=(최소, 최대, 기본)이면
    표시 기간(일) 슬라이더 포함, refresh=False면 새로고침 버튼 없는 슬림 버전.
    반환: (선택일 Timestamp, 표시일수 n | None, 캡션용 column). 기본 = 오늘.
    """
    key = f"{prefix}_day"
    if key not in st.session_state:
        st.session_state[key] = pd.Timestamp.now().normalize().date()

    def _shift(delta: int):
        st.session_state[key] = st.session_state[key] + pd.Timedelta(days=delta)

    ratios = [0.8, 1.6, 0.8]
    if refresh:
        ratios.append(1.5)
    if ndays:
        ratios.append(2.1)
    ratios.append(2.5 if ndays else 4.6)
    cols = st.columns(ratios, vertical_alignment="center")
    cols[0].button("◀ 어제", key=f"{prefix}_prev", on_click=_shift, args=(-1,), width="stretch")
    cols[1].date_input("날짜", key=key, label_visibility="collapsed")
    cols[2].button("내일 ▶", key=f"{prefix}_next", on_click=_shift, args=(1,), width="stretch")
    i = 3
    if refresh:
        if cols[i].button("실시간 새로고침", key=f"{prefix}_refresh", width="stretch",
                          help="최근 구간 실측(KPX 수급)을 다시 불러옵니다"):
            clear_live_caches()
        i += 1
    n = None
    if ndays:
        n = cols[i].slider("표시 기간(일)", ndays[0], ndays[1], ndays[2],
                           key=f"{prefix}_ndays",
                           help="시작일부터 N일치를 표시합니다")
    return pd.Timestamp(st.session_state[key]), n, cols[-1]


def help_expander(md: str, title: str = "도움말"):
    """화면 하단 도움말 — 접힌 expander에 상세 설명(markdown)."""
    with st.expander(title, expanded=False):
        st.markdown(md)


def page_header(eyebrow: str, title: str, sub: str, chain: list[tuple[str, str]]):
    """페이지 헤더 — eyebrow + 제목 + 체인 pill(점 색 = 차트 COLOR 규약과 동일). sub 빈 문자열이면 생략."""
    steps = '<span class="bf-arrow">→</span>'.join(
        f'<span class="bf-step"><i style="background:{c}"></i>{label}</span>'
        for label, c in chain)
    sub_html = f'<div class="bf-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="bf-head"><div class="bf-eyebrow">{eyebrow}</div>'
        f'<div class="bf-titlerow"><div class="bf-title">{title}</div>'
        f'<div class="bf-chain">{steps}</div></div>'
        f'{sub_html}</div>', unsafe_allow_html=True)
