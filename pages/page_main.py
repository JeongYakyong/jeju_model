# -*- coding: utf-8 -*-
"""제주 순부하 예측 대시보드 — 메인 페이지 (사이드바 5메뉴).

  종합       : 3존 기상 지도(hero) + AI 브리핑 + 핵심 지표   ← 지도는 M5, 브리핑은 M7에서 추가
  예측 확인  : 선택일부터 표시 기간(k일) 수요·신재생·순 부하 차트 + SMP + 시간별 표  ← 위험구간 음영은 M6
  예측 검증  : 모델 정확도 3탭 (일별 추이 · 지평별 성능 · 지정일 시계열)
  데이터 현황: 항목×날짜 수집률 히트맵
  관리자     : 수집·예측 실행 (비밀번호 게이트)              ← 원클릭 전체 실행은 M8

(원본: forecastmodel/08_streamlit/page_jeju.py 를 5메뉴 구조로 확장)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent   # pages/ 의 한 단계 위 = 저장소 루트
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다
from pages import common as C
from pages import chart_warn

C.page_header(
    "JEJU · NETLOAD", "제주 순부하 예측 대시보드", "",
    [("수요", C.COLOR["demand"]), ("신재생", C.COLOR["renew"]),
     ("순 부하", C.COLOR["net_load"]), ("SMP", C.COLOR["smp"])])
menu = st.sidebar.radio("메뉴", ["종합", "예측 확인", "예측 검증", "데이터 현황", "관리자"])
TODAY = pd.Timestamp.now().normalize()


# 지표 카드 공용 스타일 — 라벨·값·보조줄 전부 가운데 정렬 + delta 화살표·배경 제거(담백하게).
_METRIC_CSS = """<style>
[data-testid="stMetric"]{ text-align:center !important; }
[data-testid="stMetric"] [data-testid="stMarkdownContainer"],
[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p{
    text-align:center !important; width:100% !important; }
[data-testid="stMetricLabel"]{ display:flex !important; justify-content:center !important; width:100% !important; }
[data-testid="stMetricValue"]{ justify-content:center !important; }
[data-testid="stMetricDelta"]{ display:flex !important; justify-content:center !important;
    width:100% !important; background:none !important; padding:0 !important; }
[data-testid="stMetricDelta"] svg{ display:none !important; }
</style>"""


def inject_metric_style():
    st.markdown(_METRIC_CSS, unsafe_allow_html=True)


def _when(ts_col: pd.Series, series: pd.Series, how: str):
    """series 최대/최소가 발생한 시점 'MM-DD HH시' (없으면 None)."""
    s = series.dropna()
    if s.empty:
        return None
    i = s.idxmax() if how == "max" else s.idxmin()
    return f"{ts_col.loc[i]:%m-%d %H시}"


# =============================================================================
# 종합 — hero 지도(M5) + AI 브리핑(M7) + 핵심 지표
# =============================================================================
def render_home():
    import streamlit.components.v1 as components
    from pages import weather_map_jeju

    day, _, _cap = C.day_navigator("home", refresh=True)

    # hero — 제주 3구역 기상 지도 (Leaflet HTML 임베드, 좌우 패널 포함)
    components.html(weather_map_jeju.build_day_html(day), height=620)

    # AI 브리핑 — 생성/갱신 버튼, ai_briefings.db 저장
    from pages import brief_jeju
    brief_jeju.render_briefing_expander(day)

    df = C.jeju_range_compare(day, day + pd.Timedelta(days=1), mode="latest")
    smp = C.jeju_smp_frame(day, day + pd.Timedelta(days=1), mode="latest")
    if df["est_net_load_jeju"].isna().all():
        lo, hi = C.jeju_date_range()
        st.warning(f"선택한 날짜의 예측이 아직 없습니다. 예측 보유 범위: **{lo} ~ {hi}**.")
        return

    inject_metric_style()
    nl, ts_col = df["est_net_load_jeju"], df["timestamp"]
    danger_hours = int(smp["smp_danger"].fillna(0).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("최대 순 부하(예측·48h)", "—" if nl.isna().all() else f"{nl.max():,.0f} MW",
              _when(ts_col, nl, "max"), delta_color="off")
    m2.metric("최저 순 부하(예측·48h)", "—" if nl.isna().all() else f"{nl.min():,.0f} MW",
              _when(ts_col, nl, "min"), delta_color="off")
    m3.metric("최저 SMP(예측·48h)",
              "—" if smp["est_smp"].isna().all() else f"{smp['est_smp'].min():,.1f} 원/kWh",
              _when(smp["timestamp"], smp["est_smp"], "min"), delta_color="off")
    m4.metric("SMP 음수가격 경보", f"{danger_hours} 시간",
              "48시간 창 기준", delta_color="off")
    # 선택일 구간의 대표 발표 배지 — 시각별 최신 발표본이라 하루 안에서도 섞일 수 있어 최빈값.
    day_bases = df.loc[df["timestamp"].dt.normalize() == day.normalize(), "base"].dropna()
    badge = f" · 선택일 예보 = **{C.base_badge(day_bases.mode().iloc[0])}**" if len(day_bases) else ""
    st.caption(f"선택일부터 48시간 창 · 예측 = 목표시각마다 가장 최근 발표본{badge}")


# =============================================================================
# 예측 확인 — 선택일부터 k일 차트 + 지표 + SMP + 시간별 표
# =============================================================================
# ⚙️ 선택형 시리즈 — (라벨, 컬럼, 종류, 색, 기본 선택). 태양광·풍력은 실측·예측 둘 다 선택 가능.
JEJU_SERIES = [
    ("전력수요 실측", "real_demand_jeju", "act", C.COLOR["demand"], True),
    ("전력수요 예측", "est_demand_jeju", "est", C.COLOR["demand"], True),
    ("신재생 실측", "real_renew_gen_jeju", "act", C.COLOR["renew"], True),
    ("신재생 예측", "est_renew_gen_jeju", "est", C.COLOR["renew"], True),
    ("순 부하 실측", "real_net_load_jeju", "act", C.COLOR["net_load"], True),
    ("순 부하 예측", "est_net_load_jeju", "est", C.COLOR["net_load"], True),
    ("태양광 실측", "real_solar_gen_jeju", "act", C.COLOR["rad"], False),
    ("태양광 예측", "est_solar_gen_jeju", "est", C.COLOR["rad"], False),
    ("풍력 실측", "real_wind_gen_jeju", "act", C.COLOR["wind"], False),
    ("풍력 예측", "est_wind_gen_jeju", "est", C.COLOR["wind"], False),
]


def render_mw(df: pd.DataFrame, chosen: dict, show_da: bool, day: pd.Timestamp, k: int):
    """수요·신재생·순 부하 비교(MW) — 선택일부터 k일을 한 화면. x축은 표시 기간만큼 넓어진다."""
    cd, tmpl = C.hz_hover(df)
    ts = df["timestamp"]
    fig = C.make_fig(height=440)
    for label, col, kind, color, _ in JEJU_SERIES:
        if not chosen[label]:
            continue
        if kind == "act":
            C.add_actual(fig, ts, df[col], f"{label} (MW)", color)
        else:
            C.add_forecast(fig, ts, df[col], f"{label} (MW)", color,
                           customdata=cd, hovertemplate=tmpl)
    if show_da:
        fig.add_scatter(x=ts, y=df["jeju_est_demand_da"], name="KPX 수요예측(DA) (MW)",
                        line=dict(color=C.COLOR["reference"], dash="dash", width=2,
                                  shape="spline", smoothing=C.LINE_SMOOTHING),
                        hovertemplate="%{x|%m-%d %H시} · %{y:,.0f} MW<br>"
                        "KPX 하루전 발표<extra>%{fullData.name}</extra>")
    # 위험구간 음영 — chart_warn 규약: DatetimeIndex 필수(set_index 누락 시 심야 판정이 조용히 오동작)
    df_indexed = df.set_index("timestamp")
    smp_col = "est_smp" if "est_smp" in df.columns else None
    danger_col = "smp_danger" if "smp_danger" in df.columns else None
    chart_warn.draw_warning_zones(fig, df_indexed, smp_col=smp_col, danger_col=danger_col)
    fig.update_xaxes(range=[day, day + pd.Timedelta(days=k)])
    st.plotly_chart(fig, width="stretch")


def render_smp(smp: pd.DataFrame, day: pd.Timestamp):
    """SMP 검증(원/kWh) — 이틀 전 밤 발표 예측 vs 하루전 SMP(KPX 발표). x축 24h 고정.

    익일 지평은 하루전 발표 SMP를 그대로 쓰므로(검증 의미 없음), 우리 모델의 진짜 산출인
    모레 지평(이틀 전 밤 발표)을 그 날 하루 전 KPX가 발표한 SMP와 비교한다.
    """
    ts = smp["timestamp"]
    if smp["est_smp"].isna().all() and smp["smp_jeju_da"].isna().all():
        st.caption("이 날짜의 이틀 전 예측·하루전 SMP가 없습니다 (이틀 전 발표본·발표 SMP 필요).")
        return
    m = C.error_metrics(smp["est_smp"], smp["smp_jeju_da"])
    if m:
        st.markdown(f"**이틀 전 밤 발표 예측 vs 하루전 SMP** — MAE **{m['mae']:,.1f}** 원/kWh · "
                    f"bias **{m['bias']:+.1f}%** · 표본 {m['n']}시간")
    else:
        st.caption("이틀 전 예측이 없어 비교 불가 (발표본 누락 — 발표본 구멍).")
    fig = C.make_fig(height=340, ytitle="SMP (원/kWh)")
    C.add_actual(fig, ts, smp["smp_jeju_da"], "하루전 SMP(KPX 발표)", C.COLOR["reference"])
    fig.add_scatter(x=ts, y=smp["est_smp"], name="이틀 전 밤 발표 예측", mode="lines",
                    line=dict(color=C.COLOR["smp"], dash="dot", width=2.5,
                              shape="spline", smoothing=C.LINE_SMOOTHING),
                    hovertemplate="%{x|%m-%d %H시} · %{y:,.1f} 원/kWh<extra>%{fullData.name}</extra>")
    dz = smp[smp["smp_danger"] == 1]
    if not dz.empty:
        fig.add_scatter(x=dz["timestamp"], y=dz["est_smp"], mode="markers", name="음수가격 경보",
                        marker=dict(color=C.COLOR["alert"], size=11, symbol="triangle-down",
                                    line=dict(width=1, color="#fff")),
                        hovertemplate="%{x|%m-%d %H시} · 음수가격 경보<extra></extra>")
    fig.update_xaxes(range=[day, day + pd.Timedelta(days=1)])   # 무조건 24시간(선택일)
    st.plotly_chart(fig, width="stretch")


def render_hourly_table(df: pd.DataFrame):
    """시간별 예측 표 — 순 부하·수요·신재생·SMP. 순 부하 셀은 경고 임계값으로 색칠(저=빨강, 고=파랑)."""
    t = df[["timestamp", "est_net_load_jeju", "est_demand_jeju", "est_renew_gen_jeju",
            "est_smp", "smp_danger"]].copy()
    t = t.rename(columns={
        "timestamp": "시각", "est_net_load_jeju": "순 부하(MW)", "est_demand_jeju": "수요(MW)",
        "est_renew_gen_jeju": "신재생(MW)", "est_smp": "SMP(원/kWh)", "smp_danger": "음수경보"})
    t["시각"] = t["시각"].dt.strftime("%m-%d %H시")
    for col in ["순 부하(MW)", "수요(MW)", "신재생(MW)"]:
        t[col] = t[col].round(0)
    t["SMP(원/kWh)"] = t["SMP(원/kWh)"].round(1)
    styled = t.style.apply(
        lambda r: chart_warn.style_net_load_warnings(r, net_col="순 부하(MW)"), axis=1
    ).format({"순 부하(MW)": "{:,.0f}", "수요(MW)": "{:,.0f}",
              "신재생(MW)": "{:,.0f}", "SMP(원/kWh)": "{:,.1f}"}, na_rep="—")
    st.dataframe(styled, width="stretch", hide_index=True, height=420)


def render_forecast_view():
    """예측 확인 — day_navigator + 표시 기간 슬라이더 + MW 차트 + 지표 카드 + SMP expander + 시간별 표."""
    day, _, cap = C.day_navigator("jeju_fc")
    _, h_hi = C.jeju_horizon_range()
    # 컨트롤 묶음: …실시간 새로고침 | 설정 | 경고 | 슬라이더 (네비게이터 끝 칸을 셋으로 나눔)
    c_set, c_warn, c_sld = cap.columns([0.55, 0.55, 1.9], vertical_alignment="center")
    with c_set.popover("설정", help="표시 데이터 설정", width="stretch"):
        chosen = {label: st.checkbox(label, value=default, key=f"jmw_{col}")
                  for label, col, _, _, default in JEJU_SERIES}
        show_da = st.checkbox("KPX 수요예측(DA)", value=True, key="jmw_da")
    with c_warn.popover("경고", help="위험구간 음영 임계값 설정", width="stretch"):
        threshold_values = chart_warn.render_warning_threshold_inputs()
        if st.button("적용", key="warn_apply", type="primary", width="stretch"):
            chart_warn.commit_warning_thresholds(threshold_values)
            st.rerun()
    k = c_sld.slider("표시 기간(일)", 1, h_hi, 1, key="jeju_hzk", width="stretch",
                     help="선택일부터 며칠치를 표시할지 — 각 시각은 가장 최근 발표본으로 채워집니다.")
    end = day + pd.Timedelta(days=k - 1)
    df = C.jeju_range_compare(day, end, mode="latest")
    if df["est_demand_jeju"].isna().all() and df["real_demand_jeju"].isna().all():
        lo, hi = C.jeju_date_range()
        st.warning(f"선택한 날짜의 예측이 아직 없습니다. 예측 보유 범위: **{lo} ~ {hi}**.")
        return
    # SMP 예측을 표시 구간에 병합 — 차트 음영(Min 밴드 SMP 조건)과 시간별 표가 함께 쓴다
    smp_range = C.jeju_smp_frame(day, end, mode="latest")
    df = df.merge(smp_range[["timestamp", "est_smp", "smp_danger"]], on="timestamp", how="left")
    # 차트 ① — 수요·신재생·순 부하 (선택일부터 k일) + 위험구간 음영
    render_mw(df, chosen, show_da, day, k)

    # 지표 5개 — 차트 ①과 같은 표시 구간 전체 기준. 보조줄 = 최대/최소가 '발생한 시점'.
    sc, wc = C.jeju_renew_capacity()
    nl = df["est_net_load_jeju"]
    su, wu = df["est_solar_util_jeju"], df["est_wind_util_jeju"]
    comb = (df["est_solar_gen_jeju"] + df["est_wind_gen_jeju"]) / (sc + wc) * 100
    ts_col = df["timestamp"]

    inject_metric_style()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("최대 순 부하(예측)", "—" if nl.isna().all() else f"{nl.max():,.0f} MW",
              _when(ts_col, nl, "max"), delta_color="off")
    m2.metric("최저 순 부하(예측)", "—" if nl.isna().all() else f"{nl.min():,.0f} MW",
              _when(ts_col, nl, "min"), delta_color="off")
    m3.metric("신재생 최대 이용률", "—" if comb.isna().all() else f"{comb.max():.0f} %",
              _when(ts_col, comb, "max"), delta_color="off",
              help="태양광+풍력 합산 이용률(발전 예측 ÷ 설비용량)의 표시 구간 최대치")
    m4.metric("태양광 최대 이용률", "—" if su.isna().all() else f"{su.max() * 100:.0f} %",
              _when(ts_col, su, "max"), delta_color="off")
    m5.metric("풍력 최대 이용률", "—" if wu.isna().all() else f"{wu.max() * 100:.0f} %",
              _when(ts_col, wu, "max"), delta_color="off")

    # 차트 ② — SMP 검증(펼쳐 보기). 익일 지평=발표값 그대로라 검증 의미 없음 → 모델 산출(모레 지평)로 검증.
    smp_v = C.jeju_smp_frame(day, day, mode="fixed", value=2)
    with st.expander("SMP 검증 — 이틀 전 밤 발표 예측 vs 하루전 SMP(KPX) (선택일 24시간)"):
        render_smp(smp_v, day)
        st.caption("SMP 모델의 실제 산출은 **이틀 전 밤 발표 예측(빨강)** 이며, "
                   "**하루 전 KPX가 발표한 SMP(회색)** 와 비교합니다. ▽ = 음수가격 경보.")

    # 시간별 표 — 표시 구간 전체 (est_smp·smp_danger 는 위에서 병합됨)
    with st.expander("시간별 예측 표"):
        render_hourly_table(df)


# =============================================================================
# 예측 검증 — 3탭 (page_jeju.render_validation 이식, 무수정)
# =============================================================================
# 검증 차트 — 모델별 색(실측·예측 공용). 수요=파랑·순 부하=보라·태양광=주황·풍력=회색.
ACC_COLORS = {"수요": C.COLOR["demand"], "순 부하": C.COLOR["net_load"],
              "태양광": C.COLOR["rad"], "풍력": C.COLOR["wind"]}
VAL_PERIODS = {"7일": 7, "14일": 14, "한달": 30, "3개월": 92, "전체": None, "직접 설정": "custom"}


def _period_control(prefix: str):
    """검증기간 컨트롤(프리셋+직접설정) → (start, end) 날짜 문자열. '직접 설정' 미완료면 None."""
    psel = st.segmented_control("검증기간 (실측 대상일)", list(VAL_PERIODS),
                                default="한달", key=f"{prefix}_p") or "한달"
    mode = VAL_PERIODS[psel]
    if mode == "custom":
        lo, hi = C.jeju_date_range()
        lo_d, hi_d = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
        rng = st.date_input("대상일 범위", value=(lo_d, hi_d),
                            min_value=lo_d, max_value=hi_d, key=f"{prefix}_c")
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            return rng[0].strftime("%Y-%m-%d"), rng[1].strftime("%Y-%m-%d")
        st.caption("시작·종료일을 모두 선택하세요.")
        return None
    if mode is None:                         # '전체' = 적재 시작일부터
        return C.jeju_date_range()[0], None
    return (TODAY - pd.Timedelta(days=mode)).strftime("%Y-%m-%d"), None


BASETIME_OPTS = {"전일 밤 발표": C.BASE_HOUR_12Z, "새벽 발표(당일예보)": C.BASE_HOUR_18Z}


def _basetime_control(prefix: str) -> tuple[str, str]:
    """발표(basetime) 필터 → (base_hour, 라벨).  12z/18z 검증을 분리해서 본다 —
    두 발표를 섞으면 같은 지평에 서로 다른 리드타임이 합산된다 (2026-07-18 이원화)."""
    sel = st.segmented_control("발표", list(BASETIME_OPTS), default="전일 밤 발표",
                               key=f"{prefix}_bh") or "전일 밤 발표"
    return BASETIME_OPTS[sel], sel


def _hz_options(base_hour: str, key: str):
    """발표별 실보유 지평 selectbox — key 를 발표별로 분리해 옵션 전환 시 상태 충돌 방지."""
    opts = C.jeju_horizon_options(base_hour)
    if not opts:
        return None
    return st.selectbox("지평", opts, format_func=C.hz_label, key=f"{key}_{base_hour}")


def render_daily_trend():
    """① 일별 정확도 추이 — 검증기간 + 발표 + 지평. x=날짜, y=오차율(%). 특정일 급등=기상 급변 신호."""
    c1, c2, c3 = st.columns([2.4, 1.0, 0.6], vertical_alignment="bottom")
    with c1:
        pr = _period_control("jtrend")
    with c2:
        base_hour, bh_lbl = _basetime_control("jtrend")
    with c3:
        k = _hz_options(base_hour, "jtrend_hz")
    if k is None:
        st.caption("이 발표의 예측이 아직 없습니다 (새벽 발표는 2026-07-18 도입 — 축적 중).")
        return
    if pr is None:
        return
    start, end = pr

    # 선택 지평 카드 — 검증기간 전체에서 그 지평 정확도(4개 모델 한눈).
    acc_h = C.jeju_horizon_accuracy(start, end, base_hour)
    hz_lbl = C.hz_label(k)
    if not acc_h.empty and hz_lbl in acc_h.index:
        inject_metric_style()
        row = acc_h.loc[hz_lbl]
        st.markdown(f"**{hz_lbl} 지평 정확도** ({bh_lbl}) — 검증기간 전체 기준")
        cards = st.columns(4)
        CARD = [("수요", "수요", "MAPE"), ("순 부하", "순 부하", "nMAE"),
                ("태양광", "태양광", "설비용량 nMAE"), ("풍력", "풍력", "설비용량 nMAE")]
        for col, (title, key, kind) in zip(cards, CARD):
            v = row[key] if pd.notna(row[key]) else None
            col.metric(title, "—" if v is None else f"{v:.1f} %", kind, delta_color="off")

    acc = C.jeju_daily_accuracy(start, end, k, base_hour)
    if acc.empty:
        st.caption("이 기간·발표·지평에 집계할 데이터가 없습니다.")
        return
    st.markdown(f"**일별 추이** — {hz_lbl} 지평 ({bh_lbl})")
    fig = C.make_fig(height=400, ytitle="오차율 (%)")
    for name in C.JEJU_ACC_MODELS:
        fig.add_scatter(x=acc.index, y=acc[name], name=name, mode="lines+markers",
                        line=dict(color=ACC_COLORS[name], width=2.2), marker=dict(size=5),
                        connectgaps=False,
                        hovertemplate="%{x|%m-%d} · %{y:.1f}%<extra>" + name + "</extra>")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"각 날짜를 **{bh_lbl}** 의 **{hz_lbl}** 지평으로 예측한 값의 일별 오차율 · "
               "수요=MAPE, 순 부하=평균 대비 nMAE, "
               "태양광·풍력=설비용량 기준 nMAE · 값이 갑자기 솟은 날 = 그 날 기상이 급변(비·구름 등)해 "
               "예측이 빗나간 날로 읽습니다.")


def render_horizon_curve():
    """② 지평별 성능 — 검증기간 + 발표. x=지평(또는 리드타임 h), y=오차율(%). 표는 expander."""
    c1, c2, c3 = st.columns([2.4, 1.0, 0.6], vertical_alignment="bottom")
    with c1:
        pr = _period_control("jhz")
    with c2:
        base_hour, bh_lbl = _basetime_control("jhz")
    with c3:
        lead_axis = st.toggle("리드타임(h) 축", key="jhz_lead",
                              help="x축을 지평 대신 '발표→대상 시각'의 실측 평균 리드타임(시간)으로 — "
                                   "밤/새벽 발표의 신선도를 한 축에서 비교할 때 씁니다.")
    if pr is None:
        return
    start, end = pr
    acc = C.jeju_horizon_accuracy(start, end, base_hour)
    if acc.empty:
        st.caption("이 기간·발표에 집계할 데이터가 없습니다.")
        return
    xs = acc["리드(h)"] if lead_axis else acc["hz"]
    fig = C.make_fig(height=400, ytitle="오차율 (%)")
    for name in C.JEJU_ACC_MODELS:
        fig.add_scatter(x=xs, y=acc[name], name=name, mode="lines+markers",
                        line=dict(color=ACC_COLORS[name], width=2.2), marker=dict(size=6),
                        text=list(acc.index),
                        hovertemplate="%{text} · %{y:.1f}%<extra>" + name + "</extra>")
    if lead_axis:
        fig.update_xaxes(title="평균 리드타임 (발표 → 대상 시각, 시간)")
    else:
        fig.update_xaxes(title="지평 (당일=0, 익일=1 …)", tickmode="array",
                         tickvals=list(acc["hz"]), ticktext=list(acc.index))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{bh_lbl} 기준 · 왼→오로 갈수록(지평이 길수록) 오차가 커지는 추세 = 멀리서 예측할수록 "
               "어렵다는 뜻 · 수요=MAPE, 순 부하=평균 대비 nMAE, 태양광·풍력=설비용량 기준 nMAE.")
    with st.expander("정확한 수치 표로 보기"):
        st.dataframe(acc.drop(columns=["hz"]), width="stretch")


def render_date_series():
    """③ 지정일 예측 시계열 — 지정일 + 발표 + 지평. 그 발표본 예측(점선) vs 실측(실선), 4계열 동시."""
    lo, hi = C.jeju_date_range()
    lo_d, hi_d = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    default_day = min(hi_d, (TODAY - pd.Timedelta(days=1)).date())
    c1, c2, c3 = st.columns([2.4, 1.0, 0.6], vertical_alignment="bottom")
    day = c1.date_input("지정일", value=default_day, min_value=lo_d, max_value=hi_d, key="jds_day")
    with c2:
        base_hour, bh_lbl = _basetime_control("jds")
    with c3:
        n = _hz_options(base_hour, "jds_hz")
    if n is None:
        st.caption("이 발표의 예측이 아직 없습니다 (새벽 발표는 2026-07-18 도입 — 축적 중).")
        return
    day = pd.Timestamp(day)
    df = C.jeju_range_compare(day, day, use_live=False, mode="fixed", value=n, base_hour=base_hour)
    if df["real_demand_jeju"].isna().all() and df["est_demand_jeju"].isna().all():
        st.caption("이 날짜의 예측·실측이 모두 없습니다.")
        return
    no_fc = df["est_demand_jeju"].isna().all()
    ts = df["timestamp"]
    cd, tmpl = C.hz_hover(df)
    fig = C.make_fig(height=440)
    SER = [("수요", "real_demand_jeju", "est_demand_jeju", C.COLOR["demand"]),
           ("신재생", "real_renew_gen_jeju", "est_renew_gen_jeju", C.COLOR["renew"]),
           ("태양광", "real_solar_gen_jeju", "est_solar_gen_jeju", C.COLOR["rad"]),
           ("풍력", "real_wind_gen_jeju", "est_wind_gen_jeju", C.COLOR["wind"])]
    for name, ac, ec, color in SER:
        C.add_actual(fig, ts, df[ac], f"{name} 실측", color)
        C.add_forecast(fig, ts, df[ec], f"{name} 예측", color, customdata=cd, hovertemplate=tmpl)
    fig.update_xaxes(range=[day, day + pd.Timedelta(days=1)])    # 선택일 24시간
    st.plotly_chart(fig, width="stretch")
    hz_lbl = C.hz_label(n)
    if no_fc:
        st.caption(f"⚠ {day:%Y-%m-%d}을 {bh_lbl} {hz_lbl} 지평으로 예측한 발표본이 없어 "
                   "실측만 표시됩니다(발표본 구멍).")
    st.caption(f"{day:%Y-%m-%d}을 **{bh_lbl}** 의 **{hz_lbl}** 지평으로 예측한 값(점선) vs 실측(실선) · "
               "수요·신재생·태양광·풍력 동시 표시 · 실측이 아직 없는 미래일이면 점선만 보입니다.")


def render_validation():
    """예측 검증 — 모델 정확도 시각화 3종. SMP는 제외(예측 확인에서 따로)."""
    st.subheader("모델 정확도 검증 (제주)")
    st.caption("수요=MAPE · 순 부하=평균 대비 nMAE · 태양광·풍력=설비용량 기준 nMAE(MAE÷설비용량) · "
               "값이 낮을수록 정확 · 검증기간 = 실측이 있는 대상일 기준.")
    t1, t2, t3 = st.tabs(["① 일별 정확도 추이", "② 지평별 성능", "③ 지정일 예측 시계열"])
    with t1:
        render_daily_trend()
    with t2:
        render_horizon_curve()
    with t3:
        render_date_series()


# =============================================================================
# 데이터 현황 — 히트맵 (page_jeju.render_data_status 이식, 무수정)
# =============================================================================
def render_data_status():
    """데이터 현황 — 주요 항목 × 날짜 일별 수집률 히트맵(흰 0% → 초록 100%)."""
    import plotly.graph_objects as go

    st.subheader("데이터 현황 (제주)")
    heat = C.jeju_coverage_daily()
    cols, labels = list(heat.columns), list(heat.index)
    is_lh = {lab: lh for lab, _, _, lh in C.JEJU_COVERAGE_ITEMS}   # 장지평(예보)=초록 / 그 외=파랑
    cov = heat.values                                              # 실제 수집률(0~1)
    # 부호로 색 구분: 장지평(예보)=+수집률(초록), 그 외=−수집률(파랑). 0=빈칸(연회색).
    signed = cov.copy()
    for i, lab in enumerate(labels):
        if not is_lh.get(lab, False):
            signed[i] = -cov[i]
    # 히트맵 중간값(0% 칸) — 표면과 같은 계열로 물러나게 (라이트=연회색 / 다크=짙은 슬레이트)
    zero_cell = "#f1f5f9" if C.theme_type() == "light" else "#232b3b"
    fig = go.Figure(go.Heatmap(
        z=signed, x=cols, y=labels, customdata=cov, xgap=1, ygap=2,
        colorscale=[[0.0, "#1d6fb8"], [0.5, zero_cell], [1.0, "#059669"]], zmin=-1, zmax=1,
        hovertemplate="%{y}<br>%{x} · 수집률 %{customdata:.0%}<extra></extra>", showscale=False))
    fig.update_layout(height=max(360, 34 * len(labels) + 90),
                      margin=dict(t=10, b=10, l=10, r=10),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                      # type='category' 필수 — 안 하면 'MM-DD'를 날짜로 오인해 월 단위로 뭉갬
                      xaxis=dict(type="category", tickfont=dict(size=10), tickangle=-90))
    today_lbl = pd.Timestamp.now().strftime("%m-%d")
    if today_lbl in cols:
        fig.add_vline(x=today_lbl, line_dash="dot", line_color=C.COLOR["alert"])
    st.plotly_chart(fig, width="stretch")
    st.caption("🟩 초록 = 예보(미래까지 채워짐) · 🟦 파랑 = 실측(오늘까지 채워짐) · "
               "진할수록 그날 수집률이 높음 · 빨간 점선 = 오늘.")
    C.help_expander(
        "**초록(예보 계열)** — 수요·순 부하·태양광·풍력 예측과 기상 예보는 3일후 이후 미래까지 "
        "미리 채워집니다.\n\n"
        "**파랑(실측 계열)** — 실측값과 하루전 SMP는 오늘까지만 채워집니다. "
        "SMP 예측은 익일·모레까지만 발행됩니다.\n\n"
        "데이터는 서버 crontab 이 매일 정해진 시각에 자동 수집합니다(관리자 메뉴에서 수동 실행도 가능).")
    with st.expander("항목별 최신 현황 요약"):
        st.dataframe(C.coverage_table("jeju"), width="stretch", hide_index=True)


# =============================================================================
# 관리자 — 수집·예측 실행 (ops 게이트)
# =============================================================================
def render_admin():
    if not C.ops_gate():
        return
    # 단계 정의는 run_pipeline.py 와 공유 — cron 과 수동 버튼이 항상 같은 파이프라인을 돈다.
    from run_pipeline import PIPELINE_STEPS

    st.markdown("#### ▶ 오늘 예측 — 원클릭 전체 실행")
    st.caption("①실측 수집 → ②예보 수집 → ③예측 체인 → ④SMP 순서로 실행합니다. "
               "서버에서는 crontab 이 같은 파이프라인(run_pipeline.py)을 매일 자동 실행합니다.")
    if st.button("▶ 오늘 예측 실행 (전체)", key="adm_oneclick", type="primary"):
        failed_labels = []
        with st.status("전체 파이프라인 실행 중... (수 분 소요, 창을 닫지 마세요)",
                       expanded=True) as status:
            for key, label, script, args in PIPELINE_STEPS:
                st.write(f"⏳ {label} ...")
                rc, out = C.run_script(Path(script), args)
                if rc == 0:
                    st.write(f"✅ {label} — 완료")
                else:
                    failed_labels.append(label)
                    st.write(f"❌ {label} — 실패 (종료코드 {rc})")
                    st.code(out[-2500:] or "(출력 없음)", language="text")
            status.update(
                label="전체 파이프라인 완료" if not failed_labels
                else f"일부 실패: {', '.join(failed_labels)}",
                state="complete" if not failed_labels else "error")
        st.cache_data.clear()
        if not failed_labels:
            st.success("오늘 예측 완료 — 종합·예측 확인 메뉴에 반영되었습니다.")

    st.divider()
    st.markdown("#### 개별 실행")
    st.caption("각 단계를 따로 실행합니다. 완료 후 조회 캐시가 자동으로 비워집니다.")
    for key, label, script, args in PIPELINE_STEPS:
        script = Path(script)
        col_btn, col_hint = st.columns([1.6, 3.4], vertical_alignment="center")
        col_hint.caption(f"`{script.name} {' '.join(args)}`")
        if col_btn.button(label, key=f"adm_{key}", width="stretch"):
            with st.spinner(f"{label} 실행 중... (창을 닫지 마세요)"):
                rc, out = C.run_script(script, args)
            (st.success if rc == 0 else st.error)(f"{label} — 종료코드 {rc}")
            st.code(out[-4000:] or "(출력 없음)", language="text")
            if rc == 0:
                st.cache_data.clear()

    # 고급 실행 — 체인/SMP 의 --base·--backfill·--no-write 등 임의 인자
    with st.expander("고급 실행 (인자 직접 지정)"):
        script_options = {f"{Path(s).name}": s for _, _, s, _ in PIPELINE_STEPS}
        pick = st.selectbox("스크립트", list(script_options), key="adm_adv_script")
        raw_args = st.text_input("인자", key="adm_adv_args",
                                 placeholder="예: --base 2026-07-16 --no-write / --backfill 7")
        if st.button("실행", key="adm_adv_run"):
            args = raw_args.split()
            with st.spinner(f"{pick} {raw_args} 실행 중..."):
                rc, out = C.run_script(Path(script_options[pick]), args)
            (st.success if rc == 0 else st.error)(f"{pick} — 종료코드 {rc}")
            st.code(out[-4000:] or "(출력 없음)", language="text")
            if rc == 0:
                st.cache_data.clear()

    # 보정표 관리 — 태양광 일 스케일링(흐린날 과대예측 억제). 예보 품질이 계절·모델에 따라
    # 흘러가므로 주기적으로 점검·재적합해야 한다. 점검은 빠르고, 재적합은 과거 예측을
    # 다시 만들어야 해서 수 분 걸린다.
    with st.expander("보정 관리 — 태양광 일 스케일링"):
        st.caption(
            "예보는 흐린 날 일사를 과대예측합니다. 그대로 두면 태양광을 많게 보고 "
            "**순 부하를 적게** 예측해 발전 준비가 부족해집니다. 그래서 그날 예보 일사의 "
            "분포로 흐린 날을 골라내 예측을 낮춥니다(맑은 날은 건드리지 않습니다).")
        scale_json = Path(P.DIR_MODELS_SOLARWIND_LGBM) / "solar_scale.json"
        if scale_json.exists():
            try:
                meta = json.loads(scale_json.read_text(encoding="utf-8"))
                c1, c2, c3 = st.columns(3)
                c1.metric("적합일", meta.get("fitted_at", "?"))
                c2.metric("적합창", str(meta.get("fit_window", "?")).replace("..", " ~ "))
                c3.metric("지평 수", f"{len(meta.get('params', {}))}개")
            except Exception as exc:
                st.warning(f"보정표를 읽지 못했습니다: {exc}")
        else:
            st.warning("보정표가 아직 없습니다 — 재적합을 먼저 실행하세요.")

        col_check, col_fit = st.columns(2)
        if col_check.button("드리프트 점검", key="adm_scale_check", width="stretch"):
            with st.spinner("최근 서빙 결과에서 편향을 재는 중..."):
                rc, out = C.run_script(Path(P.FIT_SOLAR_SCALE), ["--check"])
            (st.success if rc == 0 else st.error)(f"점검 완료 — 종료코드 {rc}")
            st.code(out[-3000:] or "(출력 없음)", language="text")

        fit_months = col_fit.number_input("재적합 기간(개월)", 2, 12, 5, key="adm_scale_months")
        if col_fit.button("재적합 실행", key="adm_scale_fit", width="stretch"):
            with st.spinner("과거 예측을 다시 만들어 적합 중... (수 분 소요, 창을 닫지 마세요)"):
                rc, out = C.run_script(Path(P.FIT_SOLAR_SCALE), ["--months", str(int(fit_months))])
            (st.success if rc == 0 else st.error)(f"재적합 — 종료코드 {rc}")
            st.code(out[-4000:] or "(출력 없음)", language="text")
            if rc == 0:
                st.cache_data.clear()

    # 브리핑 수동 생성 — 종합 메뉴와 같은 로직(Gemini → ai_briefings.db)
    with st.expander("AI 브리핑 수동 생성"):
        from pages import brief_jeju
        target = st.date_input("대상 날짜", value=TODAY.date(), key="adm_brief_day")
        if st.button("브리핑 생성 / 갱신", key="adm_brief_gen"):
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                day_ts = pd.Timestamp(target)
                fact_df = brief_jeju.build_fact_frame(day_ts)
                text, facts = brief_jeju.generate_energy_narrative(
                    fact_df,
                    st.session_state.get("warn_low", 250),
                    st.session_state.get("warn_high", 750))
                from pages import brief_store
                brief_store.save(day_ts.strftime("%Y-%m-%d"), days=1, kind="daily",
                                 brief_text=text, fact_text=facts, model=brief_jeju.GEMINI_MODEL)
            st.success("저장 완료 — 종합 메뉴 브리핑에 반영됩니다.")
            st.markdown(text)


# =============================================================================
# 메뉴 라우팅
# =============================================================================
if menu == "종합":
    render_home()
elif menu == "예측 확인":
    render_forecast_view()
elif menu == "예측 검증":
    render_validation()
elif menu == "데이터 현황":
    render_data_status()
else:
    render_admin()
