# -*- coding: utf-8 -*-
"""AI 예측 브리핑 — Gemini 로 제주 일일 운영 브리핑 생성 + SQLite 저장.

(원본: Model_api_added/utils/gemini.py 의 구조를 유지하며 이식)
변경점:
  - 저장: 로컬 JSON → utils/brief_store.py (ai_briefings.db) — 동시 rerun 안전.
  - 컬럼: est_net_demand → est_net_load_jeju, smp_jeju → est_smp.
  - 임계값: chart_warn 과 같은 session_state 키를 읽는다(경고 popover 에서 바꾸면 브리핑도 따라감).
  - 리스크 추가: SMP 음수가격 경보(smp_danger==1)·음수확률(smp_neg_proba ≥ 0.5).
  - 기상: forecast_horizon(최신 base) 3구역 평균 시계열.

핵심 원칙(원본 유지): 리스크는 파이썬 `_detect_risks` 가 확정하고, LLM 은
"[감지된 리스크]에 명시된 이벤트만 언급"한다 — 추론·창작 금지.
"""
from pathlib import Path
import os
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pages import common as C
from pages import brief_store
from pages.chart_warn import SMP_MIN_THRESHOLD, _WARN_DEFAULTS

GEMINI_MODEL = "gemini-3-flash-preview"   # 원본 유지 — 교체 시 이 상수만
NEG_PROBA_ALERT = 0.5                     # 음수확률 리스크 임계
WEATHER_SUFFIXES = ["west", "east", "south"]


# ==========================================
# 데이터 조립 — 선택일 24h 팩트 프레임 (DatetimeIndex)
# ==========================================
@st.cache_data(ttl=C.CACHE_TTL)
def _weather_hourly(date: str) -> pd.DataFrame:
    """forecast_horizon(최신 base) 선택일 24h — 3구역 평균 total_cloud/rainfall/wind_spd/solar_rad."""
    cols = []
    for sx in WEATHER_SUFFIXES:
        cols += [f"total_cloud_{sx}", f"rainfall_{sx}", f"wind_spd_10m_{sx}", f"radiation_{sx}"]
    sel = ", ".join(f'"{c}"' for c in cols)
    df = C.query("jeju",
                 f"SELECT timestamp, {sel} FROM forecast_horizon fh "
                 "WHERE timestamp BETWEEN ? AND ? "
                 "AND base=(SELECT MAX(base) FROM forecast_horizon WHERE timestamp=fh.timestamp) "
                 "ORDER BY timestamp",
                 (f"{date} 00:00:00", f"{date} 23:00:00"))
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    for var, prefix in [("total_cloud", "total_cloud_"), ("rainfall", "rainfall_"),
                        ("wind_spd", "wind_spd_10m_"), ("solar_rad", "radiation_")]:
        out[var] = df[[f"{prefix}{sx}" for sx in WEATHER_SUFFIXES]].mean(axis=1)
    return out


def build_fact_frame(day: pd.Timestamp) -> pd.DataFrame:
    """선택일 24h 팩트 프레임 — 예측(순 부하·이용률)+SMP+3구역 평균 기상, DatetimeIndex."""
    df = C.jeju_range_compare(day, day, use_live=False, mode="latest")
    smp = C.jeju_smp_frame(day, day, mode="latest")
    df = df.merge(smp[["timestamp", "est_smp", "smp_neg_proba", "smp_danger"]],
                  on="timestamp", how="left")
    wx = _weather_hourly(day.strftime("%Y-%m-%d"))
    if not wx.empty:
        df = df.merge(wx, on="timestamp", how="left")
    return df.set_index("timestamp")


# ==========================================
# 리스크 감지 — LLM 대신 코드에서 확정
# ==========================================
def _time_block_summary(df: pd.DataFrame, col: str) -> str:
    """시간대별 블록 평균 요약 (심야/오전/오후/야간)."""
    blocks = {
        '심야(00-06)': (0, 5),
        '오전(06-12)': (6, 11),
        '오후(12-18)': (12, 17),
        '야간(18-23)': (18, 23),
    }
    if col not in df.columns:
        return "데이터 없음"
    parts = []
    for label, (h0, h1) in blocks.items():
        sub = df[(df.index.hour >= h0) & (df.index.hour <= h1)]
        if sub.empty or sub[col].isna().all():
            continue
        parts.append(f"{label} {sub[col].mean():.1f}")
    return " → ".join(parts) if parts else "데이터 없음"


def _detect_risks(df: pd.DataFrame, warn_low: int, warn_high: int) -> list[str]:
    """임계치 위반 시간대 추출 — chart_warn 과 같은 session_state 임계값 사용."""
    events = []
    net = "est_net_load_jeju"

    if net in df.columns and df[net].notna().any():
        low = df[df[net] < warn_low]
        if not low.empty:
            events.append(
                f"저부하: {low.index.hour.min():02d}~{low.index.hour.max():02d}시, "
                f"최저 {low[net].min():.0f}MW ({len(low)}시간)"
            )
        high = df[df[net] > warn_high]
        if not high.empty:
            events.append(
                f"고부하: {high.index.hour.min():02d}~{high.index.hour.max():02d}시, "
                f"최대 {high[net].max():.0f}MW ({len(high)}시간)"
            )

        if st.session_state.get('warn_min_enabled', _WARN_DEFAULTS['warn_min_enabled']):
            warn_min = st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min'])
            cond = df[net] < warn_min
            if 'est_smp' in df.columns:
                cond = cond | (df['est_smp'] < SMP_MIN_THRESHOLD)
            mn = df[cond]
            if not mn.empty:
                events.append(
                    f"최저발전 경고: {mn.index.hour.min():02d}~{mn.index.hour.max():02d}시, "
                    f"최저 순부하 {mn[net].min():.0f}MW ({len(mn)}시간, "
                    f"임계 {warn_min}MW 또는 SMP<{SMP_MIN_THRESHOLD}원)"
                )

        if st.session_state.get('warn_overnight_enabled', _WARN_DEFAULTS['warn_overnight_enabled']):
            warn_overnight = st.session_state.get('warn_overnight', _WARN_DEFAULTS['warn_overnight'])
            overnight = df[(df.index.hour < 6) & (df[net] < warn_overnight)]
            if not overnight.empty:
                events.append(
                    f"심야 저부하(00-06시): {overnight.index.hour.min():02d}~{overnight.index.hour.max():02d}시, "
                    f"최저 {overnight[net].min():.0f}MW ({len(overnight)}시간, 임계 {warn_overnight}MW)"
                )

    # ── 제주 추가 리스크: SMP 음수가격 경보·음수확률 ──
    if 'smp_danger' in df.columns:
        danger = df[df['smp_danger'].fillna(0) == 1]
        if not danger.empty:
            events.append(
                f"SMP 음수가격 경보: {danger.index.hour.min():02d}~{danger.index.hour.max():02d}시 "
                f"({len(danger)}시간)"
            )
    if 'smp_neg_proba' in df.columns and df['smp_neg_proba'].notna().any():
        p_max = float(df['smp_neg_proba'].max())
        if p_max >= NEG_PROBA_ALERT:
            events.append(f"SMP 음수확률 최대 {p_max:.0%} (임계 {NEG_PROBA_ALERT:.0%})")

    return events


# ==========================================
# 브리핑 생성 (Gemini)
# ==========================================
def generate_energy_narrative(df: pd.DataFrame, warn_low: int, warn_high: int) -> tuple[str, str]:
    """(브리핑 텍스트, 팩트 블록) 반환. API 키 없으면 안내 문구로 강등."""
    net = "est_net_load_jeju"
    try:
        # ── 기상 (3구역 평균) ──
        avg_cloud = df['total_cloud'].mean() if 'total_cloud' in df.columns else 0
        avg_wind = df['wind_spd'].mean() if 'wind_spd' in df.columns else 0
        avg_rain = df['rainfall'].mean() if 'rainfall' in df.columns else 0
        daytime = df[(df.index.hour >= 9) & (df.index.hour <= 16)]
        avg_solar = daytime['solar_rad'].mean() if not daytime.empty and 'solar_rad' in daytime.columns else 0

        # ── 이용률 ──
        avg_solar_util = daytime['est_solar_util_jeju'].mean() \
            if not daytime.empty and 'est_solar_util_jeju' in daytime.columns else 0
        avg_wind_util = df['est_wind_util_jeju'].mean() if 'est_wind_util_jeju' in df.columns else 0

        # ── 순 부하 & SMP ──
        min_net = df[net].min() if net in df.columns else 0
        max_net = df[net].max() if net in df.columns else 0
        if 'est_smp' in df.columns and not df['est_smp'].dropna().empty:
            smp_str = f"{df['est_smp'].min():.1f}"
        else:
            smp_str = "데이터 없음"

        # ── 시간대별 흐름 ──
        cloud_flow = _time_block_summary(df, 'total_cloud')
        rain_flow = _time_block_summary(df, 'rainfall')
        wind_flow = _time_block_summary(df, 'wind_spd')
        net_flow = _time_block_summary(df, net)

        # ── 리스크 감지 (코드 확정) ──
        risks = _detect_risks(df, warn_low, warn_high)
        risk_str = "\n".join(f"  ⚠ {r}" for r in risks) if risks else "정상 범위"
    except Exception as e:  # noqa: BLE001
        return f"Data processing error: {e}", ""

    warn_min = st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min'])
    warn_overnight = st.session_state.get('warn_overnight', _WARN_DEFAULTS['warn_overnight'])
    fact_block = f"""[데이터]
- 기상(3구역 평균): 전운량 {avg_cloud:.1f}, 일사 {avg_solar:.2f}, 풍속 {avg_wind:.1f}m/s, 강수 {avg_rain:.2f}mm
- 시간대별 전운량: {cloud_flow}
- 시간대별 강수: {rain_flow}
- 시간대별 풍속: {wind_flow}
- SMP: 최저 {smp_str}원
- 이용률: 태양광 {avg_solar_util:.2%}, 풍력 {avg_wind_util:.2%}
- 순부하: 최저 {min_net:.1f}MW, 최대 {max_net:.1f}MW
- 시간대별 순부하: {net_flow}
- 임계치: 저부하 {warn_low}MW, 고부하 {warn_high}MW, 최저발전 {warn_min}MW (또는 SMP<{SMP_MIN_THRESHOLD}원), 심야(00-06) {warn_overnight}MW
- 임계 교차 여부: 최저 순부하 {min_net:.1f}MW {"<" if min_net < warn_low else "≥"} 저부하 {warn_low}MW / 최대 순부하 {max_net:.1f}MW {">" if max_net > warn_high else "≤"} 고부하 {warn_high}MW

[감지된 리스크]
{risk_str}"""

    if not os.getenv("GEMINI_API_KEY"):
        return ("(Gemini API 키가 없어 브리핑을 생성할 수 없습니다 — .env 의 GEMINI_API_KEY 를 설정하세요.)",
                fact_block)

    sys_instruct = (
        "당신은 제주 LNG터미널 운영 핵심 요약 전문가입니다. "
        "장황한 설명 없이 데이터 기반의 운영 지침을 4~5줄 내외로 요약하여 보고하십시오."
    )
    prompt = f"""{fact_block}

[작성 규칙]
⚠ 절대 원칙: [감지된 리스크]에 명시된 이벤트만 언급하십시오. 거기에 없는 리스크(예: '고부하')를 평균·추세·증감으로부터 추론하거나 창작하지 마십시오. 해당 이벤트가 '정상 범위'면 반드시 '정상'으로만 서술하십시오.

1. 최대 5줄, 각 항목 '•' 기호 + 두 번 줄바꿈(\\n\\n) 개조식.
2. 첫 항목: 시간대별 기상 흐름 요약 ('오전 흐림→오후 비', '종일 맑고 바람 약함' 등 상태 위주).
3. 둘째 항목: 태양광/풍력 이용률 동향.
4. 강수가 0보다 크면 강수 시간대와 예측 정확도 저하 가능성 언급.
5. 셋째~넷째 항목: [감지된 리스크] 중 **주간(06-18시)** 시간대에 해당하는 이벤트에만 기반해 LNG 운영 방향을 작성. 규칙:
   - [감지된 리스크]에 '저부하' → "순부하 감소로 LNG 발전 정지 가능성"
   - [감지된 리스크]에 '최저발전 경고' → "(-) SMP 또는 저순부하로 LNG 정지·태양광 최대 발전 예상"
   - [감지된 리스크]에 'SMP 음수가격 경보' 또는 'SMP 음수확률' → "SMP 음수가격 위험 시간대 LNG 발전 정지·최소 출력 검토"
   - [감지된 리스크]에 '고부하' → "순부하 증가로 LNG 발전량 증가 예상"
   - 주간에 해당하는 위 이벤트가 **하나도 없으면** → "주간(오전·오후) 순부하 정상 범위, LNG 안정 운영 전망" (이 경우 반드시 한 줄로 작성하고 다른 리스크를 만들지 말 것).
6. 다섯번째 항목: [감지된 리스크] 중 **심야(00-06) 또는 야간(18-23)** 에 해당하는 이벤트 기반 LNG 운영 방향.
   - '심야 저부하' 감지 시 → "심야 LNG 정지/최소 출력 유지 검토 필요"
   - 심야·야간 해당 리스크가 없으면 → "심야·야간 순부하 정상 범위, LNG 안정 운영 전망"
7. "~입니다", "~습니다" 경어체.
"""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2,
            )
        )
        return response.text, fact_block
    except Exception as e:  # noqa: BLE001
        return f"Generation failed: {e}", fact_block


# ==========================================
# UI — 종합 화면 expander
# ==========================================
def _brief_card_html(text: str) -> str:
    """'•' 개조식 텍스트를 테마 brief-card(공용 CSS)로 감싼 HTML. 형식이 다르면 None."""
    items = [ln.lstrip("•").strip() for ln in text.splitlines() if ln.strip().startswith("•")]
    if not items:
        return ""
    body = "".join(f'<div class="bi">{it}</div>' for it in items)
    return f'<div class="brief-card">{body}</div>'


def render_briefing_expander(day: pd.Timestamp, title: str = "✨ AI 예측 브리핑"):
    """종합 화면 — hero 지도 아래 브리핑 expander. 표시 = store.load, 생성/갱신 → store.save."""
    date_key = day.strftime("%Y-%m-%d")
    with st.expander(title, expanded=True):
        saved = brief_store.load(date_key, days=1, kind="daily")

        col_btn, col_legend = st.columns([1, 2])
        do_generate = col_btn.button("AI 브리핑 생성 / 갱신", key=f"brief_gen_{date_key}")
        col_legend.caption("시간대 구분 · 00-06 심야 · 06-12 오전 · 12-18 오후 · 18-23 야간")

        if do_generate:
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                df = build_fact_frame(day)
                warn_low = st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low'])
                warn_high = st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high'])
                text, facts = generate_energy_narrative(df, warn_low, warn_high)
                brief_store.save(date_key, days=1, kind="daily",
                                 brief_text=text, fact_text=facts, model=GEMINI_MODEL)
            st.rerun()

        if saved and saved.get("brief_text"):
            card = _brief_card_html(saved["brief_text"])
            if card:
                st.markdown(card, unsafe_allow_html=True)
            else:
                st.markdown(saved["brief_text"])
            st.caption(f"생성 {saved['created_at']} · {saved.get('model') or GEMINI_MODEL}")
            with st.popover("근거 데이터 보기"):
                st.code(saved.get("fact_text") or "(없음)", language="text")
        else:
            st.caption("위 버튼을 눌러 해당 날짜의 브리핑을 생성하세요. 생성된 내용은 DB에 저장되어 다음에도 보입니다.")
