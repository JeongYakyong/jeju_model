# -*- coding: utf-8 -*-
"""위험구간 음영 — 순 부하 경고 밴드(vrect)·임계값 편집·표 하이라이트.

(원본: Model_api_added/utils/chart_helpers.py 에서 경고 관련 부분만 발췌 이식)

컬럼 적응: 원본 est_net_demand → 이 프로젝트의 est_net_load_jeju (NET_COL 상수),
          원본 smp_jeju → est_smp (호출부가 smp_col 인자로 전달).

★ 규약: draw_warning_zones 의 df 는 **DatetimeIndex** 여야 한다 (심야 판정에 df.index.hour 사용).
  timestamp 컬럼 프레임을 그대로 넘기면 조용히 오동작하므로 호출부에서 반드시
  ``df.set_index("timestamp")`` 후 전달할 것.

우선순위 (높음→낮음): 최저발전(Min) > 최대발전(Max) > 심야 저부하(Overnight) > 저발전/고발전.
각 시각은 활성화된 가장 높은 우선순위의 경고 한 개만 음영 표시된다(상호배타).
기본 임계값(100/250/300/750/900MW)은 제주 순 부하 스케일 기준.
"""
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NET_COL = "est_net_load_jeju"     # 경고 판정 대상 컬럼 (원본 est_net_demand)
SMP_MIN_THRESHOLD = 10            # SMP < 이 값(원/kWh)이면 최저발전(Min) 경고에 합산

# 밴드 색 — 임계값 popover 의 이모지(🔴 최저 · 🟡 저/심야 · 🔵 고 · 🟣 최대)와 일치.
# 라이트·다크 표면 양쪽에서 0.25 투명 wash 로 읽히는 중간 명도 스텝 (테마 공용).
BAND_COLORS = {
    "min":       "#d03b3b",   # 최저발전 (가장 심각 — 음수 SMP 동반)
    "overnight": "#fab219",   # 심야 저부하
    "low":       "#fab219",   # 저발전 (심야와 같은 계열 — 우선순위 배타라 동시 표시 없음)
    "high":      "#2a78d6",   # 고발전
    "max":       "#8b5cf6",   # 최대발전 (기본 OFF)
}

_WARN_DEFAULTS = {
    'warn_low':               250,
    'warn_high':              750,
    'warn_min_enabled':       True,
    'warn_min':               100,
    'warn_max_enabled':       False,
    'warn_max':               900,
    'warn_overnight_enabled': True,
    'warn_overnight':         300,
}

OVERNIGHT_END_HOUR = 6   # 심야 경고 적용 구간: hour < OVERNIGHT_END_HOUR (00:00 ~ 05:59)


def init_warning_state():
    """페이지 진입 시 경고 임계값 session_state 기본값 초기화 (메뉴 간 공유)."""
    for k, v in _WARN_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _subtract_intervals(intervals, exclude):
    """intervals 목록에서 exclude 목록에 해당하는 구간을 잘라낸다."""
    result = []
    for s, e in intervals:
        remaining = [(s, e)]
        for xs, xe in sorted(exclude):
            clipped = []
            for rs, re in remaining:
                if xe <= rs or xs >= re:
                    clipped.append((rs, re))
                elif xs <= rs and xe >= re:
                    pass
                elif xs <= rs:
                    clipped.append((xe, re))
                elif xe >= re:
                    clipped.append((rs, xs))
                else:
                    clipped.append((rs, xs))
                    clipped.append((xe, re))
            remaining = clipped
        result.extend(remaining)
    return result


def draw_danger_zones(fig, df, condition_series, fill_color,
                      annotation_text=None, show_legend_label=None,
                      layer_pos="below", fill_opacity=0.15,
                      legend_ref='legend', padding_hours=1.0,
                      exclude_intervals=None):
    """Plotly figure에 위험 구간 음영(vrect)을 추가하는 헬퍼.

    반환값: 패딩 적용 후 병합된 구간 리스트 (상위 우선순위 구간 exclusion 전달용).
    df 는 DatetimeIndex 가정(연속 시간 그룹핑에 index 사용).
    """
    if not condition_series.any():
        return []

    danger_df = df[condition_series].copy()
    danger_df['group'] = (condition_series != condition_series.shift()).cumsum()
    danger_df['temp_time'] = danger_df.index

    danger_zones = danger_df.groupby('group').agg(
        start=('temp_time', 'min'),
        end=('temp_time', 'max')
    )

    pad = timedelta(hours=padding_hours)
    raw_intervals = sorted(
        [(row['start'] - pad, row['end'] + pad) for _, row in danger_zones.iterrows()]
    )
    merged = []
    for s, e in raw_intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    draw_ivs = _subtract_intervals(merged, exclude_intervals) if exclude_intervals else merged

    for start_time, end_time in draw_ivs:
        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=fill_color, opacity=fill_opacity,
            layer=layer_pos, line_width=0,
        )
        if annotation_text is not None:
            center_time = start_time + (end_time - start_time) / 2
            fig.add_annotation(
                x=center_time,
                y=0.97, yref='paper',
                text=annotation_text,
                showarrow=False,
                yanchor='top',
                xanchor='center',
                font=dict(size=10, color='dimgray'),
            )

    if show_legend_label:
        trace_kwargs = dict(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=fill_color, symbol='square'),
            opacity=fill_opacity,
            name=show_legend_label, showlegend=True,
        )
        if legend_ref and legend_ref != 'legend':
            trace_kwargs['legend'] = legend_ref
        fig.add_trace(go.Scatter(**trace_kwargs))

    return merged


def draw_warning_zones(fig, df, smp_col=None, danger_col=None):
    """예측 차트에 경고 음영을 일괄 표시 (우선순위 기반 상호배타 마스크).

    ★ df 는 DatetimeIndex 프레임 — 호출부에서 set_index("timestamp") 필수 (모듈 docstring 참고).

    - 최저발전 (기본 ON, darkorange): NET_COL < warn_min  OR  smp < SMP_MIN_THRESHOLD
                                      OR (danger_col 지정 시) 음수가격 경보==1
    - 최대발전 (기본 OFF, purple)   : NET_COL > warn_max
    - 심야 저부하 (기본 ON, gold)   : hour < 6  AND NET_COL < warn_overnight
    - 저발전 (기본 ON, gold)        : NET_COL < warn_low
    - 고발전 (기본 ON, blue)        : NET_COL > warn_high
    """
    init_warning_state()  # idempotent — session_state 기본값 보장

    if NET_COL not in df.columns:
        return

    nd = df[NET_COL]
    false_mask = pd.Series(False, index=df.index)

    # ── 원본 조건 ──
    low_raw = nd < st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low'])
    high_raw = nd > st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high'])

    min_raw = false_mask
    if st.session_state.get('warn_min_enabled', _WARN_DEFAULTS['warn_min_enabled']):
        min_raw = nd < st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min'])
        if smp_col and smp_col in df.columns:
            min_raw = min_raw | (df[smp_col] < SMP_MIN_THRESHOLD)
        if danger_col and danger_col in df.columns:
            min_raw = min_raw | (df[danger_col].fillna(0) == 1)   # SMP 음수가격 경보 합산

    max_raw = false_mask
    if st.session_state.get('warn_max_enabled', _WARN_DEFAULTS['warn_max_enabled']):
        max_raw = nd > st.session_state.get('warn_max', _WARN_DEFAULTS['warn_max'])

    overnight_raw = false_mask
    if st.session_state.get('warn_overnight_enabled', _WARN_DEFAULTS['warn_overnight_enabled']):
        hour_mask = pd.Series(df.index.hour < OVERNIGHT_END_HOUR, index=df.index)
        overnight_raw = hour_mask & (nd < st.session_state.get('warn_overnight',
                                                               _WARN_DEFAULTS['warn_overnight']))

    # ── 우선순위 배타 처리 ──
    min_mask = min_raw
    max_mask = max_raw & ~min_mask
    overnight_mask = overnight_raw & ~(min_mask | max_mask)
    priority_mask = min_mask | max_mask | overnight_mask
    low_mask = low_raw & ~priority_mask
    high_mask = high_raw & ~priority_mask

    # ── 렌더 (모두 below 레이어, 동일 투명도 — 범례 legend2 로만 식별) ──
    opacity = 0.25
    # 저수요 트랙: Min > Overnight > Low
    min_ivs = draw_danger_zones(fig, df, min_mask, BAND_COLORS['min'],
                                show_legend_label='최저발전',
                                layer_pos='below', fill_opacity=opacity,
                                legend_ref='legend2', padding_hours=1.0)
    overnight_ivs = draw_danger_zones(fig, df, overnight_mask, BAND_COLORS['overnight'],
                                      show_legend_label='심야 저부하',
                                      layer_pos='below', fill_opacity=opacity,
                                      legend_ref='legend2',
                                      exclude_intervals=min_ivs)
    draw_danger_zones(fig, df, low_mask, BAND_COLORS['low'],
                      show_legend_label='저발전',
                      layer_pos='below', fill_opacity=opacity,
                      legend_ref='legend2',
                      exclude_intervals=min_ivs + overnight_ivs)
    # 고수요 트랙: Max > High (Min/Max 동시 발생 없음 → 크로스트랙 exclusion 불필요)
    max_ivs = draw_danger_zones(fig, df, max_mask, BAND_COLORS['max'],
                                show_legend_label='최대발전',
                                layer_pos='below', fill_opacity=opacity,
                                legend_ref='legend2')
    draw_danger_zones(fig, df, high_mask, BAND_COLORS['high'],
                      show_legend_label='고발전',
                      layer_pos='below', fill_opacity=opacity,
                      legend_ref='legend2',
                      exclude_intervals=max_ivs)

    if any(m.any() for m in [min_mask, low_mask, max_mask, high_mask, overnight_mask]):
        fig.update_layout(
            legend2=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='left', x=0,
            )
        )


def render_warning_threshold_inputs():
    """경고 임계값 입력 위젯 세트 (popover/다이얼로그 안에서 호출).

    위젯에 key 를 붙이지 않고 **잠정값 dict 를 반환** — 호출측이 '적용' 버튼에서
    `commit_warning_thresholds(values)` 로 session_state 에 일괄 반영한다.
    (Streamlit 위젯이 key 바인딩 과정에서 session_state 를 0/False 로
    덮어쓰는 문제를 회피하기 위한 원본 로직 보존.)
    """
    init_warning_state()

    w_low = st.number_input(
        "🟡 저발전 경고 (MW)",
        value=int(st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low'])),
        step=10,
    )
    w_high = st.number_input(
        "🔵 고발전 경고 (MW)",
        value=int(st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high'])),
        step=10,
    )

    w_overnight_on = st.checkbox(
        "🌙 심야 저부하 경고 활성화 (00-06시)",
        value=bool(st.session_state.get('warn_overnight_enabled',
                                        _WARN_DEFAULTS['warn_overnight_enabled'])),
    )
    w_overnight = int(st.session_state.get('warn_overnight', _WARN_DEFAULTS['warn_overnight']))
    if w_overnight_on:
        w_overnight = st.number_input(
            "심야 순부하 임계값 (MW)",
            value=w_overnight, step=10,
        )

    w_min_on = st.checkbox(
        "🔴 최저발전 경고 활성화",
        value=bool(st.session_state.get('warn_min_enabled',
                                        _WARN_DEFAULTS['warn_min_enabled'])),
    )
    w_min = int(st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min']))
    if w_min_on:
        w_min = st.number_input(
            "최저 순부하 임계값 (MW)",
            value=w_min, step=10,
        )

    w_max_on = st.checkbox(
        "🟣 최대발전 경고 활성화",
        value=bool(st.session_state.get('warn_max_enabled',
                                        _WARN_DEFAULTS['warn_max_enabled'])),
    )
    w_max = int(st.session_state.get('warn_max', _WARN_DEFAULTS['warn_max']))
    if w_max_on:
        w_max = st.number_input(
            "최대 순부하 임계값 (MW)",
            value=w_max, step=10,
        )

    st.caption(
        f"경고 우선순위: 최저 > 최대 > 심야 > 저/고발전  \n\n"
        f"최저발전 조건: 순 부하 < 임계값 **또는** SMP < {SMP_MIN_THRESHOLD}원 "
        f"**또는** 음수가격 경보  \n\n"
        f"심야 조건: 00-06시 중 순 부하 < 임계값"
    )

    return {
        'warn_low':               int(w_low),
        'warn_high':              int(w_high),
        'warn_min_enabled':       bool(w_min_on),
        'warn_min':               int(w_min),
        'warn_max_enabled':       bool(w_max_on),
        'warn_max':               int(w_max),
        'warn_overnight_enabled': bool(w_overnight_on),
        'warn_overnight':         int(w_overnight),
    }


def commit_warning_thresholds(values: dict):
    """`render_warning_threshold_inputs()` 가 반환한 dict 를 session_state 에 반영."""
    for k, v in values.items():
        st.session_state[k] = v


def style_net_load_warnings(row, net_col: str = NET_COL):
    """데이터 테이블에서 순 부하 셀 배경색 하이라이트 (저=연빨강, 고=연파랑).

    사용: ``df.style.apply(lambda r: style_net_load_warnings(r, net_col="순 부하(MW)"), axis=1)``
    """
    styles = [''] * len(row)
    if net_col in row.index:
        nd = row[net_col]
        if pd.notna(nd):
            idx = row.index.get_loc(net_col)
            # 반투명 wash — 셀 글자색을 건드리지 않아 라이트/다크 테마 모두에서 읽힌다
            if nd < st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low']):
                styles[idx] = 'background-color: rgba(208,59,59,.28)'
            elif nd > st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high']):
                styles[idx] = 'background-color: rgba(42,120,214,.28)'
    return styles
