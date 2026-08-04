# -*- coding: utf-8 -*-
"""make_jeju_zones.py — 제주 읍면동(행정동) 경계를 3구역(west/east/south)으로 병합하는 1회 전처리.

입력 : data/refdata/jeju_emd_2013.json
        (제주 읍면동 43개 경계 — southkorea-maps kostat 2013 전국판에서 code 39* 만 추출)
출력 : data/refdata/jeju_zones_3.json  (FeatureCollection **3피처** — 구역별 병합 MultiPolygon)

방법 (v3, 2026-07-17 사용자 확정)
----
v1=앵커 보로노이 → v2=읍면동 43피처에 zone 태그 → v3=**구역별 dissolve(병합)**.
v2 는 같은 구역 안 읍면동 경계선까지 지도에 그려져 거슬린다는 피드백으로,
구역 내부 경계를 없애고 **구역 사이 경계선만** 남긴다.

병합 원리: 인접한 두 읍면동이 공유하는 변(edge)은 양쪽 폴리곤에서 두 번 나타나고,
구역 외곽(해안선·다른 구역과의 경계)의 변은 한 번만 나타난다 → 한 번만 나타난 변을
모아 고리(ring)로 이어 붙이면 구역 외곽선이 된다 (kostat 데이터는 topojson 유래라
공유 변의 꼭짓점이 완전히 일치해 순수 파이썬으로 가능; shapely 불필요).
섬(우도·추자 등)은 공유 변이 없어 독립 ring 으로 그대로 남는다.

구역 명단 (ZONE_ASSIGNMENT) 을 바꾸면 이 스크립트만 재실행하면 된다.

사용:  python tools/make_jeju_zones.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

JEJU_EMD_GEOJSON = os.path.join(P.REFDATA, 'jeju_emd_2013.json')

# 구역 ↔ 읍면동 명단 (사용자 확정 2026-07-17) — 43개 전부, 빠짐·중복 없어야 한다(아래에서 검증)
ZONE_ASSIGNMENT = {
    'west': [
        '노형동', '연동', '외도동', '이호동', '도두동',
        '용담1동', '용담2동', '삼도1동', '삼도2동', '오라동',
        '애월읍', '한림읍', '한경면', '대정읍', '안덕면',
    ],
    'east': [
        '조천읍', '구좌읍', '성산읍', '우도면', '추자면',
        '일도1동', '일도2동', '이도1동', '이도2동', '건입동',
        '화북동', '삼양동', '봉개동', '아라동',
    ],
    'south': [
        '남원읍', '표선면', '송산동', '정방동', '중앙동',
        '천지동', '효돈동', '영천동', '동홍동', '서홍동',
        '대륜동', '대천동', '중문동', '예래동',
    ],
}


PRECISION = 7   # 좌표 반올림 자릿수 — 공유 변 매칭용 (원 데이터 정밀도보다 넉넉)


def _rings_of(geometry: dict) -> list[list]:
    """Polygon/MultiPolygon 의 외곽 ring 들 (구멍 없음 가정 — 읍면동 경계엔 구멍이 없다)."""
    if geometry['type'] == 'Polygon':
        return [geometry['coordinates'][0]]
    return [poly[0] for poly in geometry['coordinates']]


def dissolve(rings: list[list]) -> list[list]:
    """ring 묶음을 병합 — 두 번 나타나는(=내부 공유) 변을 지우고 남은 변을 고리로 재조립.

    반환: 병합된 외곽 ring 목록 (본체 1개 + 섬들). 재조립 실패(열린 사슬)는 예외로 알린다.
    """
    def snap(point):
        return (round(point[0], PRECISION), round(point[1], PRECISION))

    edge_count: dict[frozenset, int] = defaultdict(int)
    directed_edges: list[tuple] = []
    for ring in rings:
        points = [snap(p) for p in ring]
        if points[0] == points[-1]:
            points = points[:-1]
        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            if a == b:
                continue
            edge_count[frozenset((a, b))] += 1
            directed_edges.append((a, b))

    over_shared = sum(1 for c in edge_count.values() if c > 2)
    if over_shared:
        print(f'  경고: 3회 이상 공유된 변 {over_shared}개 (T자 접합?) — 결과 확인 필요')

    # 경계 변 = 한 번만 나타난 변 (원래 진행 방향 유지 → ring 감김 방향 보존)
    boundary = [(a, b) for (a, b) in directed_edges if edge_count[frozenset((a, b))] == 1]
    next_points: dict[tuple, list] = defaultdict(list)
    for a, b in boundary:
        next_points[a].append(b)

    used: set[tuple] = set()
    rings_out: list[list] = []
    for start_a, start_b in boundary:
        if (start_a, start_b) in used:
            continue
        ring = [start_a]
        a, b = start_a, start_b
        while True:
            used.add((a, b))
            ring.append(b)
            if b == start_a:
                break
            candidates = [c for c in next_points[b] if (b, c) not in used]
            if not candidates:
                raise SystemExit(f'열린 사슬 발생 (공유 변 꼭짓점 불일치) — 시작점 {start_a}')
            a, b = b, candidates[0]
        rings_out.append([list(p) for p in ring])
    return rings_out


def main():
    with open(JEJU_EMD_GEOJSON, encoding='utf-8') as f:
        gj = json.load(f)
    features = gj['features']
    print(f'제주 읍면동 피처 {len(features)}개')

    # 명단 ↔ 지도 대조: 중복·누락·미배정 전부 잡는다
    name_to_zone = {}
    for zone, names in ZONE_ASSIGNMENT.items():
        for name in names:
            if name in name_to_zone:
                raise SystemExit(f'중복 배정: {name} ({name_to_zone[name]} / {zone})')
            name_to_zone[name] = zone
    map_names = {ft['properties']['name'] for ft in features}
    missing_in_map = sorted(set(name_to_zone) - map_names)
    unassigned = sorted(map_names - set(name_to_zone))
    if missing_in_map:
        raise SystemExit(f'지도에 없는 명단 항목: {missing_in_map}')
    if unassigned:
        raise SystemExit(f'구역 미배정 읍면동: {unassigned}')

    out_features = []
    for zone, names in ZONE_ASSIGNMENT.items():
        member_rings = []
        for ft in features:
            if name_to_zone[ft['properties']['name']] == zone:
                member_rings += _rings_of(ft['geometry'])
        merged = dissolve(member_rings)
        # 꼭짓점 수 기준 내림차순 — 첫 ring = 본체, 나머지 = 섬(우도·추자 등)
        merged.sort(key=len, reverse=True)
        out_features.append({
            'type': 'Feature',
            'properties': {'zone': zone, 'members': names},
            'geometry': {'type': 'MultiPolygon',
                         'coordinates': [[ring] for ring in merged]},
        })
        print(f'  {zone:5s}: 읍면동 {len(names)}개 → ring {len(merged)}개 '
              f'(꼭짓점 {[len(r) for r in merged]})')

    out = {'type': 'FeatureCollection', 'features': out_features}
    with open(P.REF_JEJU_ZONES, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'저장: {P.REF_JEJU_ZONES}')


if __name__ == '__main__':
    main()
