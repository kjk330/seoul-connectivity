"""12주차 프롬프트 2: 서울 녹지 패치 확률 연결성(PC, EC, dPC_if) 계산 스크립트."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import time

# 멀티프로세싱 시 OpenBLAS/OMP 스레드 과다 생성으로 인한 메모리 부족 방지
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import shapely


# 워커 프로세스 전역 변수 (Windows spawn 환경에서 반복 직렬화 방지)
_G_GRAPH = None
_G_AREAS = None
_G_DECAY = None
_G_TOP_K = None


def _init_worker(graph: csr_matrix, areas: np.ndarray, decay: float, top_k: int) -> None:
    """워커 프로세스에 그래프와 면적 배열을 초기화한다."""
    global _G_GRAPH, _G_AREAS, _G_DECAY, _G_TOP_K
    _G_GRAPH = graph
    _G_AREAS = areas
    _G_DECAY = decay
    _G_TOP_K = top_k


def _process_batch(task: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray | None]:
    """출발 패치 구간 [start, end)에 대한 최단 경로와 p* 합계를 계산한다."""
    start, end = task
    indices = np.arange(start, end, dtype=np.int32)
    # 제자리(in-place) 연산으로 메모리 사용 절감 (inf -> exp(-inf) = 0.0, 0 -> 1.0)
    pstar = dijkstra(_G_GRAPH, directed=False, indices=indices)
    np.multiply(pstar, -_G_DECAY, out=pstar)
    np.exp(pstar, out=pstar)

    # flux_ha_i = Σ_{j≠i} a_j * p*_ij = (Σ_j a_j * p*_ij) - a_i
    total_weighted = pstar @ _G_AREAS
    flux_chunk = total_weighted - _G_AREAS[start:end]

    # 상위 top_k 패치 간 p* 행렬 추출 (행 범위가 top_k와 겹치는 경우)
    top_chunk = None
    if start < _G_TOP_K:
        row_limit = min(end, _G_TOP_K) - start
        top_chunk = pstar[:row_limit, :_G_TOP_K].astype(np.float32)

    return start, end, flux_chunk, top_chunk


def main() -> None:
    t0 = time.perf_counter()
    base_dir = Path(".")
    patches_in_path = base_dir / "green_patches.gpkg"

    links_out_path = base_dir / "patch_links.csv"
    conn_out_path = base_dir / "patch_connectivity.gpkg"
    pstar_out_path = base_dir / "pstar_top2000.npy"
    summary_out_path = base_dir / "step2_summary.txt"

    # 분석 파라미터 (규칙 정의)
    max_link_dist_m = 500.0
    d50_m = 250.0
    decay = math.log(2.0) / d50_m
    landscape_area_ha = 60668.09
    batch_size = 500
    top_k = 2000

    # 1. 패치 불러오기 및 pid 오름차순 정렬
    print("1) green_patches.gpkg 불러오는 중...", flush=True)
    gdf = gpd.read_file(patches_in_path, layer="patches")
    gdf = gdf.sort_values("pid", ascending=True).reset_index(drop=True)

    n_patches = len(gdf)
    pids = gdf["pid"].to_numpy(dtype=np.int32)
    areas = gdf["ha"].to_numpy(dtype=np.float64)
    geoms = gdf.geometry.to_numpy()
    print(f"   패치 {n_patches:,}개 로드 완료 (면적 합 {areas.sum():,.1f} ha)", flush=True)

    # 2. STRtree(dwithin 500 m)로 직접 링크 후보 찾기 및 거리 계산
    print("2) 500 m 이내 직접 링크 탐색 및 거리 계산 중...", flush=True)
    t_link = time.perf_counter()
    tree = shapely.STRtree(geoms)
    pair_indices = tree.query(geoms, predicate="dwithin", distance=max_link_dist_m)

    # 중복 제거 (i < j, pid 오름차순이므로 pid_a < pid_b와 동일)
    upper_mask = pair_indices[0] < pair_indices[1]
    idx_a = pair_indices[0, upper_mask]
    idx_b = pair_indices[1, upper_mask]

    dists = shapely.distance(geoms[idx_a], geoms[idx_b])
    valid_link_mask = dists <= max_link_dist_m
    idx_a = idx_a[valid_link_mask]
    idx_b = idx_b[valid_link_mask]
    dists = dists[valid_link_mask]

    # pid_a, pid_b 오름차순 정렬 및 patch_links.csv 저장 (반올림 없음)
    links_df = pd.DataFrame(
        {
            "pid_a": pids[idx_a],
            "pid_b": pids[idx_b],
            "dist_m": dists,
        }
    ).sort_values(["pid_a", "pid_b"], kind="mergesort")
    links_df.to_csv(links_out_path, index=False)
    print(
        f"   직접 링크 {len(links_df):,}쌍 저장 완료 ({time.perf_counter() - t_link:.1f}초)",
        flush=True,
    )

    # 3. 희소 그래프 구성 및 병렬 최단 경로(dijkstra) 계산
    print("3) 희소 그래프 구성 및 배치별 최단 경로 계산 시작...", flush=True)
    t_dijk = time.perf_counter()
    rows = np.concatenate([idx_a, idx_b])
    cols = np.concatenate([idx_b, idx_a])
    weights = np.concatenate([dists, dists])
    graph = csr_matrix((weights, (rows, cols)), shape=(n_patches, n_patches))

    batches = [
        (start, min(start + batch_size, n_patches))
        for start in range(0, n_patches, batch_size)
    ]

    flux_ha = np.zeros(n_patches, dtype=np.float64)
    actual_top_k = min(top_k, n_patches)
    pstar_top = np.zeros((actual_top_k, actual_top_k), dtype=np.float32)

    max_workers = max(1, min(os.cpu_count() or 4, 6))
    print(
        f"   총 {len(batches)}개 배치 (배치 크기 {batch_size}), 워커 {max_workers}개 사용",
        flush=True,
    )

    completed_batches = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(graph, areas, decay, actual_top_k),
    ) as executor:
        futures = [executor.submit(_process_batch, b) for b in batches]
        for fut in as_completed(futures):
            start, end, flux_chunk, top_chunk = fut.result()
            flux_ha[start:end] = flux_chunk
            if top_chunk is not None:
                row_limit = top_chunk.shape[0]
                pstar_top[start : start + row_limit, :] = top_chunk
            completed_batches += 1
            if completed_batches % 5 == 0 or completed_batches == len(batches):
                print(
                    f"   진행: {completed_batches}/{len(batches)} 배치 완료 "
                    f"({time.perf_counter() - t_dijk:.1f}초 경과)",
                    flush=True,
                )

    np.save(pstar_out_path, pstar_top)
    print(f"   pstar_top2000.npy 저장 완료 (shape={pstar_top.shape})", flush=True)

    # 4. PC, EC, dPC_if 계산 및 patch_connectivity.gpkg 저장
    q_val = float(np.sum(areas * (areas + flux_ha)))
    pc_val = q_val / (landscape_area_ha**2)
    ec_val = math.sqrt(q_val)
    ec_isolated = math.sqrt(float(np.sum(areas**2)))
    total_green_ha = float(np.sum(areas))
    ec_pct_of_green = 100.0 * ec_val / total_green_ha

    dpc_intra = 100.0 * (areas**2) / q_val
    dpc_flux = 100.0 * 2.0 * areas * flux_ha / q_val
    dpc_if = dpc_intra + dpc_flux

    # dPC_if 내림차순 순위 (1부터 시작)
    order = np.argsort(-dpc_if, kind="mergesort")
    rank_if = np.empty(n_patches, dtype=np.int32)
    rank_if[order] = np.arange(1, n_patches + 1, dtype=np.int32)

    out_gdf = gpd.GeoDataFrame(
        {
            "pid": pids,
            "ha": areas,
            "flux_ha": flux_ha,
            "dPC_intra": dpc_intra,
            "dPC_flux": dpc_flux,
            "dPC_if": dpc_if,
            "rank_if": rank_if,
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
    )
    out_gdf.to_file(conn_out_path, layer="patches", driver="GPKG")

    # 5. 요약 작성 및 step2_summary.txt 저장
    dist_min = float(np.min(dists))
    dist_median = float(np.median(dists))

    # 주요 패치 쌍 p* 확인 (pid 1-based -> 인덱스 0-based)
    pid_to_idx = {int(p): i for i, p in enumerate(pids)}
    sample_pairs = [(1, 2), (1, 3), (1, 8), (1, 10), (2, 6)]
    pair_lines = []
    for pa, pb in sample_pairs:
        val = float(pstar_top[pid_to_idx[pa], pid_to_idx[pb]])
        pair_lines.append(f"pid {pa}–pid {pb} = {val:.3f}")

    top5 = out_gdf.sort_values("rank_if", ascending=True).head(5)
    top5_lines = [
        (
            f"  {int(r.rank_if)}위: pid {int(r.pid)} ({r.ha:,.1f} ha) — "
            f"dPC_if {r.dPC_if:.2f} % (intra {r.dPC_intra:.2f} + flux {r.dPC_flux:.2f})"
        )
        for r in top5.itertuples()
    ]

    elapsed = time.perf_counter() - t0

    summary_lines = [
        "[프롬프트 2] 패치 연결성 (PC · EC · dPC_if) 요약",
        f"- 직접 링크 수 (<= {max_link_dist_m:.0f} m): {len(links_df):,}쌍",
        f"- 링크 거리: 최소 {dist_min:.1f} m, 중앙값 {dist_median:.1f} m",
        f"- Q = {q_val:,.0f} ha²",
        f"- PC = {pc_val:.5f} (A_L = {landscape_area_ha:,.2f} ha)",
        f"- EC = {ec_val:,.1f} ha (녹지 면적 합 {total_green_ha:,.1f} ha의 {ec_pct_of_green:.1f} %)",
        f"- 단절 시 EC = {ec_isolated:,.1f} ha",
        f"- 주요 쌍 p*: {', '.join(pair_lines)}",
        "- dPC_if 상위 5개 패치:",
        *top5_lines,
        f"- 실행 시간: {elapsed:.1f}초",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_out_path.write_text(summary_text, encoding="utf-8")

    print("", flush=True)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
