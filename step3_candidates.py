"""12주차 프롬프트 3: 새 녹지 1 ha 후보지별 연결성 기여(dEC, dPC) 평가 스크립트."""

import math
from pathlib import Path
import time

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import shapely


def configure_korean_font() -> str:
    """시스템에 설치된 한글 글꼴을 찾아 matplotlib 기본 글꼴로 지정한다."""
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "NanumSquare",
        "AppleGothic",
        "Gulim",
        "Dotum",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in preferred_fonts:
        if name in installed:
            chosen = name
            break
    if chosen is not None:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen or "default"


def main() -> None:
    t0 = time.perf_counter()
    base_dir = Path(".")

    s2_path = base_dir / "seoul_s2_20260516.tif"
    ndvi_path = base_dir / "seoul_ndvi.tif"
    boundary_path = base_dir / "seoul_boundary.gpkg"
    conn_path = base_dir / "patch_connectivity.gpkg"
    links_path = base_dir / "patch_links.csv"
    pstar_path = base_dir / "pstar_top2000.npy"

    cand_out_path = base_dir / "candidates.gpkg"
    summary_out_path = base_dir / "step3_summary.txt"
    map_out_path = base_dir / "step3_map.png"

    # 규칙 파라미터
    grid_step_m = 500
    half_size_m = 50.0
    ndvi_green_thr = 0.6
    water_nir_thr = 800
    max_link_dist_m = 500.0
    d50_m = 250.0
    decay = math.log(2.0) / d50_m
    cand_area_ha = 1.0

    # 1. 경계 자료 읽기 및 500 m 격자 후보 정사각형 생성
    print("1) 경계 자료 및 영상 기반 후보지 선별 중...", flush=True)
    seoul_gdf = gpd.read_file(boundary_path, layer="seoul")
    gu_gdf = gpd.read_file(boundary_path, layer="gu")
    seoul_geom = seoul_gdf.geometry.union_all()
    shapely.prepare(seoul_geom)

    minx, miny, maxx, maxy = seoul_geom.bounds
    xs = np.arange(
        math.floor(minx / grid_step_m) * grid_step_m,
        math.ceil(maxx / grid_step_m) * grid_step_m + 1,
        grid_step_m,
        dtype=np.int64,
    )
    ys = np.arange(
        math.floor(miny / grid_step_m) * grid_step_m,
        math.ceil(maxy / grid_step_m) * grid_step_m + 1,
        grid_step_m,
        dtype=np.int64,
    )

    # 남쪽 -> 북쪽(y 오름차순), 같은 y에서는 서쪽 -> 동쪽(x 오름차순)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    all_x = grid_x.ravel()
    all_y = grid_y.ravel()
    all_squares = shapely.box(
        all_x - half_size_m,
        all_y - half_size_m,
        all_x + half_size_m,
        all_y + half_size_m,
    )

    # 조건 ①: 정사각형 전체가 seoul 경계 안에 있음
    in_seoul_mask = shapely.within(all_squares, seoul_geom)
    n_in_seoul = int(np.sum(in_seoul_mask))

    seoul_x = all_x[in_seoul_mask]
    seoul_y = all_y[in_seoul_mask]
    seoul_squares = all_squares[in_seoul_mask]

    # 조건 ②, ③: 10 x 10 화소에 녹지(NDVI >= 0.6) 및 수면(B08 < 800) 없음
    with rasterio.open(ndvi_path) as ndvi_src, rasterio.open(s2_path) as s2_src:
        ndvi_arr = ndvi_src.read(1)
        b08_arr = s2_src.read(2)
        left = s2_src.transform.c
        top = s2_src.transform.f
        res_x = s2_src.transform.a
        res_y = abs(s2_src.transform.e)
        height, width = b08_arr.shape

    keep_indices = []
    for i, (cx, cy) in enumerate(zip(seoul_x, seoul_y)):
        c0 = int(round((cx - half_size_m - left) / res_x))
        r0 = int(round((top - (cy + half_size_m)) / res_y))
        c1 = c0 + 10
        r1 = r0 + 10
        if c0 < 0 or r0 < 0 or c1 > width or r1 > height:
            continue
        ndvi_win = ndvi_arr[r0:r1, c0:c1]
        b08_win = b08_arr[r0:r1, c0:c1]
        if ndvi_win.shape != (10, 10) or b08_win.shape != (10, 10):
            continue
        has_green = np.any(ndvi_win >= ndvi_green_thr)
        has_water = np.any(b08_win < water_nir_thr)
        if not has_green and not has_water:
            keep_indices.append(i)

    keep_idx_arr = np.array(keep_indices, dtype=np.int64)
    cand_x = seoul_x[keep_idx_arr]
    cand_y = seoul_y[keep_idx_arr]
    cand_geoms = seoul_squares[keep_idx_arr]
    n_cand = len(cand_geoms)
    cids = np.arange(1, n_cand + 1, dtype=np.int32)

    # 중심점 기준 자치구(gu_nm) 매핑
    cand_centers = gpd.GeoSeries(shapely.points(cand_x, cand_y), crs=seoul_gdf.crs)
    center_gdf = gpd.GeoDataFrame({"cid": cids}, geometry=cand_centers, crs=seoul_gdf.crs)
    joined_gu = gpd.sjoin(center_gdf, gu_gdf[["gu_nm", "geometry"]], how="left", predicate="within")
    joined_gu = joined_gu.drop_duplicates(subset="cid").sort_values("cid")
    gu_names = joined_gu["gu_nm"].fillna("미분류").to_numpy()

    print(
        f"   서울 내부 격자점 {n_in_seoul:,}개 -> 조건 충족 후보지 {n_cand:,}개 선별 완료",
        flush=True,
    )

    # 2. 기존 패치, 링크, pstar_top2000 불러오기
    print("2) 기존 패치 연결성 및 링크 정보 로드 중...", flush=True)
    patch_gdf = gpd.read_file(conn_path, layer="patches")
    patch_gdf = patch_gdf.sort_values("pid", ascending=True).reset_index(drop=True)
    n_patches = len(patch_gdf)
    patch_pids = patch_gdf["pid"].to_numpy(dtype=np.int32)
    patch_areas = patch_gdf["ha"].to_numpy(dtype=np.float64)
    patch_flux = patch_gdf["flux_ha"].to_numpy(dtype=np.float64)
    patch_geoms = patch_gdf.geometry.to_numpy()

    base_q = float(np.sum(patch_areas * (patch_flux + patch_areas)))
    base_ec = math.sqrt(base_q)

    links_df = pd.read_csv(links_path)
    pid_to_idx = {int(p): idx for idx, p in enumerate(patch_pids)}
    link_u = links_df["pid_a"].map(pid_to_idx).to_numpy(dtype=np.int32)
    link_v = links_df["pid_b"].map(pid_to_idx).to_numpy(dtype=np.int32)
    link_d = links_df["dist_m"].to_numpy(dtype=np.float64)

    pstar_top = np.load(pstar_path).astype(np.float64)
    top_k = pstar_top.shape[0]

    # 3. 후보지 <-> 기존 패치 500 m 이내 링크 탐색
    print("3) 후보지-기존 패치 간 500 m 이내 링크 탐색 중...", flush=True)
    patch_tree = shapely.STRtree(patch_geoms)
    cp_pairs = patch_tree.query(cand_geoms, predicate="dwithin", distance=max_link_dist_m)
    c_idx = cp_pairs[0].astype(np.int32)
    p_idx = cp_pairs[1].astype(np.int32)
    cp_dists = shapely.distance(cand_geoms[c_idx], patch_geoms[p_idx])

    valid_cp = cp_dists <= max_link_dist_m
    c_idx = c_idx[valid_cp]
    p_idx = p_idx[valid_cp]
    cp_dists = cp_dists[valid_cp]

    # 후보별 직접 링크 수(n_link) 및 최단 거리(nearest_m)
    n_link = np.bincount(c_idx, minlength=n_cand).astype(np.int32)
    nearest_m = np.full(n_cand, np.inf, dtype=np.float64)
    np.minimum.at(nearest_m, c_idx, cp_dists)
    nearest_m[n_link == 0] = np.nan
    n_no_link = int(np.sum(n_link == 0))
    print(f"   후보-패치 링크 {len(cp_dists):,}쌍 탐색 완료 (링크 없는 후보 {n_no_link}개)", flush=True)

    # 4. 방향 그래프 구성 (기존 패치는 양방향, 후보 -> 패치는 단방향) 후 일괄 Dijkstra
    print("4) 방향 그래프 Dijkstra 및 dQ 3항(intra/flux/conn) 계산 중...", flush=True)
    total_nodes = n_patches + n_cand
    dir_rows = np.concatenate([link_u, link_v, n_patches + c_idx])
    dir_cols = np.concatenate([link_v, link_u, p_idx])
    dir_weights = np.concatenate([link_d, link_d, cp_dists])
    # 주의: 거리 0.0 링크를 유지하기 위해 eliminate_zeros()를 호출하지 않음
    dir_graph = csr_matrix((dir_weights, (dir_rows, dir_cols)), shape=(total_nodes, total_nodes))

    cand_node_indices = np.arange(n_patches, total_nodes, dtype=np.int32)
    dist_cand = dijkstra(dir_graph, directed=True, indices=cand_node_indices)[:, :n_patches]

    # p*_cj 계산 (거리 inf -> 0, 거리 0 -> 1)
    np.multiply(dist_cand, -decay, out=dist_cand)
    np.exp(dist_cand, out=dist_cand)
    pstar_cand = dist_cand

    flux_cand = pstar_cand @ patch_areas
    dq_intra = np.full(n_cand, cand_area_ha**2, dtype=np.float64)
    dq_flux = 2.0 * cand_area_ha * flux_cand

    # dQ_conn 계산 (pid 1~2000 쌍에 대해)
    top_areas = patch_areas[:top_k]
    dq_conn = np.zeros(n_cand, dtype=np.float64)
    for idx_c in range(n_cand):
        u = pstar_cand[idx_c, :top_k]
        if not np.any(u > 0.0):
            continue
        diff = np.outer(u, u) - pstar_top
        np.maximum(diff, 0.0, out=diff)
        np.fill_diagonal(diff, 0.0)
        dq_conn[idx_c] = float(top_areas @ diff @ top_areas)

    dq_total = dq_intra + dq_flux + dq_conn
    dec_ha = np.sqrt(base_q + dq_total) - base_ec
    dpc_add = 100.0 * dq_total / base_q
    conn_pct = 100.0 * dq_conn / dq_total

    # dEC_ha 내림차순 순위 (1부터)
    order = np.argsort(-dec_ha, kind="mergesort")
    ranks = np.empty(n_cand, dtype=np.int32)
    ranks[order] = np.arange(1, n_cand + 1, dtype=np.int32)

    # 5. candidates.gpkg 저장
    cand_gdf = gpd.GeoDataFrame(
        {
            "cid": cids,
            "gu_nm": gu_names,
            "x": cand_x.astype(np.int32),
            "y": cand_y.astype(np.int32),
            "n_link": n_link,
            "nearest_m": nearest_m,
            "flux_ha": flux_cand,
            "dQ_intra": dq_intra,
            "dQ_flux": dq_flux,
            "dQ_conn": dq_conn,
            "dQ": dq_total,
            "dPC_add": dpc_add,
            "dEC_ha": dec_ha,
            "conn_pct": conn_pct,
            "rank": ranks,
        },
        geometry=cand_geoms,
        crs=seoul_gdf.crs,
    )
    cand_gdf.to_file(cand_out_path, layer="candidates", driver="GPKG")

    # 6. step3_summary.txt 저장
    dec_min = float(np.min(dec_ha))
    dec_median = float(np.median(dec_ha))
    dec_mean = float(np.mean(dec_ha))
    dec_max = float(np.max(dec_ha))

    top10 = cand_gdf.sort_values("rank", ascending=True).head(10)
    top10_lines = [
        (
            f"  {int(r.rank):2d}위: {r.gu_nm} ({int(r.x)}, {int(r.y)}, cid {int(r.cid)}) — "
            f"dEC {r.dEC_ha:.4f} ha, 경유 비중 {r.conn_pct:.1f} %, "
            f"링크 {int(r.n_link)}개 (최단 {r.nearest_m:.1f} m)"
        )
        for r in top10.itertuples()
    ]

    gu_stats = (
        cand_gdf.groupby("gu_nm", as_index=False)
        .agg(
            n_cand=("cid", "count"),
            dec_mean=("dEC_ha", "mean"),
            dec_max=("dEC_ha", "max"),
            conn_pct_mean=("conn_pct", "mean"),
        )
        .sort_values("dec_mean", ascending=False)
    )
    gu_lines = [
        (
            f"  {r.gu_nm:<4s}: 후보 {int(r.n_cand):2d}곳 | "
            f"평균 dEC {r.dec_mean:.4f} ha | 최대 dEC {r.dec_max:.4f} ha | "
            f"평균 경유 비중 {r.conn_pct_mean:.1f} %"
        )
        for r in gu_stats.itertuples()
    ]

    elapsed = time.perf_counter() - t0
    summary_lines = [
        "[프롬프트 3] 새 녹지 1 ha 후보지별 연결성 기여 요약",
        f"- 서울 내부 500 m 격자 정사각형: {n_in_seoul:,}개",
        f"- 세 조건 충족 후보지: {n_cand:,}개",
        f"- 링크 없는 후보지: {n_no_link}개",
        f"- 기준 Q = {base_q:,.0f} ha², 기준 EC = {base_ec:,.1f} ha",
        (
            f"- dEC_ha 통계: 최소 {dec_min:.4f}, 중앙값 {dec_median:.4f}, "
            f"평균 {dec_mean:.4f}, 최대 {dec_max:.4f} (단위 ha)"
        ),
        "- 상위 10개 후보지:",
        *top10_lines,
        "- 자치구별 평균 (평균 dEC_ha 내림차순):",
        *gu_lines,
        f"- 실행 시간: {elapsed:.1f}초",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_out_path.write_text(summary_text, encoding="utf-8")

    # 7. 시각화 (step3_map.png)
    configure_korean_font()
    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=200)

    gu_gdf.plot(ax=ax, color="#f7f7f7", edgecolor="#cccccc", linewidth=0.6, zorder=1)
    patch_gdf.plot(ax=ax, color="#c7e9c0", edgecolor="none", zorder=2)
    seoul_gdf.boundary.plot(ax=ax, color="#444444", linewidth=0.9, zorder=3)

    norm = mcolors.LogNorm(vmin=dec_min, vmax=dec_max)
    sc = ax.scatter(
        cand_gdf["x"],
        cand_gdf["y"],
        c=cand_gdf["dEC_ha"],
        cmap="plasma",
        norm=norm,
        marker="s",
        s=26,
        edgecolors="#222222",
        linewidths=0.25,
        zorder=4,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
    cbar_ticks = [t for t in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0] if dec_min <= t <= dec_max]
    if cbar_ticks:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
    cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    cbar.set_label("EC 증가량 dEC (ha, 로그 눈금)", fontsize=11)

    # 상위 10곳 순위 표시 (인접한 8위·9위 라벨 겹침 방지)
    for r in top10.itertuples():
        rank_num = int(r.rank)
        ax.scatter(
            [r.x],
            [r.y],
            marker="s",
            s=75,
            facecolors="none",
            edgecolors="#000000",
            linewidths=1.4,
            zorder=5,
        )
        offset = (6, -9) if rank_num == 9 else (5, 5)
        txt = ax.annotate(
            f"{rank_num}",
            xy=(r.x, r.y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
            zorder=6,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_title(
        f"새 녹지 1 ha 후보지({n_cand:,}곳)의 연결성 기여(dEC) 및 상위 10곳",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("X 좌표 (m, EPSG:5186)", fontsize=10)
    ax.set_ylabel("Y 좌표 (m, EPSG:5186)", fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(map_out_path, dpi=200)
    plt.close(fig)

    print("", flush=True)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
