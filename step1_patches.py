"""12주차 프롬프트 1: 서울 Sentinel-2 영상 기반 NDVI 계산 및 8-연결 녹지 패치 생성 스크립트."""

from collections import defaultdict
from pathlib import Path
import time

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid


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


def ensure_polygonal(geom):
    """도형 유효성을 보정하고 Polygon 또는 MultiPolygon만 반환한다."""
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [
            g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty
        ]
        if polys:
            return unary_union(polys)
    return geom


def main() -> None:
    t0 = time.perf_counter()
    base_dir = Path(".")
    tif_path = base_dir / "seoul_s2_20260516.tif"
    boundary_path = base_dir / "seoul_boundary.gpkg"

    ndvi_out_path = base_dir / "seoul_ndvi.tif"
    patches_out_path = base_dir / "green_patches.gpkg"
    summary_out_path = base_dir / "step1_summary.txt"
    map_out_path = base_dir / "step1_map.png"

    nodata_val = np.float32(-9999.0)
    ndvi_threshold = 0.6

    # 1. 입력 영상 읽기 및 NDVI 계산
    with rasterio.open(tif_path) as src:
        b04 = src.read(1).astype(np.float64)
        b08 = src.read(2).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        pixel_area_m2 = abs(transform.a * transform.e)

    valid_mask = (b04 > 0) & (b08 > 0)
    ndvi = np.full(b04.shape, nodata_val, dtype=np.float32)
    ndvi_valid_f64 = (b08[valid_mask] - b04[valid_mask]) / (b08[valid_mask] + b04[valid_mask])
    ndvi[valid_mask] = ndvi_valid_f64.astype(np.float32)

    # 2. seoul_ndvi.tif 저장 (Float32, NoData -9999)
    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        nodata=float(nodata_val),
        compress="lzw",
    )
    with rasterio.open(ndvi_out_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    # 3. 녹지 화소 판정 (NDVI >= 0.6) 및 8-연결 레이블링
    green_mask = valid_mask & (ndvi >= ndvi_threshold)
    structure_8 = np.ones((3, 3), dtype=np.int8)
    labeled, num_labels = ndimage.label(green_mask, structure=structure_8)

    # 화소 수 집계 (레이블 1 ~ num_labels)
    pixel_counts = np.bincount(labeled.ravel())[1:]
    single_pixel_patches = int(np.sum(pixel_counts == 1))

    # 4. 레이블별 폴리곤 생성 (패치 1개 = 폴리곤/멀티폴리곤 1행)
    geoms_by_label = defaultdict(list)
    for geom_dict, val in shapes(
        labeled.astype(np.int32),
        mask=(labeled > 0),
        transform=transform,
        connectivity=8,
    ):
        geoms_by_label[int(val)].append(shape(geom_dict))

    records = []
    for label_id in range(1, num_labels + 1):
        parts = geoms_by_label[label_id]
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        geom = ensure_polygonal(geom)
        area_m2 = geom.area
        ha = area_m2 / 10000.0
        centroid = geom.centroid
        records.append(
            {
                "area_m2": area_m2,
                "ha": ha,
                "cy": centroid.y,
                "cx": centroid.x,
                "geometry": geom,
            }
        )

    # 면적 내림차순, 같으면 중심점 y 내림차순, x 오름차순으로 pid 부여
    records.sort(key=lambda r: (-r["area_m2"], -r["cy"], r["cx"]))
    for idx, rec in enumerate(records, start=1):
        rec["pid"] = idx

    gdf = gpd.GeoDataFrame(
        {
            "pid": np.array([r["pid"] for r in records], dtype=np.int32),
            "ha": np.array([r["ha"] for r in records], dtype=np.float64),
        },
        geometry=[r["geometry"] for r in records],
        crs=crs,
    )
    gdf.to_file(patches_out_path, layer="patches", driver="GPKG")

    # 5. 통계 계산 및 step1_summary.txt 저장
    valid_count = int(np.sum(valid_mask))
    valid_area_km2 = valid_count * pixel_area_m2 / 1_000_000.0
    valid_area_ha = valid_count * pixel_area_m2 / 10_000.0

    valid_vals = ndvi[valid_mask].astype(np.float64)
    ndvi_mean = float(np.mean(valid_vals))
    ndvi_median = float(np.median(valid_vals))
    ndvi_min = float(np.min(valid_vals))
    ndvi_max = float(np.max(valid_vals))

    green_count = int(np.sum(green_mask))
    green_area_ha = green_count * pixel_area_m2 / 10_000.0
    green_pct = 100.0 * green_count / valid_count

    patch_count = len(gdf)
    top3_ha = gdf["ha"].iloc[:3].tolist()
    under_1ha = gdf[gdf["ha"] < 1.0]
    over_1ha = gdf[gdf["ha"] >= 1.0]

    elapsed = time.perf_counter() - t0

    summary_lines = [
        "[프롬프트 1] NDVI 계산과 녹지 패치 요약",
        f"- 자료 화소: {valid_count:,}개 ({valid_area_km2:.2f} km², {valid_area_ha:.2f} ha)",
        (
            f"- NDVI 통계: 평균 {ndvi_mean:.3f}, 중앙값 {ndvi_median:.3f}, "
            f"최소 {ndvi_min:.3f}, 최대 {ndvi_max:.3f}"
        ),
        (
            f"- 녹지 화소 (NDVI >= {ndvi_threshold}): {green_count:,}개 = "
            f"{green_area_ha:,.1f} ha = 서울의 {green_pct:.1f} %"
        ),
        f"- 패치 수 (8-연결): {patch_count:,}개",
        (
            f"- 상위 3개 패치 면적: 1위 {top3_ha[0]:,.1f} ha, "
            f"2위 {top3_ha[1]:,.1f} ha, 3위 {top3_ha[2]:,.1f} ha"
        ),
        f"- 화소 1개 패치: {single_pixel_patches:,}개",
        f"- 1 ha 미만 패치: {len(under_1ha):,}개 (면적 합 {under_1ha['ha'].sum():,.1f} ha)",
        f"- 1 ha 이상 패치: {len(over_1ha):,}개 (면적 합 {over_1ha['ha'].sum():,.1f} ha)",
        f"- 전체 패치 면적 합: {gdf['ha'].sum():,.1f} ha",
        f"- 실행 시간: {elapsed:.1f}초",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_out_path.write_text(summary_text, encoding="utf-8")

    # 6. 시각화 (step1_map.png)
    font_name = configure_korean_font()
    seoul_gdf = gpd.read_file(boundary_path, layer="seoul")
    gu_gdf = gpd.read_file(boundary_path, layer="gu")

    ndvi_plot = np.where(valid_mask, ndvi, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=200)

    # 왼쪽: NDVI 지도
    im = ax1.imshow(
        ndvi_plot,
        cmap="RdYlGn",
        vmin=-0.4,
        vmax=0.9,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
    )
    seoul_gdf.boundary.plot(ax=ax1, color="#333333", linewidth=0.8)
    ax1.set_title("서울 NDVI (2026-05-16)", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("X 좌표 (m, EPSG:5186)", fontsize=10)
    ax1.set_ylabel("Y 좌표 (m, EPSG:5186)", fontsize=10)
    ax1.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.03)
    cbar.set_label("NDVI", fontsize=10)

    # 오른쪽: 녹지 패치 지도
    gu_gdf.plot(ax=ax2, color="#f5f5f5", edgecolor="#cccccc", linewidth=0.6)
    gdf.plot(ax=ax2, color="#1b7837", edgecolor="none")
    seoul_gdf.boundary.plot(ax=ax2, color="#333333", linewidth=0.9)
    ax2.set_title(
        f"서울 녹지 패치 (NDVI ≥ {ndvi_threshold}, {patch_count:,}개)",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("X 좌표 (m, EPSG:5186)", fontsize=10)
    ax2.set_ylabel("Y 좌표 (m, EPSG:5186)", fontsize=10)
    ax2.set_aspect("equal")

    fig.suptitle(
        f"서울시 녹지 분포 및 패치 추출 결과 (글꼴: {font_name})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(map_out_path, dpi=200)
    plt.close(fig)

    print(summary_text, end="")


if __name__ == "__main__":
    main()
