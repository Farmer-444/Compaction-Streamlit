import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Map + interpolation helpers
import folium
from streamlit_folium import st_folium
from matplotlib import colors
from PIL import Image
import base64

st.set_page_config(page_title="Compaction Explorer", layout="wide")
st.title("🧱 Soil Compaction Explorer")
st.caption("Upload CSV/XLSX with penetrometer depths (e.g., 0in…11in), optional Lat/Lon, explore & map.")

# -------------------------
# Config (easy to change)
# -------------------------
PSI_THRESHOLDS = {"low": 150, "moderate": 200, "high": 300}
PSI_COLORS = {
    "low": (46, 125, 50),       # green
    "moderate": (251, 192, 45), # yellow
    "high": (211, 47, 47),      # red
}

FIELD_INFO_FONT_SIZE = 14  # change this number to make field info text bigger/smaller

PSI_COLORS_HEX = {"low": "#2E7D32", "moderate": "#FBC02D", "high": "#D32F2F"}

def psi_band(psi: float) -> str:
    if psi <= PSI_THRESHOLDS["low"]:
        return "low"
    if psi <= PSI_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"

def band_rgb_tuple(psi: float):
    return PSI_COLORS[psi_band(psi)]

def band_hex(psi: float):
    r, g, b = band_rgb_tuple(psi)
    return f"#{r:02x}{g:02x}{b:02x}"

# -------------------------
# Helpers
# -------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

POSSIBLE_ID  = ["Point Number", "Point", "ID", "Sample", "Station"]
POSSIBLE_LAT = ["lat", "latitude", "Lat", "Latitude"]
POSSIBLE_LON = ["lon", "long", "longitude", "Lon", "Longitude"]
POSSIBLE_ZONE= ["zone", "Zone", "swat_zone", "class"]

def infer(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in low:
            return low[k.lower()]
    for c in cols:
        for k in candidates:
            if k.lower() in c.lower():
                return c
    return None

def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    return pd.read_csv(file)

def detect_depth_columns(cols: List[str]) -> List[str]:
    depth_cols = []
    for c in cols:
        cs = c.replace(" ", "")
        if cs.endswith("in") and cs[:-2].isdigit():
            depth_cols.append(c)
    return depth_cols

def reshape_long(df: pd.DataFrame, id_col: str, lat_col: Optional[str], lon_col: Optional[str], zone_col: Optional[str]) -> Tuple[pd.DataFrame, List[int]]:
    df = clean_columns(df)
    cols = df.columns.tolist()
    depth_cols = detect_depth_columns(cols)
    if not depth_cols:
        st.error("No depth columns detected (e.g., 0in … 11in). Please check headers.")
        st.stop()

    use_cols = [id_col] + depth_cols
    if lat_col: use_cols.append(lat_col)
    if lon_col: use_cols.append(lon_col)
    if zone_col: use_cols.append(zone_col)

    work = df[use_cols].copy()
    long = work.melt(
        id_vars=[c for c in [id_col, lat_col, lon_col, zone_col] if c],
        value_vars=depth_cols,
        var_name="Depth_label",
        value_name="PSI"
    )
    long["PSI"] = pd.to_numeric(long["PSI"], errors="coerce")
    long = long.dropna(subset=["PSI"])

    long["Depth_in"] = (
        long["Depth_label"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("in", "", regex=False)
        .astype(int)
    )
    long["Depth_cm"] = long["Depth_in"] * 2.54
    return long, sorted(long["Depth_in"].unique().tolist())

# -------------------------
# Tabs (Report View + Map)
# -------------------------
tabs = st.tabs(["Report View", "Map"])

with st.sidebar:
    st.header("0) Field information")
    field_name = st.text_input("Field Name *", placeholder="e.g., VPF4")
    sample_date = st.date_input("Date of sampling *")

    st.header("1) Upload data")
    up = st.file_uploader("CSV / TSV / Excel with depth columns (0in…11in)", type=["csv", "tsv", "xlsx", "xls"])


# Require field name + date BEFORE doing anything with the CSV
if not field_name or not sample_date:
    st.info("Enter **Field Name** and **Date of sampling** in the sidebar to continue.")
    st.stop()

if up is None:
    st.info("Upload a CSV/XLSX to continue.")
    st.stop()


raw = read_any(up)
raw = clean_columns(raw)


st.success(f"Loaded {len(raw):,} rows × {len(raw.columns)} columns")
st.dataframe(raw.head(10), use_container_width=True)

with st.sidebar:
    st.divider()
    st.header("2) Map columns")
    cols = raw.columns.tolist()
    id_guess  = infer(cols, POSSIBLE_ID)  or cols[0]
    lat_guess = infer(cols, POSSIBLE_LAT)
    lon_guess = infer(cols, POSSIBLE_LON)
    zone_guess= infer(cols, POSSIBLE_ZONE)

    id_col  = st.selectbox("Point ID", options=cols, index=cols.index(id_guess))
    lat_col = st.selectbox("Latitude (optional)",  ["<none>"] + cols, index=(0 if not lat_guess else (cols.index(lat_guess) + 1)))
    lon_col = st.selectbox("Longitude (optional)", ["<none>"] + cols, index=(0 if not lon_guess else (cols.index(lon_guess) + 1)))
    zone_col= st.selectbox("Zone (optional)",      ["<none>"] + cols, index=(0 if not zone_guess else (cols.index(zone_guess) + 1)))

    lat_col = None if lat_col == "<none>" else lat_col
    lon_col = None if lon_col == "<none>" else lon_col
    zone_col= None if zone_col == "<none>" else zone_col

# Reshape to tidy
long, depth_levels = reshape_long(raw, id_col, lat_col, lon_col, zone_col)

# Interval definitions (used everywhere)
intervals = {
    "0–3 in": (0, 3),
    "4–7 in": (4, 7),
    "8–11 in": (8, 11),
}

# Per-point interval averages
avg_records = []
for name, (lo, hi) in intervals.items():
    temp = (
        long[long["Depth_in"].between(lo, hi)]
        .groupby(id_col)["PSI"].mean()
        .reset_index()
        .rename(columns={"PSI": f"PSI_avg_{name}"})
    )
    avg_records.append(temp)

avg_df = avg_records[0]
for other in avg_records[1:]:
    avg_df = avg_df.merge(other, on=id_col, how="outer")

# Field-level averages across all points
field_avg = []
for name, (lo, hi) in intervals.items():
    mask = (long["Depth_in"] >= lo) & (long["Depth_in"] <= hi)
    field_avg.append({"Interval": name, "PSI_mean": long.loc[mask, "PSI"].mean()})
field_avg = pd.DataFrame(field_avg)

# -------------------------
# TAB 1: REPORT VIEW  (Profiles merged)
# -------------------------
with tabs[0]:
    # ---------- Field Information ----------
    st.subheader("Field Information")
    # num_points will be computed after the ID column is chosen; we can compute here because mapping is done already
    try:
        num_points = int(pd.Series(raw[id_col]).nunique())
    except Exception:
        num_points = int(raw.shape[0])  # fallback

    c_fi1, c_fi2, c_fi3 = st.columns([1.2, 1.2, 1])
    with c_fi1:
        st.caption("Field name")
        st.markdown(
        f"<p style='font-size:{FIELD_INFO_FONT_SIZE}px;'>{field_name}</p>",
        unsafe_allow_html=True
        )
    with c_fi2:
        st.caption("Date of sampling")
        st.markdown(
        f"<p style='font-size:{FIELD_INFO_FONT_SIZE}px;'>{sample_date.strftime('%Y-%m-%d')}</p>",
        unsafe_allow_html=True
        )
    with c_fi3:
        st.caption("# of collected compaction points")
        st.markdown(
        f"<p style='font-size:{FIELD_INFO_FONT_SIZE}px;'>{num_points:,}</p>",
        unsafe_allow_html=True
        )


    st.divider()
    # ---------- Field Summary ----------
    st.subheader("Field Summary")

    top_rows = []
    for name, (lo, hi) in intervals.items():
        m = long[long["Depth_in"].between(lo, hi)]
        top_rows.append({"Interval": name, "Highest_PSI": m["PSI"].max() if not m.empty else np.nan})
    top_df = pd.DataFrame(top_rows)

    exceed = (long[long["Depth_in"].between(0, 11)]
              .groupby(id_col)["PSI"].max() > PSI_THRESHOLDS["high"]).mean() * 100

    def overall_rating(x):
        if x >= 50: return "Severe"
        if x >= 20: return "High"
        if x >= 5:  return "Moderate"
        return "Low"

    rating = overall_rating(exceed)

    c1, c2 = st.columns([2, 1])
    with c2:
        def safe_fmt(v): return f"{v:.0f}" if pd.notna(v) else "—"
        st.metric("Highest PSI (0–3 in)", safe_fmt(top_df.loc[top_df["Interval"]=="0–3 in","Highest_PSI"].values[0]))
        st.metric("Highest PSI (4–7 in)", safe_fmt(top_df.loc[top_df["Interval"]=="4–7 in","Highest_PSI"].values[0]))
        st.metric("Highest PSI (8–11 in)", safe_fmt(top_df.loc[top_df["Interval"]=="8–11 in","Highest_PSI"].values[0]))
        st.metric(f"% points >{PSI_THRESHOLDS['high']} PSI (0–11)", f"{exceed:.1f}%")
        st.metric("Overall Compaction Rating", rating)

    with c1:
        st.subheader("Field Average by Interval")
        fig_rep = px.bar(field_avg, x="Interval", y="PSI_mean", text="PSI_mean", title="Average PSI by Interval (All Points)")
        fig_rep.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rep.add_hline(y=PSI_THRESHOLDS["low"],      line_dash="dash", line_color=PSI_COLORS_HEX["low"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["moderate"], line_dash="dash", line_color=PSI_COLORS_HEX["moderate"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["high"],     line_dash="dash", line_color=PSI_COLORS_HEX["high"])
        st.plotly_chart(fig_rep, use_container_width=True)

    # Interval Averages per Point
    st.subheader("Interval Averages per Point")
    st.dataframe(avg_df, use_container_width=True)
    st.download_button("⬇️ Download interval averages (CSV)", data=avg_df.to_csv(index=False).encode("utf-8"),
                       file_name="interval_averages.csv", mime="text/csv")

    # Depth Explorer (by the 3 intervals)
    st.subheader("Depth Explorer (by interval)")
    chosen_interval = st.selectbox("Choose interval", list(intervals.keys()), index=0)
    lo, hi = intervals[chosen_interval]
    sel_long = long[long["Depth_in"].between(lo, hi)].copy()

    left, right = st.columns(2)
    with left:
        overall = sel_long["PSI"].agg(["count","mean","median","std","min","max"]).to_frame().T
        st.caption(chosen_interval)
        st.dataframe(overall, use_container_width=True)
    with right:
        by_point = sel_long.groupby(id_col)["PSI"].agg(["count","mean","median","std","min","max"]).reset_index()
        st.dataframe(by_point, use_container_width=True)

    # Per-point depth profile (still useful)
    st.subheader("Depth profile (single point)")
    point_choices = sorted(long[id_col].unique().tolist())
    chosen_point = st.selectbox("Point", options=point_choices, index=0)
    prof = long[long[id_col] == chosen_point].sort_values("Depth_in")
    fig_line = px.line(prof, x="Depth_in", y="PSI", markers=True,
                       title=f"Point {chosen_point} – Full depth profile")
    fig_line.update_layout(xaxis_title="Depth (in)", yaxis_title="PSI")
    st.plotly_chart(fig_line, use_container_width=True)

    # Export
    st.subheader("Export")
    st.download_button("⬇️ Download long/tidy data (CSV)",
                       data=long.to_csv(index=False).encode("utf-8"),
                       file_name="compaction_long.csv", mime="text/csv")
    st.download_button(f"⬇️ Download selection ({chosen_interval}) (CSV)",
                       data=sel_long.to_csv(index=False).encode("utf-8"),
                       file_name=f"compaction_{chosen_interval.replace(' ','').replace('–','-')}.csv",
                       mime="text/csv")

# -------------------------
# TAB 2: MAP (Satellite + IDW interpolation)
# -------------------------
with tabs[1]:
    st.subheader("Compaction Map (Satellite + Interpolation)")
    if lat_col and lon_col:
        map_interval = st.selectbox("Depth interval", options=list(intervals.keys()), index=0)
        lo, hi = intervals[map_interval]
        base = long[long["Depth_in"].between(lo, hi)]
        avg_map = (
            base.groupby(id_col)
            .agg({"PSI":"mean", lat_col:"first", lon_col:"first"})
            .reset_index()
            .rename(columns={lat_col:"lat", lon_col:"lon"})
        )

        if avg_map.empty:
            st.info("No rows available for the selected interval.")
        else:
            # Build Folium map with Esri World Imagery
            center = [float(avg_map["lat"].mean()), float(avg_map["lon"].mean())]
            m = folium.Map(location=center, zoom_start=15, tiles=None)
            folium.TileLayer(
                tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery", name="Satellite"
            ).add_to(m)

            # Grid bounds
            lat_min, lat_max = avg_map["lat"].min(), avg_map["lat"].max()
            lon_min, lon_max = avg_map["lon"].min(), avg_map["lon"].max()
            pad_lat = (lat_max - lat_min) * 0.1 or 0.0005
            pad_lon = (lon_max - lon_min) * 0.1 or 0.0005
            lat_lin = np.linspace(lat_min - pad_lat, lat_max + pad_lat, 150)
            lon_lin = np.linspace(lon_min - pad_lon, lon_max + pad_lon, 150)
            grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)

            # IDW interpolation (power=2)
            gx = grid_lon.flatten(); gy = grid_lat.flatten()
            xs = avg_map["lon"].to_numpy(); ys = avg_map["lat"].to_numpy(); zs = avg_map["PSI"].to_numpy()
            d2 = (gx[:,None]-xs[None,:])**2 + (gy[:,None]-ys[None,:])**2
            d2[d2 == 0] = 1e-12
            w = 1.0 / d2
            z_idw = (w * zs[None,:]).sum(axis=1) / w.sum(axis=1)
            Z = z_idw.reshape(grid_lat.shape)

            # Colorize to PNG bytes
            norm = colors.Normalize(vmin=max(0, float(np.nanmin(Z))), vmax=float(np.nanmax(Z)))
            cmap = colors.LinearSegmentedColormap.from_list("gyr", ["#2E7D32", "#FBC02D", "#D32F2F"])
            rgba = cmap(norm(Z), bytes=True)  # (H,W,4)
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            # Convert PNG bytes -> data URL string (Folium needs a serializable value)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"

            # Overlay (using data URL)
            bounds = [[lat_lin[0], lon_lin[0]], [lat_lin[-1], lon_lin[-1]]]
            folium.raster_layers.ImageOverlay(
                image=data_url,          # <-- pass data URL, not bytes
                bounds=bounds,
                opacity=0.45,
                name=f"IDW {map_interval}",
                origin="upper",
            ).add_to(m)


            # Point markers
            for _, r in avg_map.iterrows():
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=4,
                    weight=1,
                    color="#000000",
                    fill=True,
                    fill_opacity=0.9,
                    fill_color=band_hex(float(r["PSI"])),
                    tooltip=f"{id_col}: {r[id_col]} | {r['PSI']:.0f} PSI",
                ).add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, width=None, height=600)
    else:
        st.info("Latitude/Longitude not provided — map disabled.")
