import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.graph_objects as go

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

FIELD_INFO_FONT_SIZE = 18  # change this number to make field info text bigger/smaller

PSI_COLORS_HEX = {"low": "#2E7D32", "moderate": "#FBC02D", "high": "#D32F2F"}

# Colors for the overall rating text
RATING_COLOR = {
    "Low": PSI_COLORS_HEX["low"],          # green
    "Moderate": PSI_COLORS_HEX["moderate"],# amber
    "High": PSI_COLORS_HEX["high"],        # red
    "Severe": "#8B0000",                   # dark red
}


BAR_COLOR = "#D2B48C"  # tan bar color for 'Average PSI by Interval'

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
    
# --- small helper for colored percentage rows ---
def metric_row(label: str, value_text: str, color_hex: str):
    st.markdown(f"<div style='font-size:12px;color:gray;margin-bottom:2px;'>{label}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:20px;font-weight:700;color:{color_hex};margin-bottom:10px;'>{value_text}</div>",
        unsafe_allow_html=True
    )
# --- helper to draw the single-point profile: Depth on X, PSI on Y ---
def make_profile_figure(profile_df: pd.DataFrame,
                        colors_hex: dict,
                        thresholds: dict) -> go.Figure:
    """
    profile_df: tidy rows for ONE point, columns include Depth_in (int) and PSI (float)
    colors_hex: dict with keys low/moderate/high -> hex
    thresholds: dict { "moderate": 200, "high": 300 }
    """
    LOW  = thresholds["moderate"]
    HIGH = thresholds["high"]

    df = profile_df.sort_values("Depth_in", kind="stable").copy()
    zs = df["Depth_in"].to_numpy()    # Depth (in) -> X
    ps = df["PSI"].to_numpy()         # PSI       -> Y

    # Build line segments, splitting at 200/300 cross-overs
    bands = {"low": {"x": [], "y": []},
             "moderate": {"x": [], "y": []},
             "high": {"x": [], "y": []}}

    if len(ps) >= 2:
        for i in range(len(ps) - 1):
            z1, z2 = zs[i], zs[i+1]
            p1, p2 = ps[i], ps[i+1]
            pts = [(z1, p1)]  # (Depth, PSI)

            def add_cross(thr):
                # add exact crossing (linear)
                if (p1 - thr) * (p2 - thr) < 0:
                    t  = (thr - p1) / (p2 - p1)
                    zt = z1 + t * (z2 - z1)
                    pts.append((zt, thr))

            add_cross(LOW)
            add_cross(HIGH)
            pts.append((z2, p2))

            for j in range(len(pts) - 1):
                za, pa = pts[j]
                zb, pb = pts[j+1]
                mid = 0.5 * (pa + pb)
                key = "low" if mid <= LOW else ("moderate" if mid <= HIGH else "high")
                bands[key]["x"].extend([za, zb, np.nan])  # Depth on X
                bands[key]["y"].extend([pa, pb, np.nan])  # PSI on Y

       # Add traces for each band (always in legend, even if empty)
        for name in ["low", "moderate", "high"]:
            fig_prof.add_trace(
                go.Scatter(
                    x=bands[name]["x"] if bands[name]["x"] else [None],
                    y=bands[name]["y"] if bands[name]["y"] else [None],
                    mode="lines+markers",
                    line=dict(color=band_colors[name], width=3),
                    marker=dict(size=6),
                    name=name.capitalize(),
                    hovertemplate="PSI=%{x:.1f}<br>Depth=%{y:.1f} in<extra></extra>",
                    showlegend=True
                )
            )


    fig.update_xaxes(title="Depth (in)")
    fig.update_yaxes(title="PSI")
    # Horizontal guides at 200, 300 PSI
    fig.add_hline(y=LOW,  line_dash="dash", line_color=colors_hex["moderate"])
    fig.add_hline(y=HIGH, line_dash="dash", line_color=colors_hex["high"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


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
        st.caption("Field Name")
        st.markdown(
        f"<p style='font-size:{FIELD_INFO_FONT_SIZE}px;'>{field_name}</p>",
        unsafe_allow_html=True
        )
    with c_fi2:
        st.caption("Date of Sampling")
        st.markdown(
        f"<p style='font-size:{FIELD_INFO_FONT_SIZE}px;'>{sample_date.strftime('%Y-%m-%d')}</p>",
        unsafe_allow_html=True
        )
    with c_fi3:
        st.markdown(
        "<p style='font-size:14px; color:gray;'># Of Collected Compaction Points</p>",
        unsafe_allow_html=True
        )
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

   # % of points by compaction level (0–11 in range) using max PSI per point
    subset = long[long["Depth_in"].between(0, 11)].groupby(id_col)["PSI"].max()

    below_mod = (subset <= PSI_THRESHOLDS["moderate"]).mean() * 100  # ≤200
    exceed_mod = (subset >  PSI_THRESHOLDS["moderate"]).mean() * 100  # >200
    exceed_high = (subset >  PSI_THRESHOLDS["high"]).mean() * 100     # >300

    def overall_rating(x_percent_high):
        if x_percent_high >= 50:
            return "Severe"
        if x_percent_high >= 20:
            return "High"
        if x_percent_high >= 5:
            return "Moderate"
        return "Low"

    rating = overall_rating(exceed_high)


    c1, c2 = st.columns([2, 1])
    with c2:
        def safe_fmt(v): 
            return f"{v:.0f}" if pd.notna(v) else "–"

        st.metric("Highest PSI (0–3 in)", safe_fmt(top_df.loc[top_df["Interval"]=="0–3 in","Highest_PSI"].values[0]))
        st.metric("Highest PSI (4–7 in)", safe_fmt(top_df.loc[top_df["Interval"]=="4–7 in","Highest_PSI"].values[0]))
        st.metric("Highest PSI (8–11 in)", safe_fmt(top_df.loc[top_df["Interval"]=="8–11 in","Highest_PSI"].values[0]))

    # New percentage metrics
        metric_row(f"% points ≤{PSI_THRESHOLDS['moderate']} PSI (0–11)", f"{below_mod:.1f}%",  PSI_COLORS_HEX["low"])
        metric_row(f"% points >{PSI_THRESHOLDS['moderate']} PSI (0–11)", f"{exceed_mod:.1f}%", PSI_COLORS_HEX["moderate"])
        metric_row(f"% points >{PSI_THRESHOLDS['high']} PSI (0–11)",    f"{exceed_high:.1f}%", PSI_COLORS_HEX["high"])

        # Colored Overall Rating
        st.markdown("<div style='font-size:12px;color:gray;margin-top:8px;'>Overall Compaction Rating</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:32px;font-weight:800;color:{RATING_COLOR.get(rating, 'white')};'>{rating}</div>",
        unsafe_allow_html=True
        )



    with c1:
        st.subheader("Field Average by Interval")
        fig_rep = px.bar(field_avg, x="Interval", y="PSI_mean", text="PSI_mean", title="Average PSI by Interval (All Points)")
        fig_rep.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_color=BAR_COLOR, opacity=0.95)
        fig_rep.add_hline(y=PSI_THRESHOLDS["low"],      line_dash="dash", line_color=PSI_COLORS_HEX["low"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["moderate"], line_dash="dash", line_color=PSI_COLORS_HEX["moderate"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["high"],     line_dash="dash", line_color=PSI_COLORS_HEX["high"])
        st.plotly_chart(fig_rep, use_container_width=True) 

    # --- Depth profile (single point) ---
   
    st.subheader("Depth profile (single point)")

    point_choices = sorted(long[id_col].dropna().unique().tolist())
    chosen_point  = st.selectbox("Point", options=point_choices, index=0, key="profile_point")

    profile_df = (
        long[long[id_col] == chosen_point]
        .sort_values("Depth_in", kind="stable")
    )

    st.caption(f"Records for point {chosen_point}: {profile_df.shape[0]} depth rows")

    if profile_df.shape[0] < 2:
        st.info("Not enough depth readings to draw a profile line for this point.")
    else:
        fig_prof = make_profile_figure(profile_df, PSI_COLORS_HEX, PSI_THRESHOLDS)
        fig_prof.update_layout(title=f"Point {chosen_point} — PSI vs Depth")
        st.plotly_chart(fig_prof, use_container_width=True)

    # Interval Averages per Point

    # --- Interval Averages per Point (collapsible) ---
exp_interval = st.expander("Interval Averages per Point", expanded=False)
with exp_interval:
    st.subheader("Interval Averages per Point")

    # Build the per-point interval averages directly from `long`
    # Assumptions: `long` has columns [id_col, "Depth_in", "PSI"]
    per_point_df = (
        long[long["Depth_in"].between(0, 3)].groupby(id_col)["PSI"].mean()
            .rename("PSI_avg_0–3 in").to_frame()
        .join(
            long[long["Depth_in"].between(4, 7)].groupby(id_col)["PSI"].mean()
                .rename("PSI_avg_4–7 in")
        )
        .join(
            long[long["Depth_in"].between(8, 11)].groupby(id_col)["PSI"].mean()
                .rename("PSI_avg_8–11 in")
        )
        .reset_index()
        .sort_values(id_col, kind="stable")
    )

    st.dataframe(per_point_df, use_container_width=True)

    # CSV download
    per_point_csv = per_point_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download interval averages (CSV)",
        data=per_point_csv,
        file_name="interval_averages.csv",
        mime="text/csv",
    )

    # Depth Explorer (by the 3 intervals)
    # --- Depth Explorer (by interval) (collapsible) ---
exp_depth = st.expander("Depth Explorer (by interval)", expanded=False)
with exp_depth:
    st.subheader("Depth Explorer (by interval)")

    # Choose one of the three averaged depth bands
    interval_options = {
        "0–3 in":  (0, 3),
        "4–7 in":  (4, 7),
        "8–11 in": (8, 11),
    }
    depth_choice = st.selectbox("Choose interval", list(interval_options.keys()), index=0)
    lo, hi = interval_options[depth_choice]

    # Filter to the chosen interval (assumes long has columns: [id_col, "Depth_in", "PSI"])
    sub = long[long["Depth_in"].between(lo, hi)]

    # Overall summary for that interval
    summary_df = (
        sub["PSI"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_frame()
        .T
    )
    st.dataframe(summary_df, use_container_width=True)

    # Per-point stats for that interval
    by_point_df = (
        sub.groupby(id_col)["PSI"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values(id_col, kind="stable")
    )
    st.dataframe(by_point_df, use_container_width=True)

    # Optional download
    # interval_csv = by_point_df.to_csv(index=False).encode("utf-8")
    # st.download_button(
    #     f"⬇️ Download {depth_choice} table (CSV)",
    #     data=interval_csv,
    #     file_name=f"depth_{lo}-{hi}_stats.csv",
    #     mime="text/csv",
    # )

    # --- Export ---
    st.subheader("Export")

    # Always-available long/tidy data
    st.download_button(
        "⬇️ Download long/tidy data (CSV)",
        data=long.to_csv(index=False).encode("utf-8"),
        file_name="compaction_long.csv",
        mime="text/csv",
    )

    # Recreate the selected interval based on the saved choice
    interval_options = {
        "0–3 in":  (0, 3),
        "4–7 in":  (4, 7),
        "8–11 in": (8, 11),
    }
    choice = st.session_state.get("depth_choice", "0–3 in")  # fallback if expander not opened yet
    lo, hi = interval_options[choice]
    sel_long = long[long["Depth_in"].between(lo, hi)]

    # Clean filename from choice (remove spaces, normalize en-dash)
    clean = choice.replace(" ", "").replace("–", "-")

    st.download_button(
        f"⬇️ Download selection ({choice}) (CSV)",
        data=sel_long.to_csv(index=False).encode("utf-8"),
        file_name=f"compaction_{clean}.csv",
        mime="text/csv",
    )


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
