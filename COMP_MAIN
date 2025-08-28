import io
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Compaction Explorer", layout="wide")
st.title("🧱 Soil Compaction Explorer")
st.caption("Upload CSV/XLSX with penetrometer depths (e.g., 0in…11in), optional Lat/Lon, explore & map.")

# -------------------------
# Configuration (easy to change thresholds/colors)
# -------------------------
PSI_THRESHOLDS = {
    "low": 150,
    "moderate": 200,
    "high": 300,
}

PSI_COLORS = {
    "low": [46, 125, 50],       # green
    "moderate": [251, 192, 45], # yellow
    "high": [211, 47, 47],      # red
}

PSI_COLORS_HEX = {
    "low": "#2E7D32",
    "moderate": "#FBC02D",
    "high": "#D32F2F",
}

def psi_band(psi: float) -> str:
    if psi <= PSI_THRESHOLDS["low"]:
        return "low"
    if psi <= PSI_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"

def psi_rgb(psi: float):
    return PSI_COLORS[psi_band(psi)]

def psi_kml_color(psi: float) -> str:
    band = psi_band(psi)
    if band == "low":
        return "ff327d2e"  # green (aabbggrr)
    if band == "moderate":
        return "ff2dc0fb"  # yellow
    return "ff2f2fd3"      # red

# -------------------------
# Helpers
# -------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

POSSIBLE_ID = ["Point Number", "Point", "ID", "Sample", "Station"]
POSSIBLE_LAT = ["lat", "latitude", "Lat", "Latitude"]
POSSIBLE_LON = ["lon", "long", "longitude", "Lon", "Longitude"]
POSSIBLE_ZONE = ["zone", "Zone", "swat_zone", "class"]

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

    long["Depth_in"] = long["Depth_label"].astype(str).str.replace(" ", "", regex=False).str.replace("in", "", regex=False).astype(int)
    long["Depth_cm"] = long["Depth_in"] * 2.54
    return long, sorted(long["Depth_in"].unique().tolist())

# -------------------------
# UI
# -------------------------
tabs = st.tabs(["Profiles & Averages", "Report View", "Map"])

with st.sidebar:
    st.header("1) Upload data")
    up = st.file_uploader("CSV / TSV / Excel with depth columns (0in…11in)", type=["csv", "tsv", "xlsx", "xls"]) 
    if up is None:
        st.info("Upload a file to get started. Headers like: 'Point Number, Latitude, Longitude, 0in, 1in, …'")

if up is None:
    st.stop()

raw = read_any(up)
raw = clean_columns(raw)

# Title / Field name prompt
field_name = st.text_input("Field Name", placeholder="Enter field name (e.g., VPF4)")
if field_name:
    st.subheader(f"Field: {field_name}")

st.success(f"Loaded {len(raw):,} rows × {len(raw.columns)} columns")
st.dataframe(raw.head(10), use_container_width=True)

with st.sidebar:
    st.divider()
    st.header("2) Map columns")
    cols = raw.columns.tolist()
    id_guess = infer(cols, POSSIBLE_ID) or cols[0]
    lat_guess = infer(cols, POSSIBLE_LAT)
    lon_guess = infer(cols, POSSIBLE_LON)
    zone_guess = infer(cols, POSSIBLE_ZONE)

    id_col = st.selectbox("Point ID", options=cols, index=cols.index(id_guess))
    lat_col = st.selectbox("Latitude (optional)", ["<none>"] + cols, index=(0 if not lat_guess else (cols.index(lat_guess) + 1)))
    lon_col = st.selectbox("Longitude (optional)", ["<none>"] + cols, index=(0 if not lon_guess else (cols.index(lon_guess) + 1)))
    zone_col = st.selectbox("Zone (optional)", ["<none>"] + cols, index=(0 if not zone_guess else (cols.index(zone_guess) + 1)))

    lat_col = None if lat_col == "<none>" else lat_col
    lon_col = None if lon_col == "<none>" else lon_col
    zone_col = None if zone_col == "<none>" else zone_col

# Reshape
long, depth_levels = reshape_long(raw, id_col, lat_col, lon_col, zone_col)

# -------------------------
# Interval Averages
# -------------------------
intervals = {
    "0–3 in": (0, 3),
    "4–7 in": (4, 7),
    "8–11 in": (8, 11),
}

avg_records = []
for name, (lo, hi) in intervals.items():
    mask = (long["Depth_in"] >= lo) & (long["Depth_in"] <= hi)
    temp = (
        long[mask].groupby(id_col)["PSI"].mean().reset_index().rename(columns={"PSI": f"PSI_avg_{name}"})
    )
    avg_records.append(temp)

avg_df = avg_records[0]
for other in avg_records[1:]:
    avg_df = avg_df.merge(other, on=id_col, how="outer")

field_avg = []
for name, (lo, hi) in intervals.items():
    mask = (long["Depth_in"] >= lo) & (long["Depth_in"] <= hi)
    field_avg.append({"Interval": name, "PSI_mean": long.loc[mask, "PSI"].mean()})
field_avg = pd.DataFrame(field_avg)

# -------------------------
# Tab 1: Profiles & Averages
# -------------------------
with tabs[0]:
    st.subheader("Interval Averages per Point")
    st.dataframe(avg_df, use_container_width=True)
    csv_bytes = avg_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download interval averages (CSV)", data=csv_bytes, file_name="interval_averages.csv", mime="text/csv")

    st.subheader("Field Average by Interval")
    fig_avg = px.bar(field_avg, x="Interval", y="PSI_mean", text="PSI_mean", title="Average PSI by Interval (All Points)")
    fig_avg.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_avg.add_hline(y=PSI_THRESHOLDS["low"], line_dash="dash", line_color=PSI_COLORS_HEX["low"])
    fig_avg.add_hline(y=PSI_THRESHOLDS["moderate"], line_dash="dash", line_color=PSI_COLORS_HEX["moderate"])
    fig_avg.add_hline(y=PSI_THRESHOLDS["high"], line_dash="dash", line_color=PSI_COLORS_HEX["high"])
    st.plotly_chart(fig_avg, use_container_width=True)

# -------------------------
# Tab 2: Report View
# -------------------------
with tabs[1]:
    st.subheader("Field Summary")
    surf_mask = (long["Depth_in"] >= 0) & (long["Depth_in"] <= 6)
    subs_mask = (long["Depth_in"] >= 7) & (long["Depth_in"] <= 11)
    high_surface = long.loc[surf_mask, "PSI"].max()
    high_subsurface = long.loc[subs_mask, "PSI"].max()
    exceed = (long[long["Depth_in"].between(0, 11)].groupby(id_col)["PSI"].max() > PSI_THRESHOLDS["high"]).mean() * 100

    def overall_rating(x):
        if x >= 50:
            return "Severe"
        if x >= 20:
            return "High"
        if x >= 5:
            return "Moderate"
        return "Low"

    rating = overall_rating(exceed)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.metric("Highest PSI (0–6 in)", f"{high_surface:.0f}")
        st.metric("Highest PSI (7–11 in)", f"{high_subsurface:.0f}")
        st.metric(f"% points >{PSI_THRESHOLDS['high']} PSI (0–11)", f"{exceed:.1f}%")
        st.metric("Overall Compaction Rating", rating)
    with c1:
        fig_rep = px.area(field_avg, x="Interval", y="PSI_mean", title="Average PSI by Interval")
        fig_rep.add_hline(y=PSI_THRESHOLDS["low"], line_dash="dash", line_color=PSI_COLORS_HEX["low"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["moderate"], line_dash="dash", line_color=PSI_COLORS_HEX["moderate"])
        fig_rep.add_hline(y=PSI_THRESHOLDS["high"], line_dash="dash", line_color=PSI_COLORS_HEX["high"])
        st.plotly_chart(fig_rep, use_container_width=True)

# -------------------------
# Tab 3: Map + KML export
# -------------------------
with tabs[2]:
    st.subheader("Compaction Map")
    if lat_col and lon_col:
        depth_options = sorted(long["Depth_in"].unique().tolist())
        sel_depths = st.multiselect("Show depths (in)", options=depth_options, default=[depth_options[0]])
        radius = st.slider("Point radius (px)", 4, 20, 8)

        layers = []
        for d in sel_depths:
            layer_df = long[(long["Depth_in"] == d) & long[[lat_col, lon_col]].notna().all(axis=1)][[id_col, lat_col, lon_col, "PSI"]].rename(columns={lat_col: "lat", lon_col: "lon"})
            if layer_df.empty:
                continue
            layer_df["rgb"] = layer_df["PSI"].apply(psi_rgb)
            lyr = pdk.Layer(
                "ScatterplotLayer",
                data=layer_df,
                get_position='[lon, lat]',
                get_radius=radius,
                pickable=True,
                get_fill_color='rgb',
            )
            layers.append(lyr)
        if layers:
            view = pdk.ViewState(latitude=float(long[lat_col].mean()), longitude=float(long[lon_col].mean()), zoom=12)
            deck = pdk.Deck(layers=layers, initial_view_state=view)
            st.pydeck_chart(deck)
        else:
            st.info("Select at least one depth to display.")

        try:
            import simplekml
            kml = simplekml.Kml()
            for d in sel_depths:
                folder = kml.newfolder(name=f"Depth {d} in")
                rows = long[(long["Depth_in"] == d) & long[[lat_col, lon_col]].notna().all(axis=1)]
                for _, r in rows.iterrows():
                    p = folder.newpoint(name=f"{id_col}: {r[id_col]} | {r['PSI']:.0f} PSI", coords=[(float(r[lon_col]), float(r[lat_col]))])
                    p.style.iconstyle.color = psi_kml_color(float(r["PSI"]))
                    p.style.iconstyle.scale = 0.8
            kml_bytes = kml.kml().encode("utf-8")
            st.download_button("⬇️ Download selected depths (KML for Google Earth)", data=kml_bytes, file_name="compaction_points.kml", mime="application/vnd.google-earth.kml+xml")
        except Exception:
            st.caption("Install 'simplekml' to enable KML export.")
    else:
        st.info("Latitude/Longitude not provided — map disabled.")

st.sidebar.caption("Built with Streamlit • Plotly • PyDeck")
