import datetime
import os

from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from pymongo import MongoClient


load_dotenv()

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
if not CONNECTION_STRING:
    raise ValueError("CONNECTION_STRING not found in environment variables")

client = MongoClient(CONNECTION_STRING)
loadDb = client.LoadDetail

CHICAGO = [102, 106, 274, 267, 130, 126, 278, 125, 127, 129, 105, 122, 268, 124, 123, 284, 285]

CONTINENTAL_US = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

CUSTOMER_ID = 6451
EQUIPMENT_FILTER = [
    "Conestoga",
    "Flatbed",
    "Flatbed or Conestoga",
    "Flatbed,Hotshot",
    "Sprinter Van",
    "Straight Truck",
    "Van",
]

TOP_N_NETWORK = 150
TOP_N_CUSTOMER = 25

START_DATE = datetime.datetime(2025, 1, 1)
END_DATE = datetime.datetime(2026, 1, 2)

EMPH_ORIGINS = {
    ("BETTENDORF", "IA"),
    ("MIDDLEBURY", "IN"),
    ("GLENPOOL", "OK"),
}

# Optional forced stars if you ever want them back
FORCED_STARS = [
    # {"city": "RENO", "state": "NV", "lat": 39.5296, "lon": -119.8138},
]


def pull_data(customer_id=None) -> pd.DataFrame:
    query = {
        "PickupDate": {"$gte": START_DATE, "$lte": END_DATE},
        "CustomerTerminalCode": {"$in": CHICAGO},
        "OriginData.OriginCountry": "USA",
        "DestinationData.DestinationCountry": "USA",
        "OriginData.OriginState": {"$in": CONTINENTAL_US},
        "DestinationData.DestinationState": {"$in": CONTINENTAL_US},
        "LoadStatus": {"$in": ["Delivered", "Dispatched", "Planned"]},
        "EquipmentType": {"$in": EQUIPMENT_FILTER},
        "OriginData.OriginLatitude": {"$exists": True, "$ne": None},
        "OriginData.OriginLongitude": {"$exists": True, "$ne": None},
        "DestinationData.DestinationLatitude": {"$exists": True, "$ne": None},
        "DestinationData.DestinationLongitude": {"$exists": True, "$ne": None},
    }

    if customer_id is not None:
        query["CustomerID"] = customer_id

    projection = {
        "_id": 0,
        "OriginData.OriginCity": 1,
        "OriginData.OriginState": 1,
        "OriginData.OriginLatitude": 1,
        "OriginData.OriginLongitude": 1,
        "DestinationData.DestinationCity": 1,
        "DestinationData.DestinationState": 1,
        "DestinationData.DestinationLatitude": 1,
        "DestinationData.DestinationLongitude": 1,
    }

    rows = list(loadDb["v4loadDetail"].find(query, projection))

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    df = df.rename(columns={
        "OriginData.OriginCity": "origin_city",
        "OriginData.OriginState": "origin_state",
        "OriginData.OriginLatitude": "origin_lat",
        "OriginData.OriginLongitude": "origin_lon",
        "DestinationData.DestinationCity": "dest_city",
        "DestinationData.DestinationState": "dest_state",
        "DestinationData.DestinationLatitude": "dest_lat",
        "DestinationData.DestinationLongitude": "dest_lon",
    })

    expected_cols = [
        "origin_city", "origin_state", "origin_lat", "origin_lon",
        "dest_city", "dest_state", "dest_lat", "dest_lon"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns after Mongo pull: {missing_cols}")

    for c in ["origin_city", "origin_state", "dest_city", "dest_state"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    return df


def filter_continental_bbox(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    for c in ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    min_lat, max_lat = 24.0, 50.0
    min_lon, max_lon = -125.0, -66.0

    before = len(df)
    df = df.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"])

    df = df[
        (df["origin_lat"].between(min_lat, max_lat)) &
        (df["dest_lat"].between(min_lat, max_lat)) &
        (df["origin_lon"].between(min_lon, max_lon)) &
        (df["dest_lon"].between(min_lon, max_lon))
    ].copy()

    after = len(df)
    print(f"Filtered bbox continental coords: {before:,} -> {after:,}")
    return df


def filter_to_emph_origins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    return df[
        df.apply(lambda r: (r["origin_city"], r["origin_state"]) in EMPH_ORIGINS, axis=1)
    ].copy()


def top_lanes_by_volume(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df

    agg = (
        df.groupby(["origin_city", "origin_state", "dest_city", "dest_state"], as_index=False)
        .agg(
            load_count=("origin_city", "size"),
            origin_lat=("origin_lat", "median"),
            origin_lon=("origin_lon", "median"),
            dest_lat=("dest_lat", "median"),
            dest_lon=("dest_lon", "median"),
        )
        .sort_values("load_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return agg


def to_lines_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    df = df.copy()
    df["geometry"] = df.apply(
        lambda r: LineString([
            (r["origin_lon"], r["origin_lat"]),
            (r["dest_lon"], r["dest_lat"])
        ]),
        axis=1
    )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def build_star_points(customer_raw: pd.DataFrame) -> gpd.GeoDataFrame:
    star_rows = []

    if customer_raw is not None and not customer_raw.empty:
        emph_raw = customer_raw[
            customer_raw.apply(lambda r: (r["origin_city"], r["origin_state"]) in EMPH_ORIGINS, axis=1)
        ]

        if not emph_raw.empty:
            emph_raw = emph_raw.drop_duplicates(subset=["origin_city", "origin_state"]).copy()
            for _, r in emph_raw.iterrows():
                star_rows.append({
                    "city": r["origin_city"],
                    "state": r["origin_state"],
                    "lat": float(r["origin_lat"]),
                    "lon": float(r["origin_lon"]),
                })

    for s in FORCED_STARS:
        star_rows.append({
            "city": s["city"].strip().upper(),
            "state": s["state"].strip().upper(),
            "lat": float(s["lat"]),
            "lon": float(s["lon"]),
        })

    if not star_rows:
        return gpd.GeoDataFrame(columns=["city", "state", "lat", "lon", "geometry"], crs="EPSG:4326")

    stars_df = pd.DataFrame(star_rows).drop_duplicates(subset=["city", "state"])
    stars_gdf = gpd.GeoDataFrame(
        stars_df,
        geometry=gpd.points_from_xy(stars_df["lon"], stars_df["lat"]),
        crs="EPSG:4326"
    )
    return stars_gdf


def plot_map(network_gdf: gpd.GeoDataFrame,
             customer_gdf: gpd.GeoDataFrame,
             customer_raw: pd.DataFrame):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    shape_path = os.path.join(base_dir, "files", "shapefile", "tl_2023_us_state.shp")

    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"Shapefile not found: {shape_path}")

    us = gpd.read_file(shape_path)
    us = us[us["STUSPS"].isin(CONTINENTAL_US)]

    fig, ax = plt.subplots(figsize=(14, 10))
    us.boundary.plot(ax=ax, linewidth=0.6, zorder=1)

    if not network_gdf.empty:
        network_gdf.plot(
            ax=ax,
            linewidth=2.0,
            alpha=0.22,
            color="#4FA3D1",
            label="Circle Network",
            zorder=2
        )

    if not customer_gdf.empty:
        customer_gdf.plot(
            ax=ax,
            linewidth=3.2,
            alpha=0.75,
            color="#1E6F3D",
            label="Circle Arconic Lanes",
            zorder=3
        )

    stars = build_star_points(customer_raw)
    if not stars.empty:
        stars.plot(
            ax=ax,
            marker="*",
            markersize=520,
            color="white",
            edgecolor="white",
            linewidth=2.5,
            alpha=0.95,
            zorder=9
        )
        stars.plot(
            ax=ax,
            marker="*",
            markersize=380,
            color="gold",
            edgecolor="black",
            linewidth=2.2,
            alpha=0.98,
            label="Key Origins",
            zorder=10
        )

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_axis_off()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper right",
              bbox_to_anchor = (1.02, 1),
              borderaxespad = 0
            )
    plt.show()
    plt.savefig("lane_map.png", dpi=300, bbox_inches="tight")


def main():
    print("Pulling Chicago terminal network loads...")
    network_raw = pull_data(customer_id=None)
    print(f"Network loads (raw): {len(network_raw):,}")
    network_raw = filter_continental_bbox(network_raw)

    print(f"Pulling CustomerID {CUSTOMER_ID} loads...")
    customer_raw = pull_data(customer_id=CUSTOMER_ID)
    print(f"Customer loads (raw): {len(customer_raw):,}")
    customer_raw = filter_continental_bbox(customer_raw)

    customer_emph = filter_to_emph_origins(customer_raw)
    print(f"Customer loads from EMPH_ORIGINS: {len(customer_emph):,}")

    print("Aggregating top lanes by volume...")
    network_top = top_lanes_by_volume(network_raw, TOP_N_NETWORK)
    customer_top = top_lanes_by_volume(customer_emph, TOP_N_CUSTOMER)

    print(f"Top network lanes plotted: {len(network_top):,}")
    print(f"Top customer emphasized-origin lanes plotted: {len(customer_top):,}")

    network_gdf = to_lines_gdf(network_top)
    customer_gdf = to_lines_gdf(customer_top)

    plot_map(network_gdf, customer_gdf, customer_raw)


if __name__ == "__main__":
    main()