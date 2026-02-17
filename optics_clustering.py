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
client = MongoClient(host=CONNECTION_STRING)
loadDb = client.LoadDetail

CHICAGO = [102,106,274,267,130,126,278,125,127,129,105,122,268,124,123,284,285]

CONTINENTAL_US = [
    "AL","AZ","AR","CA","CO","CT","DE","DC","FL","GA","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH",
    "OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

CUSTOMER_ID = 4933
EQUIPMENT_FILTER = ["Van", "Van or Reefer"]

TOP_N_NETWORK = 150
TOP_N_CUSTOMER = 25

START_DATE = datetime.datetime(2025, 1, 1)
END_DATE   = datetime.datetime(2026, 1, 2)

# These are "only if they exist in customer_raw"
EMPH_ORIGINS = {
    ("ELKHART", "IN"),
    ("MABLETON", "GA"),
    ("JONESTOWN", "PA"),
}

# This star should ALWAYS appear even if there are no loads
FORCED_STARS = [
    {"city": "RENO", "state": "NV", "lat": 39.5296, "lon": -119.8138},
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
        "origin_city": "$OriginData.OriginCity",
        "origin_state": "$OriginData.OriginState",
        "origin_lat": "$OriginData.OriginLatitude",
        "origin_lon": "$OriginData.OriginLongitude",
        "dest_city": "$DestinationData.DestinationCity",
        "dest_state": "$DestinationData.DestinationState",
        "dest_lat": "$DestinationData.DestinationLatitude",
        "dest_lon": "$DestinationData.DestinationLongitude",
    }

    rows = list(loadDb["v4loadDetail"].find(query, projection))
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    for c in ["origin_city", "origin_state", "dest_city", "dest_state"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    return df


def filter_continental_bbox(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for c in ["origin_lat","origin_lon","dest_lat","dest_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    MIN_LAT, MAX_LAT = 24.0, 50.0
    MIN_LON, MAX_LON = -125.0, -66.0

    before = len(df)
    df = df.dropna(subset=["origin_lat","origin_lon","dest_lat","dest_lon"])

    df = df[
        (df["origin_lat"].between(MIN_LAT, MAX_LAT)) &
        (df["dest_lat"].between(MIN_LAT, MAX_LAT)) &
        (df["origin_lon"].between(MIN_LON, MAX_LON)) &
        (df["dest_lon"].between(MIN_LON, MAX_LON))
    ].copy()

    after = len(df)
    print(f"Filtered bbox continental coords: {before:,} -> {after:,}")
    return df


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
        lambda r: LineString([(r["origin_lon"], r["origin_lat"]), (r["dest_lon"], r["dest_lat"])]),
        axis=1
    )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def build_star_points(customer_raw: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame of star points:
      - from customer_raw (for EMPH_ORIGINS, if present)
      - plus forced stars (always shown)
    """
    star_rows = []

    # Stars from customer data (only if present)
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

    # Forced stars (always)
    for s in FORCED_STARS:
        star_rows.append({
            "city": s["city"].strip().upper(),
            "state": s["state"].strip().upper(),
            "lat": float(s["lat"]),
            "lon": float(s["lon"]),
        })

    if not star_rows:
        return gpd.GeoDataFrame(columns=["city","state","lat","lon","geometry"], crs="EPSG:4326")

    stars_df = pd.DataFrame(star_rows).drop_duplicates(subset=["city","state"])
    stars_gdf = gpd.GeoDataFrame(
        stars_df,
        geometry=gpd.points_from_xy(stars_df["lon"], stars_df["lat"]),
        crs="EPSG:4326"
    )
    return stars_gdf


def plot_map(network_gdf: gpd.GeoDataFrame,
             customer_gdf: gpd.GeoDataFrame,
             customer_raw: pd.DataFrame):

    us = gpd.read_file("./files/shapefile/tl_2023_us_state.shp")
    us = us[us["STUSPS"].isin(CONTINENTAL_US)]

    fig, ax = plt.subplots(figsize=(14, 10))
    us.boundary.plot(ax=ax, linewidth=0.6, zorder=1)

    if not network_gdf.empty:
        network_gdf.plot(
            ax=ax, linewidth=2.0, alpha=0.22, color="#4FA3D1",
            label="Circle Network", zorder=2
        )

    if not customer_gdf.empty:
        customer_gdf.plot(
            ax=ax, linewidth=3.2, alpha=0.75, color="#1E6F3D",
            label="Circle MSC Lanes", zorder=3
        )

    # Stars (customer + forced)
    stars = build_star_points(customer_raw)
    if not stars.empty:
        # halo then star
        stars.plot(ax=ax, marker="*", markersize=520, color="white",
                   edgecolor="white", linewidth=2.5, alpha=0.95, zorder=9)
        stars.plot(ax=ax, marker="*", markersize=380, color="gold",
                   edgecolor="black", linewidth=2.2, alpha=0.98,
                   label="Key Origins", zorder=10)

        # OPTIONAL labels:
        # for _, r in stars.iterrows():
        #     ax.text(r["lon"], r["lat"], f"  {r['city'].title()}, {r['state']}",
        #             fontsize=11, weight="bold", zorder=11)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    # Remove border box
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper right")
    plt.show()


def main():
    print("Pulling Chicago terminal network loads...")
    network_raw = pull_data(customer_id=None)
    print(f"Network loads (raw): {len(network_raw):,}")
    network_raw = filter_continental_bbox(network_raw)

    print(f"Pulling CustomerID {CUSTOMER_ID} loads...")
    customer_raw = pull_data(customer_id=CUSTOMER_ID)
    print(f"Customer loads (raw): {len(customer_raw):,}")
    customer_raw = filter_continental_bbox(customer_raw)

    print("Aggregating top lanes by volume...")
    network_top = top_lanes_by_volume(network_raw, TOP_N_NETWORK)
    customer_top = top_lanes_by_volume(customer_raw, TOP_N_CUSTOMER)

    network_gdf = to_lines_gdf(network_top)
    customer_gdf = to_lines_gdf(customer_top)

    plot_map(network_gdf, customer_gdf, customer_raw)


if __name__ == "__main__":
    main()
