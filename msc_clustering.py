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

START_DATE = datetime.datetime(2022, 7, 5)
END_DATE   = datetime.datetime(2025, 7, 6)

# Chicago terminal filter (kept)
CHICAGO = [102, 106, 274, 267, 130, 126, 278, 125, 127, 129, 105, 122, 268, 124, 123, 284, 285]

# Continental US only (no AK/HI/PR)
CONTINENTAL_US = [
    "AL","AZ","AR","CA","CO","CT","DE","DC","FL","GA","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH",
    "OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

EQUIPMENT_FILTER = ["Van", "Van or Reefer"]

# Origins to emphasize for this customer
EMPH_ORIGINS = {
    ("ELKHART", "IN"),
    ("MABLETON", "GA"),
    ("JONESTOWN", "PA"),
}

BASE_PROJECTION = {
    "_id": 0,
    "load_id": "$LoadID",
    "customer": "$Customer",
    "customer_id": "$CustomerID",
    "equipment": "$EquipmentType",

    "origin_city": "$OriginData.OriginCity",
    "origin_state": "$OriginData.OriginState",
    "origin_latitude": "$OriginData.OriginLatitude",
    "origin_longitude": "$OriginData.OriginLongitude",

    "destination_city": "$DestinationData.DestinationCity",
    "destination_state": "$DestinationData.DestinationState",
    "destination_latitude": "$DestinationData.DestinationLatitude",
    "destination_longitude": "$DestinationData.DestinationLongitude",
}

def pull_location_data(customer_id=None) -> pd.DataFrame:
    """
    Pull lanes for Chicago terminal only, continental US only, equipment filtered.
    - customer_id=None   -> all Chicago terminal loads (still filtered)
    - customer_id=4933   -> only that customer (still filtered)
    """

    query = {
        "PickupDate": {"$gte": START_DATE, "$lte": END_DATE},

        "CustomerTerminalCode": {"$in": CHICAGO},

        # Hard exclude non-US and non-continental destinations/origins
        "OriginData.OriginCountry": "USA",
        "DestinationData.DestinationCountry": "USA",
        "OriginData.OriginState": {"$in": CONTINENTAL_US},
        "DestinationData.DestinationState": {"$in": CONTINENTAL_US},

        "LoadStatus": {"$in": ["Delivered", "Dispatched", "Planned"]},

        "OriginData.OriginLatitude": {"$exists": True, "$ne": None},
        "OriginData.OriginLongitude": {"$exists": True, "$ne": None},
        "DestinationData.DestinationLatitude": {"$exists": True, "$ne": None},
        "DestinationData.DestinationLongitude": {"$exists": True, "$ne": None},

        "EquipmentType": {"$in": EQUIPMENT_FILTER},
    }

    if customer_id is not None:
        query["CustomerID"] = customer_id

    rows = list(loadDb["v4loadDetail"].find(query, BASE_PROJECTION))
    df = pd.DataFrame(rows)

    # Normalize city/state for consistent matching
    if not df.empty:
        df["origin_city_norm"] = df["origin_city"].astype(str).str.strip().str.upper()
        df["origin_state_norm"] = df["origin_state"].astype(str).str.strip().str.upper()

    print(f"Pulled {len(df):,} loads (customer_id={customer_id})")
    return df

def to_lane_geodata(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    lines = [
        LineString([(olon, olat), (dlon, dlat)])
        for olon, olat, dlon, dlat in zip(
            df["origin_longitude"], df["origin_latitude"],
            df["destination_longitude"], df["destination_latitude"]
        )
    ]
    return gpd.GeoDataFrame(df.copy(), geometry=lines, crs="EPSG:4326")

def plot_customer_with_emphasis(customer_gdf: gpd.GeoDataFrame):
    """
    Clean map:
      - Plot ONLY customer lanes (no gray company layer)
      - Emphasize select origin facilities with distinct markers
    """

    us_map = gpd.read_file("./files/shapefile/tl_2023_us_state.shp")
    us_map = us_map[us_map["STUSPS"].isin(CONTINENTAL_US)]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Cleaner base map (no fill clutter)
    us_map.boundary.plot(ax=ax, linewidth=0.6)

    if customer_gdf.empty:
        ax.set_title("No customer lanes found for the selected filters/time window.")
        ax.set_axis_off()
        plt.show()
        return

    # Plot customer lane lines (single color, higher alpha than before)
    customer_gdf.plot(
        ax=ax,
        linewidth=1.0,
        alpha=0.25,
        label="CustomerID 4933 lanes"
    )

    # Emphasize those origin facilities (points)
    emph_mask = customer_gdf.apply(
        lambda r: (r["origin_city_norm"], r["origin_state_norm"]) in EMPH_ORIGINS,
        axis=1
    )
    emph = customer_gdf[emph_mask].copy()

    if not emph.empty:
        emph_points = gpd.GeoDataFrame(
            emph,
            geometry=gpd.points_from_xy(emph["origin_longitude"], emph["origin_latitude"]),
            crs="EPSG:4326"
        )

        emph_points.plot(
            ax=ax,
            marker="o",
            markersize=35,
            alpha=0.9,
            label="Emphasized origin facilities"
        )

        # Add text labels near those points (basic, but helpful)
        for _, row in emph_points.iterrows():
            label = f"{row['origin_city']}, {row['origin_state']}"
            ax.annotate(
                label,
                (row["origin_longitude"], row["origin_latitude"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=9
            )

    ax.set_title(
        f"CustomerID 4933 — Chicago Terminal Lanes (Van / Van or Reefer)\n"
        f"{START_DATE.date()} to {END_DATE.date()} (Continental US only)"
    )
    ax.set_axis_off()
    ax.legend(loc="lower left")
    plt.show()

def __main__():
    # Only plotting customer per your new preference (no gray background layer)
    customer_df = pull_location_data(customer_id=4933)
    customer_gdf = to_lane_geodata(customer_df)
    plot_customer_with_emphasis(customer_gdf)

if __name__ == "__main__":
    __main__()
