from huggingface_hub import HfApi
import os
from pathlib import Path
import logging
import json
import requests
import pandas as pd
import dagster as dg

logger = logging.getLogger(__name__)
# 1. Configuration & Parameters
BASE_URL = "https://services.api.no/api/acies/v1/custom/PropcloudLegacyTransaction"

monthly_partitions = dg.MonthlyPartitionsDefinition(
    start_date="2025-01-01",
    timezone="Europe/Oslo"
)

newspaper_partitions = dg.StaticPartitionsDefinition(["budsti", "smaalenene", "laagen", "dramti", "avisao", ])

multi_partitions_def = dg.MultiPartitionsDefinition({
    "month": monthly_partitions,
    "newspaper": newspaper_partitions,
})

FIELDS = (
    "(propertyId,municipality,imageUrl,sellerList,buyerList,cadestralId,"
    "groupCount,housingCoopName,saleType,saleId,price,date,includesBuilding,"
    "propertyArea,livingArea,postalCode,postalArea,streetName,houseNumber,"
    "houseLetter,county,countyCode,region,municipalityCode,boroughCode,"
    "citySector,cityArea,geometry,propertyType,buildingType,attachedTransactions,withdrawn)"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

def build_params(start_date, end_date, newspaper) -> dict[str, str]:
    return {
        "greaterThanOrEqual": f"date:{start_date}",
        "lessThan": f"date:{end_date}",
        "limit": "9999",
        "sortBy": "date",
        "sortDirection": "desc",
        "fields": FIELDS,
        "notEqual": "saleType:Uskiftebevilling",
        "inPolygon": f"geometry:local_realestate_area:{newspaper}",
        "equal": "propertyType:Bolig",
    }


def fetch_property_data(start_date: str, end_date: str, newspaper: str):
    print(f"Fetching real estate transactions from {start_date} to {end_date}...")

    try:
        params = build_params(start_date, end_date, newspaper)
        response = requests.get(BASE_URL, headers=HEADERS, params=params)

        # Print final requested URL for debugging
        print(f"Request URL: {response.url}")

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None


def process_and_save_data(data, postfix: str, newspaper: str):
    # Extract records (adjust key if the JSON wrapper uses a specific root key like 'data' or 'results')
    # If the API returns a direct list or a dictionary with items:
    records = (
        data.get("results", data) if isinstance(data, dict) else data
    )

    if not records:
        print("No records found in the API response.")
        return

    print(f"Successfully retrieved {len(records)} records.")

    # Flatten nested structures (e.g., buyerList, sellerList, geometry) for CSV export
    flattened_records = []
    for item in records:
        row = item.copy()

        # Convert lists to JSON strings so pandas can handle them in a flat CSV
        if "sellerList" in row and isinstance(row["sellerList"], list):
            row["sellers"] = json.dumps(row["sellerList"], ensure_ascii=False)
            del row["sellerList"]

        if "buyerList" in row and isinstance(row["buyerList"], list):
            row["buyers"] = json.dumps(row["buyerList"], ensure_ascii=False)
            del row["buyerList"]

        if "geometry" in row and isinstance(row["geometry"], dict):
            row["geometry_geom"] = row["geometry"].get("geom", "")
            del row["geometry"]

        flattened_records.append(row)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(flattened_records)

    # Save to file
    parquet_filename = Path("/tmp") / newspaper / f"re_{postfix}.parquet"
    parquet_filename.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(parquet_filename, index=False)
    print(f"Saved flattened dataset to '{parquet_filename}' ({len(df)} rows)")


@dg.asset(partitions_def=multi_partitions_def, group_name="RealEstate")
def fetch_amedia_property(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    partition_keys = context.partition_key.keys_by_dimension
    newspaper = partition_keys["newspaper"]
    time_window = context.partition_time_window
    date_pattern = "%Y-%m-%d"
    start_date = time_window.start.strftime(date_pattern)
    end_date = time_window.end.strftime(date_pattern)
    context.log.info(f"Starting to fetch {start_date} to {end_date}")
    raw_data = fetch_property_data(start_date, end_date, newspaper)
    if not raw_data:
        raise Exception(f"No data returned for {newspaper} from {start_date} to {end_date}")
    process_and_save_data(raw_data, start_date[:-3], newspaper)


@dg.asset(
    deps=[fetch_amedia_property],
    partitions_def=multi_partitions_def,
    group_name="RealEstate",
)
def upload_re_hf_bucket(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    partition_keys = context.partition_key.keys_by_dimension
    newspaper = partition_keys["newspaper"]
    time_window = context.partition_time_window
    date_pattern = "%Y-%m"
    postfix = time_window.start.strftime(date_pattern)

    path_in_repo = f"{newspaper}/re_{postfix}.parquet"     # The name the file will have in HF
    file_path = f"/tmp/{path_in_repo}"
    repo_id = "knuthp/realestate"

    # 1. Verify local file exists
    if not os.path.exists(file_path):
        context.log.error(f"File not found: {file_path}")
        raise dg.Failure(f"Source file {file_path} does not exist.")

    # 2. Get Token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        context.log.error("HF_TOKEN environment variable is not set.")
        raise dg.Failure("HF_TOKEN must be set to upload to Hugging Face.")


    try:
        context.log.info(f"Starting upload of {file_path} to HF repo {repo_id}...")
        api = HfApi(token=token)

        api.batch_bucket_files(
            bucket_id=repo_id,
            add=[
                (file_path, path_in_repo)
            ],
        )
        context.log.info(f"Successfully uploaded {file_path} to https://huggingface.co/buckets/{repo_id}")
        return f"https://huggingface.co/buckets/{repo_id}/{path_in_repo}"

    except Exception as e:
        context.log.error(f"Failed to upload to Hugging Face: {str(e)}")
        raise dg.Failure(f"Upload failed: {e}")


if __name__ == "__main__":
    start_date = "2026-10-01"
    end_date="2026-11-01"
    newspaper = "smaale"
    raw_data = fetch_property_data(start_date, end_date, newspaper)
    if raw_data:
        process_and_save_data(raw_data, start_date[:-3], newspaper)
    else:
        print("Error no data")
