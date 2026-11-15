from pathlib import Path
import pandas as pd
import dagster as dg
import requests
import os
from dagster import Failure
from huggingface_hub import HfApi


from datetime import datetime

municipality_partitions = dg.StaticPartitionsDefinition(["asker", "baerum", "lillestroem"])
# 2. Map the keys to your actual ArcGIS IDs
ARCGIS_IDS = {
    "asker": "20016755948d419a8a497115440bd2d1",
    "baerum": "5977d524cd884257bd26623ada5599e9",
    "lier": "482b9ef4172a4a4d86518367d9ef8c8d", # Not returning anything. Needs investigation
    "lillestroem": "01c62103540a41338498506ec433b15b",
    "nesodden": "26a87659174d421ab8b0809dd2b746ba", # Not active and listed as Venus muncipal
    "nordre_follo": "dc12372c0a514bd585387e398b023c8d", # Requires sign in

}

@dg.asset(partitions_def=municipality_partitions)
def road_lights(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Scrapes ArcGIS data and returns the path to the saved data"""
    where = context.partition_key
    app_id = ARCGIS_IDS[where]
    config_url = f"https://www.arcgis.com/sharing/rest/content/items/{app_id}/data?f=json"

    headers = {"User-Agent": "Mozilla/5.0"}

    context.log.info(f"Fetching configuration for Experience app: {app_id}...")
    response = requests.get(config_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch app configuration: {response.status_code}")

    app_config = response.json()
    data_sources = app_config.get("dataSources", {})
    feature_layer_urls = []

    # 2. Parse data sources to find REST endpoints
    for ds_id, ds_info in data_sources.items():
        if "url" in ds_info:
            feature_layer_urls.append(ds_info["url"])
        elif "itemId" in ds_info:
            item_id = ds_info["itemId"]
            item_url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}?f=json"
            item_resp = requests.get(item_url, headers=headers).json()
            item_type = item_resp.get("type", "")

            if "Web Map" in item_type:
                map_data_url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data?f=json"
                map_data = requests.get(map_data_url, headers=headers).json()
                for op_layer in map_data.get("operationalLayers", []):
                    if "url" in op_layer:
                        feature_layer_urls.append(op_layer["url"])
            elif "Feature Service" in item_type or "Map Service" in item_type:
                if "url" in item_resp:
                    feature_layer_urls.append(item_resp["url"])

    feature_layer_urls = list(set(feature_layer_urls))
    context.log.info(f"Discovered {len(feature_layer_urls)} target Feature Layer service(s).")

    # 3. Query Feature Layers with full pagination (No Capping)
    all_records = []
    BATCH_SIZE = 1000

    for layer_url in feature_layer_urls:
        query_url = layer_url if layer_url.endswith("/query") else f"{layer_url}/query"

        offset = 0
        while True:
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": BATCH_SIZE,
                "f": "geojson"
            }

            context.log.info(f"Querying layer at offset {offset}...")
            res = requests.get(query_url, params=params)
            if res.status_code != 200:
                context.log.warning(f"Warning: Could not query {query_url} (Status: {res.status_code})")
                break

            data = res.json()
            features = data.get("features", [])

            if not features:
                break

            for feature in features:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})

                record = dict(props)
                if geom and "coordinates" in geom:
                    coords = geom["coordinates"]
                    if isinstance(coords, list) and len(coords) >= 2:
                        record["longitude"] = coords[0]
                        record["latitude"] = coords[1]

                all_records.append(record)

            if len(features) < BATCH_SIZE:
                break

            offset += BATCH_SIZE

    context.log.info(f"\nTotal records retrieved across all layers: {len(all_records)}")

    # 4. Create Pandas DataFrame
    if all_records:
        df = pd.DataFrame(all_records)
        context.log.info(f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")

        today = datetime.now().strftime("%Y-%m-%d")
        dst = Path("/tmp") / where / f"rls_{today}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)


@dg.asset(
    deps=[road_lights],
    partitions_def=municipality_partitions,
)
def upload_hf_bucket(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
        """
        Reads /tmp/<where>/rls_<date>.csv and uploads it to the Hugging Face buckets.
        """
        where = context.partition_key
        today = datetime.now().strftime("%Y-%m-%d")
        path_in_repo = f"{where}/rls_{today}.parquet"     # The name the file will have in HF
        file_path = f"/tmp/{path_in_repo}"
        repo_id = "knuthp/roadlightstatus"  # Assuming this is a dataset repo

        # 1. Verify local file exists
        if not os.path.exists(file_path):
            context.log.error(f"File not found: {file_path}")
            raise Failure(f"Source file {file_path} does not exist.")

        # 2. Get Token from environment
        token = os.getenv("HF_TOKEN")
        if not token:
            context.log.error("HF_TOKEN environment variable is not set.")
            raise Failure("HF_TOKEN must be set to upload to Hugging Face.")


        try:
            context.log.info(f"Starting upload of {file_path} to HF repo {repo_id}...")
            api = HfApi(token=token)

            api.batch_bucket_files(
                bucket_id=repo_id,
                add=[
                    (file_path, path_in_repo)
                ],
            )
            context.log.info(f"Successfully uploaded {file_path} to https://huggingface.co/datasets/{repo_id}")
            return f"https://huggingface.co/datasets/{repo_id}/{path_in_repo}"

        except Exception as e:
            context.log.error(f"Failed to upload to Hugging Face: {str(e)}")
            raise Failure(f"Upload failed: {e}")
