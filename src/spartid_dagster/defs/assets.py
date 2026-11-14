from pathlib import Path
import dagster as dg


@dg.asset
def my_asset(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    Path("/tmp/a.txt").write_text("Hello From Dagster")
