from spartid_dagster.defs.realestate import fetch_property_data


def test_fetch_property_data():
    start_date = "2025-01-01"
    end_date = "2025-02-01"
    newspaper = "budsti"

    data = fetch_property_data(start_date, end_date, newspaper)

    assert data is not None
    assert isinstance(data, list)
    assert 40 < len(data) < 5_000