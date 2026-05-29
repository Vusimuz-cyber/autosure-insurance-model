import pandas as pd
from rapidfuzz import process, fuzz

def load_reference_data(
    cars_path="../data/cars.csv",
    areas_path="../data/areas.csv"
):
    cars_df = pd.read_csv(cars_path)
    areas_df = pd.read_csv(areas_path)

    # Clean data
    cars_df.fillna("", inplace=True)
    areas_df.fillna("", inplace=True)

    return cars_df, areas_df


def best_car_match(car_input, cars_df, score_cutoff=55):
    if not car_input or cars_df.empty:
        return None

    cars_df = cars_df.copy()

    cars_df["combined"] = (
        cars_df["Brand"].astype(str).str.lower()
        + " "
        + cars_df["Model"].astype(str).str.lower()
    )

    result = process.extractOne(
        car_input.lower(),
        cars_df["combined"],
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff
    )

    if result is None:
        return None

    match, score, index = result

    print(f"🚗 Car matched: {match} ({score}%)")

    return cars_df.iloc[index]


def best_area_match(area_input, areas_df, score_cutoff=55):
    if not area_input or areas_df.empty:
        return None

    areas_df = areas_df.copy()

    areas_df["combined"] = (
        areas_df["Place"].astype(str).str.lower()
        + " "
        + areas_df["Province"].astype(str).str.lower()
    )

    result = process.extractOne(
        area_input.lower(),
        areas_df["combined"],
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff
    )

    if result is None:
        return None

    match, score, index = result

    print(f"📍 Area matched: {match} ({score}%)")

    return areas_df.iloc[index]