import pandas as pd
from rapidfuzz import process, fuzz

def load_reference_data(cars_path="../data/cars.csv", areas_path="../data/areas.csv"):
    cars_df = pd.read_csv(cars_path)
    areas_df = pd.read_csv(areas_path)
    return cars_df, areas_df

def best_car_match(car_input, cars_df, score_cutoff=55):
    if not car_input:
        return None
    # Combine Brand and Model for matching
    cars_df["combined"] = cars_df["Brand"].str.lower() + " " + cars_df["Model"].str.lower()
    result = process.extractOne(car_input.lower(), cars_df["combined"], scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if result is None:
        return None
    match, score, index = result
    return cars_df.iloc[index]

def best_area_match(area_input, areas_df, score_cutoff=55):
    if not area_input:
        return None
    result = process.extractOne(area_input.lower(), areas_df["Place"].str.lower(), scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if result is None:
        return None
    match, score, index = result
    return areas_df.iloc[index]