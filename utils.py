import math
import requests
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from geopy.distance import geodesic
import mgrs as mgrs_lib
import folium
from astral import LocationInfo
from astral.sun import sun


# ======================================================
# ABI- JA PARSIFUNKTSIOONID
# ======================================================

def parse_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")


def parse_date(d_str: str) -> date:
    return datetime.strptime(d_str, "%Y-%m-%d").date()


def parse_clock(t_str: str) -> time:
    return datetime.strptime(t_str, "%H:%M").time()


def minutes_between(dt1: datetime, dt2: datetime) -> float:
    return (dt2 - dt1).total_seconds() / 60.0


def mgrs_to_latlon(mgrs_code: str) -> Tuple[float, float]:
    converter = mgrs_lib.MGRS()
    lat, lon = converter.toLatLon(mgrs_code)
    return float(lat), float(lon)


def straight_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return geodesic((lat1, lon1), (lat2, lon2)).meters


# ======================================================
# PÄIKESETÕUS / PÄIKESELOOJANG
# ======================================================

def enrich_control_points_with_coordinates(control_points: pd.DataFrame) -> pd.DataFrame:
    cp = control_points.copy().sort_values("jarjekord").reset_index(drop=True)
    lats = []
    lons = []
    for code in cp["mgrs"]:
        lat, lon = mgrs_to_latlon(code)
        lats.append(lat)
        lons.append(lon)
    cp["lat"] = lats
    cp["lon"] = lons
    return cp


def get_reference_location(control_points: pd.DataFrame) -> Tuple[float, float]:
    lat = float(control_points["lat"].mean())
    lon = float(control_points["lon"].mean())
    return lat, lon


def compute_sun_times(control_points: pd.DataFrame, race_config: dict) -> dict:
    cp = enrich_control_points_with_coordinates(control_points)
    ref_lat, ref_lon = get_reference_location(cp)
    race_date = parse_date(race_config["voistluse_kuupaev"])
    timezone_name = race_config.get("timezone", "Europe/Tallinn")

    location = LocationInfo(name="RaceArea", region="", timezone=timezone_name, latitude=ref_lat, longitude=ref_lon)
    s = sun(location.observer, date=race_date, tzinfo=timezone_name)

    race_config_out = dict(race_config)
    race_config_out["paeva_algus"] = s["sunrise"].strftime("%H:%M")
    race_config_out["pimeduse_algus"] = s["sunset"].strftime("%H:%M")
    race_config_out["sunrise_full"] = s["sunrise"]
    race_config_out["sunset_full"] = s["sunset"]
    race_config_out["reference_lat"] = ref_lat
    race_config_out["reference_lon"] = ref_lon
    return race_config_out


# ======================================================
# DISTANTSIDE LEIDMINE
# ======================================================

def road_distance_m_osrm(lat1: float, lon1: float, lat2: float, lon2: float, timeout: int = 15) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full"
    )
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        routes = data.get("routes", [])
        if routes:
            route = routes[0]
            distance = float(route["distance"])
            geometry = route.get("geometry", {})
            if "coordinates" in geometry:
                coords = [(lat, lon) for lon, lat in geometry["coordinates"]]
                return distance, coords
            else:
                # Fallback to straight line if no geometry
                return distance, [(lat1, lon1), (lat2, lon2)]
    except Exception:
        return None
    return None


def calculate_segment_distances(control_points: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    cp_lookup = control_points.set_index("kp_id").to_dict("index")
    seg = segments.copy().sort_values("segment_id").reset_index(drop=True)

    straight_distances = []
    road_distances = []
    used_distances = []
    notes = []
    route_coords = []

    for _, row in seg.iterrows():
        start_cp = cp_lookup[row["algus_kp_id"]]
        end_cp = cp_lookup[row["lopp_kp_id"]]

        lat1, lon1 = start_cp["lat"], start_cp["lon"]
        lat2, lon2 = end_cp["lat"], end_cp["lon"]

        straight_m = straight_distance_m(lat1, lon1, lat2, lon2)
        road_result = None
        used_m = None
        note = ""
        coords = None

        if row["liikumisviis"] == "tee":
            road_result = road_distance_m_osrm(lat1, lon1, lat2, lon2)
            if road_result is None:
                road_m = straight_m * 1.2
                note = "OSRM ebaõnnestus, kasutati varuplaani: linnulend * 1.2"
                coords = [(lat1, lon1), (lat2, lon2)]
            else:
                road_m, coords = road_result
                note = "Tee-distants OSRM-ist"
            used_m = road_m
        else:
            used_m = straight_m * 1.5
            note = "Varjatud liikumine: linnulend * 1.5"
            coords = [(lat1, lon1), (lat2, lon2)]

        straight_distances.append(straight_m)
        road_distances.append(road_m if road_result else None)
        used_distances.append(used_m)
        notes.append(note)
        route_coords.append(coords)

    seg["sirge_kaugus_m"] = straight_distances
    seg["tee_kaugus_m"] = road_distances
    seg["kasutatav_kaugus_m"] = used_distances
    seg["distance_note"] = notes
    seg["route_coords"] = route_coords
    return seg


# ======================================================
# KIIRUSTE RAKENDAMINE
# ======================================================

def apply_speeds(segments: pd.DataFrame, default_speeds: dict, overrides: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    seg = segments.copy()
    valge_list = []
    pime_list = []

    for _, row in seg.iterrows():
        seg_id = int(row["segment_id"])
        mode = row["liikumisviis"]

        default_valge = float(default_speeds[mode]["valge"])
        default_pime = float(default_speeds[mode]["pime"])
        final_valge = float(overrides.get(seg_id, {}).get("valge", default_valge))
        final_pime = float(overrides.get(seg_id, {}).get("pime", default_pime))

        if final_valge <= 0 or final_pime <= 0:
            raise ValueError(f"Lõigu {seg_id} kiirused peavad olema positiivsed.")

        valge_list.append(final_valge)
        pime_list.append(final_pime)

    seg["kiirus_valges_kmh"] = valge_list
    seg["kiirus_pimedas_kmh"] = pime_list
    return seg


# ======================================================
# STARDIAJAD
# ======================================================

def generate_team_start_times(race_config: dict) -> pd.DataFrame:
    first_start = parse_datetime(race_config["esimese_voistkonna_start"])
    interval = int(race_config["stardi_intervall_min"])
    earlier = int(race_config["nulltiimi_earlier_min"])
    team_count = int(race_config["voistkondade_arv"])

    rows = [{"team_id": 0, "start_time": first_start - timedelta(minutes=earlier)}]
    for i in range(1, team_count + 1):
        rows.append({"team_id": i, "start_time": first_start + timedelta(minutes=(i - 1) * interval)})
    return pd.DataFrame(rows)


# ======================================================
# VALGE / PIME LOOGIKA
# ======================================================

def is_light(dt: datetime, race_config: dict) -> bool:
    paeva_algus = parse_clock(race_config["paeva_algus"])
    pimeduse_algus = parse_clock(race_config["pimeduse_algus"])
    t = dt.time()
    return paeva_algus <= t < pimeduse_algus


def get_next_light_boundary(dt: datetime, race_config: dict) -> datetime:
    paeva_algus = parse_clock(race_config["paeva_algus"])
    pimeduse_algus = parse_clock(race_config["pimeduse_algus"])

    today = dt.date()
    today_day_start = datetime.combine(today, paeva_algus)
    today_dark_start = datetime.combine(today, pimeduse_algus)
    tomorrow_day_start = datetime.combine(today + timedelta(days=1), paeva_algus)

    if is_light(dt, race_config):
        return today_dark_start
    else:
        if dt < today_day_start:
            return today_day_start
        return tomorrow_day_start


def classify_interval(start_dt: datetime, end_dt: datetime, race_config: dict) -> str:
    start_light = is_light(start_dt, race_config)
    end_minus = end_dt - timedelta(seconds=1)
    end_light = is_light(end_minus, race_config)
    boundary = get_next_light_boundary(start_dt, race_config)

    if end_dt <= boundary and start_light == end_light:
        return "valge" if start_light else "pime"
    return "segalõik"


def calculate_segment_end_time(segment_distance_m: float, start_datetime: datetime, speed_white_kmh: float, speed_dark_kmh: float, race_config: dict) -> datetime:
    remaining_distance = float(segment_distance_m)
    current_time = start_datetime

    while remaining_distance > 0.01:
        current_is_light = is_light(current_time, race_config)
        next_boundary = get_next_light_boundary(current_time, race_config)

        speed_kmh = speed_white_kmh if current_is_light else speed_dark_kmh
        speed_m_per_min = speed_kmh * 1000.0 / 60.0
        available_minutes = max(minutes_between(current_time, next_boundary), 0.0)
        possible_distance = speed_m_per_min * available_minutes

        if possible_distance >= remaining_distance or available_minutes == 0:
            needed_minutes = remaining_distance / speed_m_per_min
            current_time = current_time + timedelta(minutes=needed_minutes)
            remaining_distance = 0.0
        else:
            remaining_distance -= possible_distance
            current_time = next_boundary

    return current_time


# ======================================================
# SIMULATSIOON
# ======================================================

def simulate_team_route(team_id: int, team_start: datetime, control_points: pd.DataFrame, segments: pd.DataFrame, race_config: dict, start_duration_min: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cp_lookup = control_points.set_index("kp_id").to_dict("index")
    seg_sorted = segments.sort_values("segment_id")

    segment_rows = []
    checkpoint_rows = []
    current_time = team_start + timedelta(minutes=start_duration_min)

    for _, seg in seg_sorted.iterrows():
        seg_id = int(seg["segment_id"])
        start_kp_id = int(seg["algus_kp_id"])
        end_kp_id = int(seg["lopp_kp_id"])
        dist_m = float(seg["kasutatav_kaugus_m"])
        speed_w = float(seg["kiirus_valges_kmh"])
        speed_d = float(seg["kiirus_pimedas_kmh"])

        seg_start = current_time
        seg_end = calculate_segment_end_time(dist_m, seg_start, speed_w, speed_d, race_config)
        classification = classify_interval(seg_start, seg_end, race_config)

        segment_rows.append({
            "team_id": team_id,
            "segment_id": seg_id,
            "algus_kp_id": start_kp_id,
            "lopp_kp_id": end_kp_id,
            "start_time": seg_start,
            "end_time": seg_end,
            "distance_m": dist_m,
            "light_classification": classification,
            "minutes_total": minutes_between(seg_start, seg_end),
        })

        # Lisa kontrollpunkt (va start)
        if end_kp_id > 0:
            kp_duration = float(cp_lookup[end_kp_id]["kestvus_min"])
            kp_arrival = seg_end
            kp_departure = kp_arrival + timedelta(minutes=kp_duration)
            checkpoint_rows.append({
                "team_id": team_id,
                "kp_id": end_kp_id,
                "kp_nimi": cp_lookup[end_kp_id]["nimi"],
                "arrival_time": kp_arrival,
                "departure_time": kp_departure
            })
            current_time = kp_departure
        else:
            current_time = seg_end

    return pd.DataFrame(segment_rows), pd.DataFrame(checkpoint_rows)


def simulate_all_teams(control_points: pd.DataFrame, segments: pd.DataFrame, race_config: dict, start_duration_min: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_times_df = generate_team_start_times(race_config)
    all_segment_results = []
    all_checkpoint_results = []

    for _, team_row in start_times_df.iterrows():
        team_id = int(team_row["team_id"])
        team_start = pd.to_datetime(team_row["start_time"])
        seg_res, cp_res = simulate_team_route(team_id, team_start, control_points, segments, race_config, start_duration_min)
        all_segment_results.append(seg_res)
        all_checkpoint_results.append(cp_res)

    segment_results_df = pd.concat(all_segment_results, ignore_index=True)
    checkpoint_results_df = pd.concat(all_checkpoint_results, ignore_index=True)
    return start_times_df, segment_results_df, checkpoint_results_df


def run_full_simulation(control_points_input: pd.DataFrame, segments_input: pd.DataFrame, race_config: dict, default_speeds: dict, overrides: Dict[int, Dict[str, float]], start_mgrs: str, start_duration_min: int):
    validate_inputs(control_points_input, segments_input, race_config)

    # Lisa start kontrollpunktide hulka
    start_lat, start_lon = mgrs_to_latlon(start_mgrs)
    start_row = pd.DataFrame([{
        "kp_id": 0,
        "nimi": "Start",
        "mgrs": start_mgrs,
        "kestvus_min": start_duration_min,
        "jarjekord": 0
    }])
    cp_input = pd.concat([start_row, control_points_input], ignore_index=True)

    if race_config.get("kasuta_automaatset_paikest", False):
        race_config = compute_sun_times(cp_input, race_config)

    cp = enrich_control_points_with_coordinates(cp_input)
    seg = calculate_segment_distances(cp, segments_input)
    seg = apply_speeds(seg, default_speeds, overrides)

    start_times_df, segment_results_df, checkpoint_results_df = simulate_all_teams(cp, seg, race_config, start_duration_min)

    # Lisa start koordinaadid
    results = {
        "race_config": race_config,
        "control_points": cp,
        "segments": seg,
        "start_times": start_times_df,
        "segment_results": segment_results_df,
        "checkpoint_results": checkpoint_results_df,
        "start_lat": start_lat,
        "start_lon": start_lon,
    }
    return results


# ======================================================
# SISENDI VALIDEERIMINE
# ======================================================

def validate_inputs(control_points: pd.DataFrame, segments: pd.DataFrame, race_config: dict):
    required_cp = {"kp_id", "nimi", "mgrs", "kestvus_min", "jarjekord"}
    required_seg = {"segment_id", "algus_kp_id", "lopp_kp_id", "liikumisviis"}

    if not required_cp.issubset(control_points.columns):
        missing = required_cp - set(control_points.columns)
        raise ValueError(f"Kontrollpunktide tabelist puuduvad veerud: {missing}")

    if not required_seg.issubset(segments.columns):
        missing = required_seg - set(segments.columns)
        raise ValueError(f"Lõikude tabelist puuduvad veerud: {missing}")

    if control_points["kp_id"].duplicated().any():
        raise ValueError("kp_id peab olema unikaalne.")

    if control_points["jarjekord"].duplicated().any():
        raise ValueError("jarjekord peab olema unikaalne.")

    if (control_points["kestvus_min"] < 0).any():
        raise ValueError("Kontrollpunkti kestvus ei tohi olla negatiivne.")

    allowed_modes = {"tee", "varjatud"}
    if not segments["liikumisviis"].isin(allowed_modes).all():
        wrong = segments.loc[~segments["liikumisviis"].isin(allowed_modes), "liikumisviis"].tolist()
        raise ValueError(f"Lubatud liikumisviisid on ainult {allowed_modes}. Vigased väärtused: {wrong}")

    existing_kp_ids = set(control_points["kp_id"]) | {0}
    if not segments["algus_kp_id"].isin(existing_kp_ids).all():
        raise ValueError("Kõik algus_kp_id väärtused peavad viitama olemasolevale kontrollpunktile.")
    if not segments["lopp_kp_id"].isin(existing_kp_ids).all():
        raise ValueError("Kõik lopp_kp_id väärtused peavad viitama olemasolevale kontrollpunktile.")

    if race_config["voistkondade_arv"] < 1:
        raise ValueError("Võistkondade arv peab olema vähemalt 1.")
    if race_config["stardi_intervall_min"] <= 0:
        raise ValueError("Stardiintervall peab olema suurem kui 0.")
    if race_config["nulltiimi_earlier_min"] < 0:
        raise ValueError("0-tiimi eelaeg ei tohi olla negatiivne.")


# ======================================================
# KOORMUSE ANALÜÜS
# ======================================================

def compute_checkpoint_load(checkpoint_results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for kp_id, group in checkpoint_results.groupby("kp_id"):
        events = []
        kp_name = group["kp_nimi"].iloc[0]
        for _, row in group.iterrows():
            events.append((pd.to_datetime(row["arrival_time"]), +1))
            events.append((pd.to_datetime(row["departure_time"]), -1))
        events.sort(key=lambda x: (x[0], -x[1]))

        current_load = 0
        max_load = 0
        max_time = None
        for ts, delta in events:
            current_load += delta
            if current_load > max_load:
                max_load = current_load
                max_time = ts

        records.append({
            "kp_id": kp_id,
            "kp_nimi": kp_name,
            "maksimaalne_koormus": max_load,
            "maks_koormuse_hetk": max_time,
            "kokku_kulastusi": group.shape[0],
            "varaseim_saabumine": pd.to_datetime(group["arrival_time"]).min(),
            "hiliseim_lahkumine": pd.to_datetime(group["departure_time"]).max(),
        })

    return pd.DataFrame(records).sort_values("kp_id").reset_index(drop=True)


# ======================================================
# VÄLJUNDITABELID
# ======================================================

def format_output_tables(results: dict):
    cp = results["control_points"].copy()
    seg = results["segments"].copy()
    starts = results["start_times"].copy()
    seg_res = results["segment_results"].copy()
    cp_res = results["checkpoint_results"].copy()
    kp_load = compute_checkpoint_load(results["checkpoint_results"])

    for df in [starts, seg_res, cp_res, kp_load]:
        for col in df.columns:
            if "time" in col or "saabumine" in col or "lahkumine" in col or "hetk" in col:
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

    seg["sirge_kaugus_km"] = (seg["sirge_kaugus_m"] / 1000).round(2)
    seg["tee_kaugus_km"] = (seg["tee_kaugus_m"] / 1000).round(2)
    seg["kasutatav_kaugus_km"] = (seg["kasutatav_kaugus_m"] / 1000).round(2)
    seg_res["distance_km"] = (seg_res["distance_m"] / 1000).round(2)
    seg_res["minutes_total"] = seg_res["minutes_total"].apply(lambda x: math.ceil(x / 5) * 5).round(0)

    return cp, seg, starts, seg_res, cp_res, kp_load


def summarize_segment_classifications(segment_results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        segment_results_df
        .groupby(["segment_id", "light_classification"])
        .size()
        .reset_index(name="team_count")
        .sort_values(["segment_id", "light_classification"])
    )


# ======================================================
# KAARDIVADE
# ======================================================

def create_map(control_points: pd.DataFrame, segments: pd.DataFrame):
    cp = control_points.sort_values("jarjekord").reset_index(drop=True)
    center_lat = cp["lat"].mean()
    center_lon = cp["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for _, row in cp.iterrows():
        popup = (
            f"<b>{row['nimi']}</b><br>"
            f"KP ID: {row['kp_id']}<br>"
            f"MGRS: {row['mgrs']}<br>"
            f"Kestvus: {row['kestvus_min']} min"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=popup,
            tooltip=row["nimi"]
        ).add_to(m)

    cp_lookup = cp.set_index("kp_id").to_dict("index")
    for _, seg in segments.iterrows():
        start_cp = cp_lookup[seg["algus_kp_id"]]
        end_cp = cp_lookup[seg["lopp_kp_id"]]
        if "route_coords" in seg and seg["route_coords"] is not None:
            points = seg["route_coords"]
        else:
            points = [(start_cp["lat"], start_cp["lon"]), (end_cp["lat"], end_cp["lon"])]
        popup = (
            f"Lõik {seg['segment_id']}<br>"
            f"Tüüp: {seg['liikumisviis']}<br>"
            f"Kaugus: {seg['kasutatav_kaugus_m']/1000:.2f} km"
        )
        folium.PolyLine(points, popup=popup).add_to(m)

    return m


# ======================================================
# EXCEL EKSPORT
# ======================================================

def export_results_to_excel(results: dict) -> bytes:
    cp, seg, starts, seg_res, cp_res, kp_load = format_output_tables(results)
    class_summary = summarize_segment_classifications(results["segment_results"])

    from io import BytesIO
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cp.to_excel(writer, sheet_name="kontrollpunktid", index=False)
        seg.to_excel(writer, sheet_name="loigud", index=False)
        starts.to_excel(writer, sheet_name="stardiajad", index=False)
        seg_res.to_excel(writer, sheet_name="loigutulemused", index=False)
        cp_res.to_excel(writer, sheet_name="kp_ajad", index=False)
        kp_load.to_excel(writer, sheet_name="kp_koormus", index=False)
        class_summary.to_excel(writer, sheet_name="valgus_kokkuvote", index=False)

        # Formateeri race_config
        race_config_formatted = results["race_config"].copy()
        for key, value in race_config_formatted.items():
            if isinstance(value, datetime):
                race_config_formatted[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([race_config_formatted]).to_excel(writer, sheet_name="seadistused", index=False)

    output.seek(0)
    return output.getvalue()


# ======================================================
# MUUTMISFUNKTSIOONID
# ======================================================

def update_segment_speed(overrides: Dict[int, Dict[str, float]], segment_id: int, valge: Optional[float] = None, pime: Optional[float] = None):
    if segment_id not in overrides:
        overrides[segment_id] = {}
    if valge is not None:
        overrides[segment_id]["valge"] = float(valge)
    if pime is not None:
        overrides[segment_id]["pime"] = float(pime)


def update_control_point_mgrs(control_points: pd.DataFrame, kp_id: int, new_mgrs: str) -> pd.DataFrame:
    cp = control_points.copy()
    idx = cp.index[cp["kp_id"] == kp_id]
    if len(idx) == 0:
        raise ValueError(f"Kontrollpunkti kp_id={kp_id} ei leitud.")
    cp.loc[idx, "mgrs"] = new_mgrs
    return cp