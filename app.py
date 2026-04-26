from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils import (
    run_full_simulation,
    format_output_tables,
    summarize_segment_classifications,
    create_map,
    export_results_to_excel,
)


# ======================================================
# Vaikimisi andmed
# ======================================================

DEFAULT_RACE_CONFIG = {
    "esimese_voistkonna_start": "2026-05-15 20:00",
    "voistkondade_arv": 12,
    "stardi_intervall_min": 10,
    "nulltiimi_earlier_min": 45,
    "paeva_algus": "06:00",
    "pimeduse_algus": "22:00",
    "kasuta_automaatset_paikest": True,
    "voistluse_kuupaev": "2026-05-15",
    "timezone": "Europe/Tallinn",
}

DEFAULT_SPEEDS = {
    "tee": {"valge": 4.0, "pime": 3.5},
    "varjatud": {"valge": 2.0, "pime": 1.5},
}

DEFAULT_CONTROL_POINTS = pd.DataFrame([
    {"kp_id": 1, "nimi": "KP1", "mgrs": "35VLL2445309927", "kestvus_ettevalmistus_min": 5, "kestvus_uleanne_min": 7, "kestvus_tagasiside_min": 3, "kestvus_min": 15, "jarjekord": 1},
    {"kp_id": 2, "nimi": "KP2", "mgrs": "35VLL2863210814", "kestvus_ettevalmistus_min": 5, "kestvus_uleanne_min": 7, "kestvus_tagasiside_min": 3, "kestvus_min": 15, "jarjekord": 2},
    {"kp_id": 3, "nimi": "KP3", "mgrs": "35VLL3192511098", "kestvus_ettevalmistus_min": 10, "kestvus_uleanne_min": 15, "kestvus_tagasiside_min": 5, "kestvus_min": 30, "jarjekord": 3},
    {"kp_id": 4, "nimi": "KP4", "mgrs": "35VLL3479411624", "kestvus_ettevalmistus_min": 5, "kestvus_uleanne_min": 10, "kestvus_tagasiside_min": 5, "kestvus_min": 20, "jarjekord": 4},
])

# ======================================================
# UI abifunktsioonid
# ======================================================

def parse_default_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def rename_columns(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    visible_labels = {key: value for key, value in labels.items() if key in df.columns}
    return df.rename(columns=visible_labels)


def prepare_control_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not df.empty:
        df = df.dropna(subset=["kp_id"])
    for column in ["kestvus_ettevalmistus_min", "kestvus_uleanne_min", "kestvus_tagasiside_min"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df["jarjekord"] = range(1, len(df) + 1)
    return df.astype({
        "kp_id": int,
        "kestvus_ettevalmistus_min": int,
        "kestvus_uleanne_min": int,
        "kestvus_tagasiside_min": int,
        "jarjekord": int,
    })


def prepare_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not df.empty:
        df = df.dropna(subset=["segment_id"])
    df = df.astype({
        "segment_id": int,
        "algus_kp_id": int,
        "lopp_kp_id": int,
    })
    return df


def apply_default_durations(df: pd.DataFrame, preparation_min: int, task_min: int, feedback_min: int) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "kestvus_ettevalmistus_min": preparation_min,
        "kestvus_uleanne_min": task_min,
        "kestvus_tagasiside_min": feedback_min,
    }
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default_value)
    return df


def build_segments_from_control_points(control_points: pd.DataFrame, start_movement_mode: str) -> pd.DataFrame:
    cp = control_points.copy()
    if cp.empty or "kp_id" not in cp.columns:
        return pd.DataFrame(columns=["segment_id", "algus_kp_id", "lopp_kp_id", "liikumisviis"])

    cp = cp.dropna(subset=["kp_id"]).reset_index(drop=True)
    rows = []
    previous_kp_id = 0
    for index, row in cp.iterrows():
        segment_id = index + 1
        mode = start_movement_mode if segment_id == 1 else "tee"
        current_kp_id = int(row["kp_id"])
        rows.append({
            "segment_id": segment_id,
            "algus_kp_id": previous_kp_id,
            "lopp_kp_id": current_kp_id,
            "liikumisviis": mode,
        })
        previous_kp_id = current_kp_id
    return pd.DataFrame(rows)


def prepare_overrides(df: pd.DataFrame) -> dict:
    df = df.copy()
    if df.empty:
        return {}

    df = df.dropna(subset=["segment_id", "liikumiskiirus"])
    if df.empty:
        return {}

    df = df.astype({
        "segment_id": int,
        "liikumiskiirus": float,
    })
    return {
        int(row["segment_id"]): {
            "valge": float(row["liikumiskiirus"]),
            "pime": float(row["liikumiskiirus"]),
        }
        for _, row in df.iterrows()
    }


def render_downloads(results: dict):
    excel_data = export_results_to_excel(results)
    st.download_button(
        label="Laadi alla täielik Excel",
        data=excel_data,
        file_name="raja_tulemused.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    try:
        from utils import export_variant1
        variant1_data = export_variant1(results)
        st.download_button(
            label="Variant 1: iga võistkond eraldi",
            data=variant1_data,
            file_name="variant1_iga_vk_eraldi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    except ImportError:
        st.caption("Variant 1 eksport ei ole saadaval.")

    try:
        from utils import export_variant2
        variant2_data = export_variant2(results)
        st.download_button(
            label="Variant 2: kontrollpunktide kaupa",
            data=variant2_data,
            file_name="variant2_kp_de_kaupa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    except ImportError:
        st.caption("Variant 2 eksport ei ole saadaval.")


def render_summary(results: dict, seg_out: pd.DataFrame, kp_load_out: pd.DataFrame, class_summary: pd.DataFrame):
    total_distance_km = results["segments"]["kasutatav_kaugus_m"].sum() / 1000
    first_start = pd.to_datetime(results["start_times"]["start_time"]).min().strftime("%d.%m %H:%M")
    last_departure = pd.to_datetime(results["checkpoint_results"]["departure_time"]).max().strftime("%d.%m %H:%M")
    mixed_segments = int((class_summary["sega"] > 0).sum()) if "sega" in class_summary.columns else 0

    if kp_load_out.empty:
        peak_load_label = "-"
    else:
        peak_row = kp_load_out.sort_values("maksimaalne_koormus", ascending=False).iloc[0]
        peak_load_label = f"{peak_row['kp_nimi']} ({peak_row['maksimaalne_koormus']})"

    metric_cols = st.columns(5)
    metric_cols[0].metric("Võistkondi", int(results["race_config"]["voistkondade_arv"]))
    metric_cols[1].metric("Raja pikkus", f"{total_distance_km:.1f} km")
    metric_cols[2].metric("Ajavahemik", f"{first_start} - {last_departure}")
    metric_cols[3].metric("Segavalgusega lõike", mixed_segments)
    metric_cols[4].metric("Suurim KP koormus", peak_load_label)

    st.subheader("Valguspiirid")
    sun_cols = st.columns(4)
    sun_cols[0].metric("Päeva algus", results["race_config"]["paeva_algus"])
    sun_cols[1].metric("Pime algab", results["race_config"]["pimeduse_algus"])
    if "sunrise_full" in results["race_config"]:
        sun_cols[2].metric("Päikesetõus", pd.to_datetime(results["race_config"]["sunrise_full"]).strftime("%H:%M"))
        sun_cols[3].metric("Päikeseloojang", pd.to_datetime(results["race_config"]["sunset_full"]).strftime("%H:%M"))

    st.subheader("Olulised lõigud")
    segment_columns = [
        "segment_id", "algus_kp_id", "lopp_kp_id", "liikumisviis",
        "kasutatav_kaugus_km", "valgustingimused", "liikumiskiirus",
        "liikumise aeg ümardatud (min)",
    ]
    st.dataframe(
        rename_columns(seg_out[segment_columns], TABLE_LABELS),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Kontrollpunktide koormus")
    st.dataframe(
        rename_columns(kp_load_out, TABLE_LABELS),
        width="stretch",
        hide_index=True,
    )


TABLE_LABELS = {
    "team_id": "Võistkond",
    "segment_id": "Lõik",
    "algus_kp_id": "Algus KP",
    "lopp_kp_id": "Lõpp KP",
    "liikumisviis": "Liikumisviis",
    "sirge_kaugus_km": "Sirge kaugus (km)",
    "tee_kaugus_km": "Tee kaugus (km)",
    "kasutatav_kaugus_km": "Arvestatud kaugus (km)",
    "valgustingimused": "Valgus",
    "liikumiskiirus": "Kiirus (km/h)",
    "liikumise aeg täpne (min)": "Liikumisaeg täpne (min)",
    "liikumise aeg ümardatud (min)": "Liikumisaeg ümardatud (min)",
    "distance_note": "Distantsi allikas",
    "start_time": "Algus",
    "end_time": "Lõpp",
    "distance_m": "Kaugus (m)",
    "distance_km": "Kaugus (km)",
    "reaalne_valgustingimus": "Tegelik valgus",
    "chosen_light_condition": "Arvutuses kasutatud valgus",
    "exact_minutes": "Täpne aeg (min)",
    "minutes_total": "Ümardatud aeg (min)",
    "kp_id": "KP",
    "kp_nimi": "Kontrollpunkt",
    "arrival_time": "Saabumine",
    "departure_time": "Lahkumine",
    "maksimaalne_koormus": "Maksimaalne koormus",
    "maks_koormuse_hetk": "Maks koormuse hetk",
    "kokku_kulastusi": "Külastusi",
    "varaseim_saabumine": "Varaseim saabumine",
    "hiliseim_lahkumine": "Hiliseim lahkumine",
    "valge": "Valges",
    "pime": "Pimedas",
    "sega": "Segavalguses",
    "nimi": "Nimi",
    "mgrs": "MGRS",
    "kestvus_ettevalmistus_min": "Ettevalmistus (min)",
    "kestvus_uleanne_min": "Ülesanne (min)",
    "kestvus_tagasiside_min": "Tagasiside (min)",
    "kestvus_min": "Kokku (min)",
    "jarjekord": "Järjekord",
}


# ======================================================
# Streamlit App
# ======================================================

st.set_page_config(
    page_title="Raja arvutamise tööriist",
    layout="wide",
)

default_start = parse_default_datetime(DEFAULT_RACE_CONFIG["esimese_voistkonna_start"])
default_race_date = parse_default_datetime(f"{DEFAULT_RACE_CONFIG['voistluse_kuupaev']} 00:00").date()
default_day_start = parse_default_datetime(f"2026-01-01 {DEFAULT_RACE_CONFIG['paeva_algus']}").time()
default_dark_start = parse_default_datetime(f"2026-01-01 {DEFAULT_RACE_CONFIG['pimeduse_algus']}").time()

st.title("Võistlusmatka raja arvutamine")
st.caption("Sisesta rada, stardid ja liikumiskiirused. Tulemused koondatakse ülevaateks, tabeliteks, ajajooneks ja kaardiks.")

with st.sidebar:
    st.header("Arvutus")
    start_date = st.date_input("Esimese võistkonna stardipäev", value=default_start.date())
    start_clock = st.time_input("Esimese võistkonna stardikell", value=default_start.time())
    team_count = st.number_input("Võistkondade arv", min_value=1, value=DEFAULT_RACE_CONFIG["voistkondade_arv"])
    interval = st.number_input("Stardi intervall (min)", min_value=1, value=DEFAULT_RACE_CONFIG["stardi_intervall_min"])
    zero_early = st.number_input("0-tiim enne (min)", min_value=0, value=DEFAULT_RACE_CONFIG["nulltiimi_earlier_min"])

    st.divider()
    st.header("Valgus")
    race_date = st.date_input("Võistluse kuupäev", value=default_race_date)
    timezone = st.text_input("Ajavöönd", DEFAULT_RACE_CONFIG["timezone"])
    auto_sun = st.checkbox("Arvuta päike automaatselt", value=DEFAULT_RACE_CONFIG["kasuta_automaatset_paikest"])
    day_start_time = st.time_input("Päeva algus", value=default_day_start, disabled=auto_sun)
    dark_start_time = st.time_input("Pime algab", value=default_dark_start, disabled=auto_sun)

    st.divider()
    st.header("Start")
    start_mgrs = st.text_input("Start MGRS", value="35VLL2445309927", help="Stardi asukoht MGRS formaadis")
    start_duration_min = st.number_input("Start kestus (min)", min_value=0, value=0, help="Aeg stardis enne liikumist")
    start_movement_mode = st.selectbox("Start -> KP1 liikumisviis", ["tee", "varjatud"], index=0)

input_tab, result_tab, map_tab, export_tab = st.tabs(["Sisendid", "Tulemused", "Kaart", "Eksport"])

with input_tab:
    st.subheader("Rada")
    duration_cols = st.columns(3)
    default_preparation_min = duration_cols[0].number_input("Vaikimisi ettevalmistus (min)", min_value=0, value=5)
    default_task_min = duration_cols[1].number_input("Vaikimisi ülesanne (min)", min_value=0, value=10)
    default_feedback_min = duration_cols[2].number_input("Vaikimisi tagasiside (min)", min_value=0, value=5)

    st.markdown("**Kontrollpunktid**")
    control_points_base = apply_default_durations(
        DEFAULT_CONTROL_POINTS[[
            "kp_id", "nimi", "mgrs",
        ]],
        default_preparation_min,
        default_task_min,
        default_feedback_min,
    )
    control_points_df = st.data_editor(
        control_points_base,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="control_points_editor_v2",
        column_config={
            "kp_id": st.column_config.NumberColumn("KP", min_value=1, step=1),
            "nimi": st.column_config.TextColumn("Nimi"),
            "mgrs": st.column_config.TextColumn("MGRS"),
            "kestvus_ettevalmistus_min": st.column_config.NumberColumn("Ettevalmistus", min_value=0, step=1),
            "kestvus_uleanne_min": st.column_config.NumberColumn("Ülesanne", min_value=0, step=1),
            "kestvus_tagasiside_min": st.column_config.NumberColumn("Tagasiside", min_value=0, step=1),
        },
    )

    control_points_for_segments = apply_default_durations(
        control_points_df,
        default_preparation_min,
        default_task_min,
        default_feedback_min,
    )
    generated_segments = build_segments_from_control_points(control_points_for_segments, start_movement_mode)

    st.markdown("**Automaatselt loodud lõigud**")
    segments_df = st.data_editor(
        generated_segments,
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        key="segments_editor_v2",
        disabled=["segment_id", "algus_kp_id", "lopp_kp_id"],
        column_config={
            "segment_id": st.column_config.NumberColumn("Lõik"),
            "algus_kp_id": st.column_config.NumberColumn("Algus KP"),
            "lopp_kp_id": st.column_config.NumberColumn("Lõpp KP"),
            "liikumisviis": st.column_config.SelectboxColumn("Liikumisviis", options=["tee", "varjatud"], required=True),
        },
    )

    st.subheader("Kiirused")
    speed_cols = st.columns([1, 1])
    with speed_cols[0]:
        st.dataframe(
            pd.DataFrame([
                {"Liikumisviis": "tee", "Päev (km/h)": DEFAULT_SPEEDS["tee"]["valge"], "Öö (km/h)": DEFAULT_SPEEDS["tee"]["pime"]},
                {"Liikumisviis": "varjatud", "Päev (km/h)": DEFAULT_SPEEDS["varjatud"]["valge"], "Öö (km/h)": DEFAULT_SPEEDS["varjatud"]["pime"]},
            ]),
            width="stretch",
            hide_index=True,
        )
    with speed_cols[1]:
        overrides_df = st.data_editor(
            pd.DataFrame(columns=["segment_id", "liikumiskiirus"]),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="overrides_editor",
            column_config={
                "segment_id": st.column_config.NumberColumn("Lõik", min_value=1, step=1),
                "liikumiskiirus": st.column_config.NumberColumn("Kiirus (km/h)", min_value=0.1, step=0.1),
            },
        )

    calculate = st.button("Arvuta rada", type="primary", width="stretch")

    if calculate:
        try:
            race_config = {
                "esimese_voistkonna_start": f"{start_date.isoformat()} {start_clock.strftime('%H:%M')}",
                "voistkondade_arv": int(team_count),
                "stardi_intervall_min": int(interval),
                "nulltiimi_earlier_min": int(zero_early),
                "voistluse_kuupaev": race_date.isoformat(),
                "timezone": timezone,
                "kasuta_automaatset_paikest": auto_sun,
                "paeva_algus": day_start_time.strftime("%H:%M"),
                "pimeduse_algus": dark_start_time.strftime("%H:%M"),
            }

            control_points_parsed = prepare_control_points(
                apply_default_durations(
                    control_points_df,
                    default_preparation_min,
                    default_task_min,
                    default_feedback_min,
                )
            )
            segments_parsed = prepare_segments(segments_df)
            overrides = prepare_overrides(overrides_df)

            with st.spinner("Arvutan distantse, valgusolusid ja ajagraafikut..."):
                results = run_full_simulation(
                    control_points_parsed,
                    segments_parsed,
                    race_config,
                    DEFAULT_SPEEDS,
                    overrides,
                    start_mgrs,
                    start_duration_min,
                )

            st.session_state["results"] = results
            st.success("Arvutus lõpetatud. Ava tulemuste või kaardi vahekaart.")
        except Exception as e:
            st.error(f"Arvutus ebaõnnestus: {e}")
            st.info("Kontrolli, et MGRS väärtused, KP viited, kuupäevad ja numbrilised väljad oleksid korrektsed.")

if "results" not in st.session_state:
    with result_tab:
        st.info("Sisesta raja andmed ja vajuta vahekaardil Sisendid nuppu Arvuta rada.")
    with map_tab:
        st.info("Kaart ilmub pärast arvutust.")
    with export_tab:
        st.info("Ekspordid ilmuvad pärast arvutust.")
else:
    results = st.session_state["results"]
    cp_out, seg_out, starts_out, seg_res_out, cp_res_out, kp_load_out, cp_sync_out, sync_fig = format_output_tables(results)
    class_summary = summarize_segment_classifications(results["segment_results"])

    with result_tab:
        overview_tab, segments_tab, checkpoints_tab, teams_tab, sync_tab = st.tabs([
            "Ülevaade", "Lõigud", "Kontrollpunktid", "Võistkonnad", "Ajajoon",
        ])

        with overview_tab:
            render_summary(results, seg_out, kp_load_out, class_summary)

        with segments_tab:
            segment_columns = [
                "segment_id", "algus_kp_id", "lopp_kp_id", "liikumisviis",
                "sirge_kaugus_km", "tee_kaugus_km", "kasutatav_kaugus_km",
                "valgustingimused", "liikumiskiirus",
                "liikumise aeg täpne (min)", "liikumise aeg ümardatud (min)",
                "distance_note",
            ]
            st.dataframe(
                rename_columns(seg_out[segment_columns], TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Valgusklassifikatsioon võistkondade kaupa**")
            st.dataframe(
                rename_columns(class_summary, TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )

        with checkpoints_tab:
            st.markdown("**Kontrollpunktid**")
            st.dataframe(
                rename_columns(cp_out, TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Koormus**")
            st.dataframe(
                rename_columns(kp_load_out, TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**KP ajad võistkondade kaupa**")
            st.dataframe(
                rename_columns(cp_res_out, TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )

        with teams_tab:
            st.markdown("**Stardiajad**")
            st.dataframe(
                rename_columns(starts_out, TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Lõigutulemused**")
            display_columns = [
                "team_id", "segment_id", "algus_kp_id", "lopp_kp_id",
                "start_time", "end_time", "distance_km",
                "reaalne_valgustingimus", "chosen_light_condition",
                "exact_minutes", "minutes_total",
            ]
            display_columns = [col for col in display_columns if col in seg_res_out.columns]
            st.dataframe(
                rename_columns(seg_res_out[display_columns], TABLE_LABELS),
                width="stretch",
                hide_index=True,
            )

        with sync_tab:
            st.plotly_chart(sync_fig, width="stretch")

    with map_tab:
        st.subheader("Kaardivaade")
        m = create_map(cp_out, results["segments"], results["checkpoint_results"], seg_res_out)
        st_folium(m, width=1200, height=560)

    with export_tab:
        st.subheader("Ekspordi tulemused")
        render_downloads(results)
