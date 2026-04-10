import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_folium import st_folium
import folium

from utils import (
    run_full_simulation,
    format_output_tables,
    summarize_segment_classifications,
    create_map,
    export_results_to_excel,
    validate_inputs,
)

# ======================================================
# VAikimisi andmed
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

DEFAULT_SEGMENTS = pd.DataFrame([
    {"segment_id": 1, "algus_kp_id": 0, "lopp_kp_id": 1, "liikumisviis": "tee"},
    {"segment_id": 2, "algus_kp_id": 1, "lopp_kp_id": 2, "liikumisviis": "varjatud"},
    {"segment_id": 3, "algus_kp_id": 2, "lopp_kp_id": 3, "liikumisviis": "tee"},
    {"segment_id": 4, "algus_kp_id": 3, "lopp_kp_id": 4, "liikumisviis": "tee"},
])

# ======================================================
# Streamlit App
# ======================================================

st.title("Võistlusmatka Raja Arvutamise Tööriist")

st.sidebar.header("Sisendandmed")

# Kontrollpunktid
with st.sidebar:
    st.subheader("Kontrollpunktid")
    control_points_df = st.data_editor(
        DEFAULT_CONTROL_POINTS[["kp_id", "nimi", "mgrs", "kestvus_ettevalmistus_min", "kestvus_uleanne_min", "kestvus_tagasiside_min", "jarjekord"]],
        num_rows="dynamic",
        use_container_width=True,
        key="control_points_editor",
    )

    # Lõigud
    st.subheader("Lõigud")
    segments_df = st.data_editor(
        DEFAULT_SEGMENTS,
        num_rows="dynamic",
        use_container_width=True,
        key="segments_editor",
    )

    # Kiiruste ülekirjutused
    st.subheader("Kiiruste ülekirjutused")
    overrides_df = st.data_editor(
        pd.DataFrame(
            [
                {"segment_id": 1, "algus_kp_id": 0, "lopp_kp_id": 1, "liikumisviis": "tee", "liikumiskiirus": 4.0},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="overrides_editor",
    )

# Start koordinaat
st.sidebar.subheader("Start")
start_mgrs = st.sidebar.text_input("Start MGRS", value="35VLL2445309927", help="Stardi asukoht MGRS formaadis")
start_duration_min = st.sidebar.number_input("Start kestus (min)", min_value=0, value=0, help="Aeg stardis enne liikumist")
start_movement_mode = st.sidebar.selectbox("Start liikumisviis", ["tee", "varjatud"], index=0, help="Kuidas liigutakse stardist KP1-ni")

st.sidebar.markdown(
    "**Vaikekiirused:**\n"
    "- Varjatud päev: 2 km/h\n"
    "- Varjatud öö: 1.5 km/h\n"
    "- Tee päev: 4 km/h\n"
    "- Tee öö: 3.5 km/h\n"
    "Kui lõik on segalõik, siis arvestatakse automaatselt öökiirusega, kui osa ajast on pimedas."
)

# Konfiguratsioon
st.sidebar.subheader("Konfiguratsioon")
start_time = st.sidebar.text_input("Esimese võistkonna start", DEFAULT_RACE_CONFIG["esimese_voistkonna_start"])
team_count = st.sidebar.number_input("Võistkondade arv", min_value=1, value=DEFAULT_RACE_CONFIG["voistkondade_arv"])
interval = st.sidebar.number_input("Stardi intervall (min)", min_value=1, value=DEFAULT_RACE_CONFIG["stardi_intervall_min"])
zero_early = st.sidebar.number_input("0-tiim enne (min)", min_value=0, value=DEFAULT_RACE_CONFIG["nulltiimi_earlier_min"])
race_date = st.sidebar.text_input("Võistluse kuupäev", DEFAULT_RACE_CONFIG["voistluse_kuupaev"])
timezone = st.sidebar.text_input("Ajavöönd", DEFAULT_RACE_CONFIG["timezone"])
auto_sun = st.sidebar.checkbox("Automaatne päike", value=DEFAULT_RACE_CONFIG["kasuta_automaatset_paikest"])
day_start = st.sidebar.text_input("Päeva algus (kui mitte auto)", DEFAULT_RACE_CONFIG["paeva_algus"])
dark_start = st.sidebar.text_input("Pime algab (kui mitte auto)", DEFAULT_RACE_CONFIG["pimeduse_algus"])

if st.sidebar.button("Arvuta"):
    try:
        control_points_df = control_points_df.copy()
        control_points_df = control_points_df.dropna(subset=["kp_id"]) if not control_points_df.empty else control_points_df
        control_points_df = control_points_df.astype({
            "kp_id": int,
            "kestvus_ettevalmistus_min": int,
            "kestvus_uleanne_min": int,
            "kestvus_tagasiside_min": int,
            "jarjekord": int,
        })

        segments_df = segments_df.copy()
        segments_df = segments_df.dropna(subset=["segment_id"]) if not segments_df.empty else segments_df
        segments_df = segments_df.astype({
            "segment_id": int,
            "algus_kp_id": int,
            "lopp_kp_id": int,
        })

        # Muuda esimene segment start liikumisviisiks
        segments_df.loc[segments_df["segment_id"] == 1, "liikumisviis"] = start_movement_mode

        overrides_df_parsed = overrides_df.copy()
        if not overrides_df_parsed.empty:
            overrides_df_parsed = overrides_df_parsed.dropna(subset=["segment_id"])
            overrides_df_parsed = overrides_df_parsed.astype({
                "segment_id": int,
                "algus_kp_id": int,
                "lopp_kp_id": int,
                "liikumiskiirus": float,
            })

        race_config = {
            "esimese_voistkonna_start": start_time,
            "voistkondade_arv": int(team_count),
            "stardi_intervall_min": int(interval),
            "nulltiimi_earlier_min": int(zero_early),
            "voistluse_kuupaev": race_date,
            "timezone": timezone,
            "kasuta_automaatset_paikest": auto_sun,
            "paeva_algus": day_start,
            "pimeduse_algus": dark_start,
        }

        overrides = {}
        if not overrides_df_parsed.empty:
            for _, row in overrides_df_parsed.iterrows():
                overrides[int(row["segment_id"])] = {
                    "valge": float(row["liikumiskiirus"]),
                    "pime": float(row["liikumiskiirus"]),
                }

        # Käivita simulatsioon
        results = run_full_simulation(control_points_df, segments_df, race_config, DEFAULT_SPEEDS, overrides, start_mgrs, start_duration_min)

        # Salvesta sessiooni
        st.session_state["results"] = results

        st.success("Arvutus lõpetatud!")

    except Exception as e:
        st.error(f"Viga: {e}")

# Tulemuste kuvamine
if "results" in st.session_state:
    results = st.session_state["results"]

    st.header("Päikese info / Valguspiirid")
    st.write(f"Päeva algus: {results['race_config']['paeva_algus']}")
    st.write(f"Pime algab: {results['race_config']['pimeduse_algus']}")
    if "sunrise_full" in results["race_config"]:
        st.write(f"Päikesetõus: {results['race_config']['sunrise_full']}")
        st.write(f"Päikeseloojang: {results['race_config']['sunset_full']}")

    cp_out, seg_out, starts_out, seg_res_out, cp_res_out, kp_load_out = format_output_tables(results)
    class_summary = summarize_segment_classifications(results["segment_results"])

    st.header("Kontrollpunktid")
    st.dataframe(cp_out)

    st.header("Lõigud")
    st.dataframe(seg_out[[
        "segment_id", "algus_kp_id", "lopp_kp_id", "liikumisviis",
        "sirge_kaugus_km", "tee_kaugus_km", "kasutatav_kaugus_km",
        "distance_note"
    ]])

    st.header("Stardiajad")
    st.dataframe(starts_out)

    st.header("Lõikude valgusklassifikatsiooni kokkuvõte")
    st.dataframe(class_summary)

    st.header("Kontrollpunktide koormus")
    st.dataframe(kp_load_out)

    st.header("Võistkondade lõigutulemused")
    st.dataframe(seg_res_out)

    st.header("Võistkondade KP ajad")
    st.dataframe(cp_res_out)

    st.header("Kaardivaade")
    m = create_map(cp_out, results["segments"])
    st_folium(m, width=700, height=500)

    # Excel eksport
    excel_data = export_results_to_excel(results)
    st.download_button(
        label="Laadi alla Excel",
        data=excel_data,
        file_name="raja_tulemused.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )