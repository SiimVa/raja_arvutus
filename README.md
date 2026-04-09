# Võistlusmatka Raja Arvutamise Tööriist

See on Streamlit rakendus võistlusmatka raja planeerimise ja ajagraafiku simulatsiooni jaoks. Rakendus arvutab võistkondade liikumise ajas ja ruumis, arvestades kontrollpunkte, liikumisviise, kiirusi ja valgusolusid.

## Funktsioonid

- **Sisend**: Kontrollpunktide asukohad (MGRS), lõikude liikumisviisid, võistkondade stardid ja intervallid.
- **Distantside arvutus**: Tee-pikkused OSRM-iga, varjatud lõikudel linnulend × 1.5.
- **Ajasimulatsioon**: Iga võistkonna ajagraafik kontrollpunktides.
- **Valgusanalüüs**: Päikesetõusu/loojangu automaatne arvutus, lõikude klassifikatsioon (valge/pime/segalõik).
- **Koormusanalüüs**: Kontrollpunktide maksimaalne koormus.
- **Visualiseerimine**: Tabelid ja interaktiivne kaart.
- **Iteratiivne parandamine**: Muuda kiirusi ja arvuta uuesti.
- **Eksport**: Excel faili allalaadimine.

## Käivitus

1. Klooni repo.
2. Installi sõltuvused: `pip install -r requirements.txt`
3. Käivita: `streamlit run app.py`
4. Ava brauseris http://localhost:8501

## GitHub Hosting

Rakendus on mõeldud hostimiseks Streamlit Cloud'is või sarnasel platvormil.

- Loo uus repo GitHub'is.
- Laadi üles failid: `app.py`, `utils.py`, `requirements.txt`, `README.md`.
- Loo Streamlit Cloud konto ja seo repo.
- Rakendus töötab avalikult.

## Kasutus

- Sisesta andmed külgribas.
- Vajuta "Arvuta".
- Vaata tulemusi ja kaarti.
- Laadi alla Excel.

## Teekid

- streamlit
- pandas
- geopy
- mgrs
- requests
- folium
- streamlit-folium
- openpyxl
- astral