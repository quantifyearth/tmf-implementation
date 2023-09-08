from gzip import GzipFile
import os
import sys
import tempfile
from glob import glob

import geopandas # type: ignore
import requests

from ..common import DownloadError

# Mapping of country codes to OSM IDs
# Derived from https://github.com/simonepri/osm-countries/blob/master/osm.json
# and the Wikipedia table of ISO country codes.
# XK is a temporary code for Kosovo, coded as XKX in that JSON file, which is
# not an official code, so that was fixed up manually.
# The list was sorted by country name, which is included for reference.
osm_id = {
    "AF": 303427,   # Afghanistan
    "AX": 1650407,  # Åland Islands
    "AL": 53292,    # Albania
    "DZ": 192756,   # Algeria
    "AS": 2177187,  # American Samoa
    "AD": 9407,     # Andorra
    "AO": 195267,   # Angola
    "AI": 2177161,  # Anguilla
    "AQ": 2186646,  # Antarctica
    "AG": 536900,   # Antigua and Barbuda
    "AR": 286393,   # Argentina
    "AM": 364066,   # Armenia
    "AW": 1231749,  # Aruba
    "AU": 80500,    # Australia
    "AT": 16239,    # Austria
    "AZ": 364110,   # Azerbaijan
    "BS": 547469,   # Bahamas
    "BH": 378734,   # Bahrain
    "BD": 184640,   # Bangladesh
    "BB": 547511,   # Barbados
    "BY": 59065,    # Belarus
    "BE": 52411,    # Belgium
    "BZ": 287827,   # Belize
    "BJ": 192784,   # Benin
    "BM": 1993208,  # Bermuda
    "BT": 184629,   # Bhutan
    "BO": 252645,   # Bolivia (Plurinational State of)
    "BQ": 1216720,  # Bonaire, Sint Eustatius and Saba
    "BA": 2528142,  # Bosnia and Herzegovina
    "BW": 1889339,  # Botswana
    "BV": 2425963,  # Bouvet Island
    "BR": 59470,    # Brazil
    "IO": 1993867,  # British Indian Ocean Territory
    "BN": 2103120,  # Brunei Darussalam
    "BG": 186382,   # Bulgaria
    "BF": 192783,   # Burkina Faso
    "BI": 195269,   # Burundi
    "CV": 535774,   # Cabo Verde
    "KH": 49898,    # Cambodia
    "CM": 192830,   # Cameroon
    "CA": 1428125,  # Canada
    "KY": 2185366,  # Cayman Islands
    "CF": 192790,   # Central African Republic
    "TD": 2361304,  # Chad
    "CL": 167454,   # Chile
    "CN": 270056,   # China
    "CX": 2177207,  # Christmas Island
    "CC": 82636,    # Cocos (Keeling) Islands
    "CO": 120027,   # Colombia
    "KM": 535790,   # Comoros
    "CG": 192794,   # Congo
    "CD": 192795,   # Congo, Democratic Republic of the
    "CK": 2184233,  # Cook Islands
    "CR": 287667,   # Costa Rica
    "CI": 192779,   # Côte d'Ivoire
    "HR": 214885,   # Croatia
    "CU": 307833,   # Cuba
    "CW": 1216719,  # Curaçao
    "CY": 307787,   # Cyprus
    "CZ": 51684,    # Czechia
    "DK": 50046,    # Denmark
    "DJ": 192801,   # Djibouti
    "DM": 307823,   # Dominica
    "DO": 307828,   # Dominican Republic
    "EC": 108089,   # Ecuador
    "EG": 1473947,  # Egypt
    "SV": 1520612,  # El Salvador
    "GQ": 192791,   # Equatorial Guinea
    "ER": 296961,   # Eritrea
    "EE": 79510,    # Estonia
    "SZ": 88210,    # Eswatini
    "ET": 192800,   # Ethiopia
    "FK": 2185374,  # Falkland Islands (Malvinas)
    "FO": 52939,    # Faroe Islands
    "FJ": 571747,   # Fiji
    "FI": 54224,    # Finland
    "FR": 2202162,  # France
    "GF": 1260551,  # French Guiana
    "PF": 3412620,  # French Polynesia
    "TF": 2186658,  # French Southern Territories
    "GA": 192793,   # Gabon
    "GM": 192774,   # Gambia
    "GE": 28699,    # Georgia
    "DE": 51477,    # Germany
    "GH": 192781,   # Ghana
    "GI": 1278736,  # Gibraltar
    "GR": 192307,   # Greece
    "GL": 2184073,  # Greenland
    "GD": 550727,   # Grenada
    "GP": 1047206,  # Guadeloupe
    "GU": 306001,   # Guam
    "GT": 1521463,  # Guatemala
    "GG": 270009,   # Guernsey
    "GN": 192778,   # Guinea
    "GW": 192776,   # Guinea-Bissau
    "GY": 287083,   # Guyana
    "HT": 307829,   # Haiti
    "HM": 2177227,  # Heard Island and McDonald Islands
    "VA": 36989,    # Holy See
    "HN": 287670,   # Honduras
    "HK": 913110,   # Hong Kong
    "HU": 21335,    # Hungary
    "IS": 299133,   # Iceland
    "IN": 304716,   # India
    "ID": 304751,   # Indonesia
    "IR": 304938,   # Iran (Islamic Republic of)
    "IQ": 304934,   # Iraq
    "IE": 62273,    # Ireland
    "IM": 62269,    # Isle of Man
    "IL": 1473946,  # Israel
    "IT": 365331,   # Italy
    "JM": 555017,   # Jamaica
    "JP": 382313,   # Japan
    "JE": 367988,   # Jersey
    "JO": 184818,   # Jordan
    "KZ": 214665,   # Kazakhstan
    "KE": 192798,   # Kenya
    "KI": 571178,   # Kiribati
    "KP": 192734,   # Korea (Democratic People's Republic of)
    "KR": 307756,   # Korea, Republic of
    "XK": 2088990,  # Kosovo
    "KW": 305099,   # Kuwait
    "KG": 178009,   # Kyrgyzstan
    "LA": 49903,    # Lao People's Democratic Republic
    "LV": 72594,    # Latvia
    "LB": 184843,   # Lebanon
    "LS": 2093234,  # Lesotho
    "LR": 192780,   # Liberia
    "LY": 192758,   # Libya
    "LI": 1155955,  # Liechtenstein
    "LT": 72596,    # Lithuania
    "LU": 2171347,  # Luxembourg
    "MO": 1867188,  # Macao
    "MG": 447325,   # Madagascar
    "MW": 195290,   # Malawi
    "MY": 2108121,  # Malaysia
    "MV": 536773,   # Maldives
    "ML": 192785,   # Mali
    "MT": 365307,   # Malta
    "MH": 571771,   # Marshall Islands
    "MQ": 1260552,  # Martinique
    "MR": 192763,   # Mauritania
    "MU": 535828,   # Mauritius
    "YT": 1259885,  # Mayotte
    "MX": 114686,   # Mexico
    "FM": 571802,   # Micronesia (Federated States of)
    "MD": 58974,    # Moldova, Republic of
    "MC": 1124039,  # Monaco
    "MN": 161033,   # Mongolia
    "ME": 53296,    # Montenegro
    "MS": 537257,   # Montserrat
    "MA": 3630439,  # Morocco
    "MZ": 195273,   # Mozambique
    "MM": 50371,    # Myanmar
    "NA": 195266,   # Namibia
    "NR": 571804,   # Nauru
    "NP": 184633,   # Nepal
    "NL": 2323309,  # Netherlands, Kingdom of the
    "NC": 3407643,  # New Caledonia
    "NZ": 556706,   # New Zealand
    "NI": 287666,   # Nicaragua
    "NE": 192786,   # Niger
    "NG": 192787,   # Nigeria
    "NU": 1558556,  # Niue
    "NF": 2574988,  # Norfolk Island
    "MK": 53293,    # North Macedonia
    "MP": 306004,   # Northern Mariana Islands
    "NO": 2978650,  # Norway
    "OM": 305138,   # Oman
    "PK": 307573,   # Pakistan
    "PW": 571805,   # Palau
    "PS": 1703814,  # Palestine, State of
    "PA": 287668,   # Panama
    "PG": 307866,   # Papua New Guinea
    "PY": 287077,   # Paraguay
    "PE": 288247,   # Peru
    "PH": 443174,   # Philippines
    "PN": 2185375,  # Pitcairn
    "PL": 49715,    # Poland
    "PT": 295480,   # Portugal
    "PR": 4422604,  # Puerto Rico
    "QA": 305095,   # Qatar
    "RE": 2470060,  # Réunion
    "RO": 90689,    # Romania
    "RU": 60189,    # Russian Federation
    "RW": 171496,   # Rwanda
    "BL": 537967,   # Saint Barthélemy
    "SH": 1964272,  # Saint Helena, Ascension and Tristan da Cunha
    "KN": 536899,   # Saint Kitts and Nevis
    "LC": 550728,   # Saint Lucia
    "MF": 1891583,  # Saint Martin (French part)
    "PM": 3406826,  # Saint Pierre and Miquelon
    "VC": 550725,   # Saint Vincent and the Grenadines
    "WS": 1872673,  # Samoa
    "SM": 54624,    # San Marino
    "ST": 535880,   # Sao Tome and Principe
    "SA": 307584,   # Saudi Arabia
    "SN": 192775,   # Senegal
    "RS": 1741311,  # Serbia
    "SC": 536765,   # Seychelles
    "SL": 192777,   # Sierra Leone
    "SG": 536780,   # Singapore
    "SX": 1231790,  # Sint Maarten (Dutch part)
    "SK": 14296,    # Slovakia
    "SI": 218657,   # Slovenia
    "SB": 1857436,  # Solomon Islands
    "SO": 192799,   # Somalia
    "ZA": 87565,    # South Africa
    "GS": 1983628,  # South Georgia and the South Sandwich Islands
    "SS": 1656678,  # South Sudan
    "ES": 1311341,  # Spain
    "LK": 536807,   # Sri Lanka
    "SD": 192789,   # Sudan
    "SR": 287082,   # Suriname
    "SJ": 3245620,  # Svalbard and Jan Mayen
    "SE": 52822,    # Sweden
    "CH": 51701,    # Switzerland
    "SY": 184840,   # Syrian Arab Republic
    "TW": 449220,   # Taiwan, Province of China
    "TJ": 214626,   # Tajikistan
    "TZ": 195270,   # Tanzania, United Republic of
    "TH": 2067731,  # Thailand
    "TL": 305142,   # Timor-Leste
    "TG": 192782,   # Togo
    "TK": 2186600,  # Tokelau
    "TO": 2186665,  # Tonga
    "TT": 555717,   # Trinidad and Tobago
    "TN": 192757,   # Tunisia
    "TR": 174737,   # Türkiye
    "TM": 223026,   # Turkmenistan
    "TC": 547479,   # Turks and Caicos Islands
    "TV": 2177266,  # Tuvalu
    "UG": 192796,   # Uganda
    "UA": 60199,    # Ukraine
    "AE": 307763,   # United Arab Emirates
    "GB": 62149,    # United Kingdom of Great Britain and Northern Ireland
    "UM": 2185386,  # United States Minor Outlying Islands
    "US": 148838,   # United States of America
    "UY": 287072,   # Uruguay
    "UZ": 196240,   # Uzbekistan
    "VU": 2177246,  # Vanuatu
    "VE": 272644,   # Venezuela (Bolivarian Republic of)
    "VN": 49915,    # Viet Nam
    "VG": 285454,   # Virgin Islands (British)
    "VI": 286898,   # Virgin Islands (U.S.)
    "WF": 3412448,  # Wallis and Futuna
    "EH": 2559126,  # Western Sahara
    "YE": 305092,   # Yemen
    "ZM": 195271,   # Zambia
    "ZW": 195272    # Zimbabwe
}

def download_osm_polygons(target_filename: str) -> None:
    source_url = "https://osm-boundaries.com/Download/Submit?apiKey=0cc5cf14238f15fa2573541ae1cead42&db=osm20230605&minAdminLevel=2&maxAdminLevel=2&format=GeoJSON&srid=4326&landOnly&osmIds="
    # TODO: we could filter by countries of interest here
    country_codes = osm_id.keys()
    # The IDs are negated by this API; I have no idea why and I'm not about to ask.
    osm_ids = {-osm_id[cc]: cc for cc in country_codes}
    source_url += ",".join([str(id) for id in osm_ids.keys()])
    print("Downloading: ", source_url)
    with tempfile.TemporaryDirectory() as tmpdir:
        download_path = os.path.join(tmpdir, "countries.gz")
        response = requests.get(source_url, stream=True, timeout=60)
        if not response.ok:
            raise DownloadError(response.status_code, response.reason, source_url)
        with open(download_path, 'wb') as output_file:
            for chunk in response.iter_content(chunk_size=1024*1024):
                output_file.write(chunk)

        compressed_data = GzipFile(download_path, "rb")
        shape_file_data = geopandas.read_file(compressed_data)
        # Later code expects the country code in this format
        shape_file_data['ISO_A2'] = shape_file_data['osm_id'].apply(lambda osm_id: osm_ids[osm_id])
        shape_file_data.to_file(target_filename, driver='GeoJSON')

def main() -> None:
    try:
        target_filename = sys.argv[1]
    except IndexError:
        print(f"Usage: {sys.argv[0]} OUTPUT_GEOJSON_FILENAME", file=sys.stderr)
        sys.exit(1)

    if not target_filename.endswith('.geojson'):
        print("Expected target filename to end with .geojson", file=sys.stderr)
        sys.exit(1)

    try:
        download_osm_polygons(target_filename)
    except DownloadError as exc:
        print(f"Failed to download file: {exc.msg}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
