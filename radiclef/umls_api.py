from radiclef import RESOURCES_DIR

import requests
import os
import json

API_KEY: str = ""

AUTH_ENDPOINT: str = "https://utslogin.nlm.nih.gov"
SERVICE: str = "http://umlsks.nlm.nih.gov"

CONCEPT_MAP_PATH = os.path.join(RESOURCES_DIR, "concept-map.json")


def get_tgt(api_key: str) -> str:
    """Get Ticket Granting Ticket (TGT) from UMLS API."""
    response = requests.post(f"{AUTH_ENDPOINT}/cas/v1/api-key", data={'apikey': api_key})
    if response.status_code == 201 and 'location' in response.headers:
        return response.headers['location']
    raise Exception(f"Failed to get TGT: {response.status_code} - {response.text}")


def get_service_ticket(tgt: str) -> str:
    """Use TGT to get a Service Ticket."""
    response = requests.post(tgt, data={'service': SERVICE})
    if response.status_code == 200:
        return response.text
    raise Exception(f"Failed to get service ticket: {response.status_code} - {response.text}")


def fetch_canonical_name(cui: str, tgt: str) -> str | None:
    """Fetch canonical (preferred) name for a given CUI."""
    try:
        ticket: str = get_service_ticket(tgt)
        url: str = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
        response = requests.get(url, params={'ticket': ticket})
        if response.status_code == 200:
            data = response.json()
            return data.get('result', {}).get('name', None)
        else:
            print(f"Error fetching CUI {cui}: {response.status_code}")
    except Exception as e:
        print(f"Exception fetching CUI {cui}: {e}")
    return None


if __name__ == "__main__":

    if not os.path.exists(CONCEPT_MAP_PATH):
        with open(os.path.join(RESOURCES_DIR, "cui-alphabet.txt"), "r") as f:
            cui_mappings = {_line.strip(): None for _line in f.readlines()[4:] if _line.startswith("C")}

        tgt = get_tgt(API_KEY)
        for _cui in cui_mappings.keys():
            _name = fetch_canonical_name(_cui, tgt)
            print("{}: {}".format(_cui, _name))
            cui_mappings[_cui] = _name

        with open(CONCEPT_MAP_PATH, "w") as f:
            json.dump(cui_mappings, f, indent=2)
