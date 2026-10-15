
import requests
from huggingface_hub import get_token

url = "https://datasets-server.huggingface.co/filter"

params = {
    "dataset": "ai4bharat/IndicVoices",
    "config": "hindi",
    "split": "valid",
    "where": '"state"=\'Madhya Pradesh\'',
    "offset": 0,
    "length": 5,
}

token = get_token()
headers = {"Authorization": f"Bearer {token}"} if token else {}

try:
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=45,
    )

    print("HTTP status:", response.status_code)

    data = response.json()

    if response.status_code == 200:
        print("Dataset:", data.get("dataset"))
        print("Split:", data.get("split"))
        print("Total matching rows:", data.get("num_rows_total"))
        print("Partial result:", data.get("partial"))
        print("Returned states:")

        for item in data.get("rows", []):
            print(item["row"].get("state"))
    else:
        print("Response:", data)

except requests.RequestException as exc:
    print("Network error:", exc)
except ValueError:
    print("Response was not valid JSON.")
    print(response.text[:1000])
