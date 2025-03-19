"""
https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view
"""

import requests
          
dataset_id = "d_4a086da0a5553be1d89383cd90d07ecd"
url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/poll-download"
        
response = requests.get(url)
json_data = response.json()
if json_data['code'] != 0:
    print(json_data['errMsg'])
    exit(1)

url = json_data['data']['url']
response = requests.get(url)
print(response.text)


#
import json
import pandas as pd
import re

# Load the GEOJSON file (replace with your file path if needed)
geojson_file = "hawker_centres.geojson"

# Read the GEOJSON file
with open(geojson_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract features
features = data["features"]

# Initialize a list to store extracted data
hawker_centres = []

# Function to extract postal code using regex (since it's inside a long text)
def extract_postal_code(description):
    match = re.search(r"ADDRESSPOSTALCODE</th>\s*<td>(\d+)</td>", description)
    return match.group(1) if match else None

# Iterate over each feature
for feature in features:
    props = feature["properties"]
    geom = feature["geometry"]

    hawker_name = props.get("NAME", "Unknown")  # Hawker Centre Name
    street_name = props.get("ADDRESSSTREETNAME", "Unknown")  # Street Name
    postal_code = extract_postal_code(props["Description"])  # Extract postal code
    latitude, longitude = geom["coordinates"][:2]  # Get lat/lon

    hawker_centres.append({
        "Hawker Centre": hawker_name,
        "Street Name": street_name,
        "Postal Code": postal_code,
        "Latitude": latitude,
        "Longitude": longitude
    })

# Convert to DataFrame
df = pd.DataFrame(hawker_centres)

# Save to CSV for easy use
csv_filename = "hawker_centres_singapore.csv"
df.to_csv(csv_filename, index=False)

# Display first few rows
print("✅ Hawker Centre Data Extracted!")
print(df.head())