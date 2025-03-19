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