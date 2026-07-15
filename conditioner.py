import requests
import os
import json
import csv
import pandas as pd


# set global API info
API_URL = "https://api.collegefootballdata.com/games"
API_KEY = 'NUNYA BUSINESS'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}'
}

# create a variable to store the conditioned data
conditioned_data = []

# present a number of years to grab data from
year_i = 1990
year_f = 2025

# important statistics
categories = ['startDate', 'season', 'seasonType', 'neutralSite' ,'conferenceGame', 'venueId', 'homeTeam', 'homeId', 'homeLineScores', 'awayTeam', 'awayId', 'awayLineScores']

# loop through the important years specified previously
for year in range(year_i, year_f+1):
    os.system('cls')
    print(f'Year: {year}')

    parameter = {
        'year': year
    }

    # get a response from the site regarding a specific year
    response = requests.get(API_URL, headers = HEADERS, params = parameter)
    response_json = response.json()

    # print(response_json)
    # break

    # for each game in the year, grab the important categories
    for item in response_json:
        
        if item['homeClassification'] == 'fbs' and item['awayClassification'] == 'fbs':
            # define a chunk of data which contains the important categories
            package = {}

            for category in categories:

                if isinstance(item[category], list):
                    package[category] = list(item[category])
                else:
                    package[category] = item[category]
            try:
                package['margin'] = item['homePoints'] - item['awayPoints']
            except:
                package['margin'] = 0

            # add the package
            conditioned_data.append(package)

            # print(item['startDate'])

# convert the conditioned data to a CSV stored in the folder
df = pd.DataFrame(conditioned_data)
df.to_csv('games.csv', index=False)
