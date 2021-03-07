import requests, shutil
import csv
import sys
import subprocess as sp
from bs4 import BeautifulSoup as bs

api_key = '***'

csv_file_path = './population_kor_2018-10-01.csv'
img_path = './img/'

with open(csv_file_path, newline='') as csvfile:
    print('file opened: ' + csv_file_path)
    reader = csv.reader(csvfile, delimiter=',')
    print('csv read: ' + str(reader))
    i = 0
    for row in reader:
        if i % 300 == 0 and i >= 18000 and i <= 8800000:
            print(str(i) + ' ' + str(row))
            url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + row[0] + ',' + row[1] +'&zoom=18&size=512x512&maptype=satellite&key=' + api_key
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(img_path + str(i//100) + ',' + row[0] + ', ' + row[1] + ', ' + row[3] + '.png', 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            else:
                print('code: ' + str(response.status_code))
        i += 1
    print('end of program.')
