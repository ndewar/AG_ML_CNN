import geopandas as gp
import pandas as pd
import os
import urllib
import lxml.html
import shutil

# Fetch the yield CSV data

def download_csv(f):
    fn = os.path.split(f)[1][0:4]
    os.system('curl -o ca_{}.csv {}'.format(fn,f))

def main():

	base_url = 'https://www.nass.usda.gov'
	connection = urllib.urlopen('https://www.nass.usda.gov/Statistics_by_State/California/Publications/AgComm/Detail/index.php')

	dom =  lxml.html.fromstring(connection.read())
	links = []
	for link in dom.xpath('//a/@href'): # select the url in href for all a tags(links)
	    links.append(link)

	foi = [x for x in links if x.endswith(".csv")]
	urls = [(base_url+x) for x in foi]

	csv_dir = os.path.join(os.getcwd(),"csvs")

	for i in urls:
	    download_csv(i)

	if os.path.exists(os.path.join(os.getcwd(),"csvs")):
		pass
	else:
		os.mkdir(csv_dir)

	csvs = [os.path.join(x) for x in os.listdir(os.getcwd()) if x.endswith(".csv")]
	for i in csvs:
		shutil.move(i,csv_dir)

if __name__ == '__main__':
	main()