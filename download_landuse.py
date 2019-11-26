import os
import shutil
import zipfile

def download_url(url):
    fn = os.path.split(url)[1]
    os.system('curl -o ca_{} {}'.format(fn,url))
    return fn


def main():
	shp_url = 'https://gis.water.ca.gov/app/CADWRLandUseViewer/downloads/atlas_i15_CropMapping2014.zip'#https://www.sciencebase.gov/catalog/file/get/592f007de4b0e9bd0ea793c2?f=__disk__58/fd/37/58fd371e1214abb153c4d2e8151c46032b9c9d8a#'#ftp://ftp.wildlife.ca.gov/BDB/GIS/BIOS/Public_Datasets/2600_2699/ds2677.zip'
	fn = download_url(shp_url)
	file = [x for x in os.listdir(os.getcwd()) if x.endswith(fn)][0]

	# unzip it
	zip_ref = zipfile.ZipFile(file, 'r')
	zip_ref.extractall(os.path.join(os.getcwd(),"landuse"))
	zip_ref.close()

	# clean up
	os.remove(file)

main()