import os
import ee
import numpy as np
import pandas as pd
ee.Initialize()

# Functions

def filter_date(product,year):
    startdate = ee.Date.fromYMD(year,1,1)
    enddate =ee.Date.fromYMD(year+1,1,1)
    prod = product.filterDate(startdate, enddate).sort('system:time_start', False).select("precipitation")
    return prod

def calcMean(imageCollection,year):
	mylist = ee.List([])
	months = range(1,13) # Before and during the growing season
	for m in months:
		w = imageCollection.filter(ee.Filter.calendarRange(year, year, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).sum();
		mylist = mylist.add(w.set('year', year).set('month', m).set('date', ee.Date.fromYMD(year,m,1)).set('system:time_start',ee.Date.fromYMD(year,m,1)))
	return ee.ImageCollection.fromImages(mylist)

def aggregate(product,year,area):

    # Filter
    filtered = filter_date(product, year)

    # calculate the monthly means
    seasonal = ee.ImageCollection(calcMean(filtered, year))

    # select the region of interest, 500 is the cellsize in meters
    yearly = seasonal.getRegion(area,25000,"epsg:4326").getInfo()

    return yearly 

def make_df_from_imcol(imcol):
    df = pd.DataFrame(imcol, columns = imcol[0])
    df = df[1:]
    
    lons = np.array(df.longitude)
    lats = np.array(df.latitude)
    data = np.array(df.NDVI)
    
    return lons, lats, data
    return df

def df_from_ee_object(imcol):
    df = pd.DataFrame(imcol, columns = imcol[0])
    df = df[1:]
    return(df)


def process_county(county_number, product):

	area = (ee.FeatureCollection('ft:15cgNIs0G_vYtNUhpYVmz0u_sPs6XrIUYpi19pjkt')
       .filter(ee.Filter().eq('id', county_number)))

	# import the RS products
	imcol = ee.ImageCollection(product)
	 
	# Define time range
	years = [x for x in range(2000, 2016)]

	# Aggregate
	aggregated= []
	for year in years:
		aggregated.append(aggregate(imcol,year,area))

	# make dfs from image cols 
	dfs = []
	for i in aggregated:
	    dfs.append(df_from_ee_object(i))

	for df in dfs:
		df.drop(["time"], axis=1, inplace = True)

	# Zip the years and dfs 
	yearly = dict(zip(years,dfs))

	# Make the out directories for modis data and sub directories for counties 
	outpath = os.path.join(os.getcwd(), product.replace("/","_"))
	if os.path.exists(outpath):
		pass
	else:
		os.mkdir(outpath)

	county_dir = os.path.join(outpath,str(county_number).zfill(3))
	if os.path.exists(county_dir):
		pass
	else:
		os.mkdir(county_dir)

	for k,v in yearly.items():
	    v.to_csv(os.path.join(county_dir,str(k)+".csv"))

def main():
	products = ['UCSB-CHG/CHIRPS/PENTAD','TRMM/3B42']
	county_codes = [i for i in range(1, 117, 2)]
	for i in county_codes:
		print("Processing County Number " + str(i))
		try:
			process_county(i, products[1] )
		except:
			print(str(i) + " FAILED ==============")

if __name__ == '__main__':
	main()

