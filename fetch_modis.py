import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gp

"""
Downloads the NASA MOD13Q1 8-day NDVI product summed annually over each county in california at 500m resolution. 
"""

def filter_date_modis(product,year):
    startyear = ee.Date.fromYMD(year,1,1)
    endyear =ee.Date.fromYMD(year+1,1,1)
    prod = product.filterDate(startyear, endyear).sort('system:time_start', False).select("NDVI")
    return prod

def aggregate(product,year,area):

    # Filter
    filtered = filter_date_modis(product, year)

    # calculate the monthly mean
    def calcMean(imageCollection,year):
        mylist = ee.List([])
        months = range(2,13) # Growing Season 
        for m in months:
                w = imageCollection.filter(ee.Filter.calendarRange(year, year, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).mean();
                mylist = mylist.add(w.set('year', year).set('month', m).set('date', ee.Date.fromYMD(year,m,1)).set('system:time_start',ee.Date.fromYMD(year,m,1)))
        return ee.ImageCollection.fromImages(mylist)

    # run the calcMonthlyMean function
    seasonal = ee.ImageCollection(calcMean(filtered, year))

    # select the region of interest, 500 is the cellsize in meters
    seasonal = seasonal.getRegion(area,500,"epsg:4326").getInfo()

    return seasonal 

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
	ee.Initialize()

	area = (ee.FeatureCollection('ft:1QPasan0i6O9uUlcYkjqj91D7mbnhTZCmzS4t7t_g')
	      .filter(ee.Filter().eq('id', county_number)))

	# import the RS products
	modis = ee.ImageCollection(product) #
	#modis = ee.ImageCollection('MODIS/MOD13Q1') #MCD43A4_NDVI
	 
	# Define time range
	years = [x for x in range(2000, 2016)]
	aggregated = []

	# Aggregate
	for year in years:
	    aggregated.append(aggregate(modis,year,area))

	# Make dataframes
	fin_dfs = []
	for i in aggregated:
	    fin_dfs.append(df_from_ee_object(i))

	for df in fin_dfs:
		df.drop(['time'], axis=1, inplace = True) #(['id',"longitude","latitude","time"], axis=1, inplace = True)
	    
	# Make a dict of the years with the data frames
	dfs_by_year = dict(zip(years,fin_dfs))

	# Make the out directories for modis data and sub directories for counties 
	outpath = os.path.join(os.getcwd(), product.replace("/","_"))
	if os.path.exists(outpath):
		pass
	else:
		os.mkdir(outpath)

	county_dir = os.path.join(outpath,county_number)
	if os.path.exists(county_dir):
		pass
	else:
		os.mkdir(county_dir)

	for k,v in dfs_by_year.items():

	    v.to_csv(os.path.join(county_dir,str(k)+".csv"))

def main():
	county_codes = [str(i).zfill(3) for i in range(1, 117, 2)]
	product = 'MCD43A4_NDVI'
	for i in county_codes:
		print("Processing County Number " + str(i))
		try:
			process_county(i, product)
		except:
			print(i + " FAILED ==============")

if __name__ == "__main__":
	main()
