import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# EE / Numpy functions

def filter_date_modis(product,year):
	startyear = ee.Date.fromYMD(year,1,1)
	endyear =ee.Date.fromYMD(year+1,1,1)
	prod = product.filterDate(startyear, endyear).sort('system:time_start', False).select("NDVI")
	return prod

def aggregate(product,year,area):

	# Filter
	filtered = filter_date_modis(product, year)
	
	# bbox 
	bounds = area.geometry().bounds()

	# calculate the monthly mean
	def calcMean(imageCollection,year):
		mylist = ee.List([])
		months = range(2,9)
		for m in months:
			w = imageCollection.filter(ee.Filter.calendarRange(year, year, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).mean();
			clipped = w.clip(area)
			mask = clipped.mask()
			masked = ee.Image(0).where(mask, clipped)
			fin = masked.clip(bounds)
			mylist = mylist.add(fin.set('year', year).set('month', m).set('date', ee.Date.fromYMD(year,m,1)).set('system:time_start',ee.Date.fromYMD(year,m,1)))
		return ee.ImageCollection.fromImages(mylist)

	# run the calcMonthlyMean function
	seasonal = ee.ImageCollection(calcMean(filtered, year))
	
	# select the region of interest, 500 is the cellsize in meters
	seasonal = seasonal.getRegion(bounds,500,"epsg:4326").getInfo()

	return seasonal 

def make_df_from_imcol(imcol):
	df = pd.DataFrame(imcol, columns = imcol[0])
	df = df[1:]
	
	lons = np.array(df.longitude)
	lats = np.array(df.latitude)
	data = np.array(df.constant)
	
	return lons, lats, data

def df_from_ee_object(imcol):
	df = pd.DataFrame(imcol, columns = imcol[0])
	df = df[1:]
	return(df)

def array_from_df(df, variable):
	df = df[df.id == "6"] # TODO : average on 0-6
	
	# get data from df as arrays
	lons = np.array(df.longitude)
	lats = np.array(df.latitude)
	data = np.array(df[variable]) # Set var here 
	  
	# get the unique coordinates
	uniqueLats = np.unique(lats)
	uniqueLons = np.unique(lons)

	# get number of columns and rows from coordinates
	ncols = len(uniqueLons)	
	nrows = len(uniqueLats)

	# determine pixelsizes
	ys = uniqueLats[1] - uniqueLats[0] 
	xs = uniqueLons[1] - uniqueLons[0]

	# create an array with dimensions of image
	arr = np.zeros([nrows, ncols], np.float32)

	# fill the array with values
	counter =0
	for y in range(0,len(arr),1):
		for x in range(0,len(arr[0]),1):
			if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
				counter+=1
				arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
	
	return arr

# ML functions

def baseline_model(X_train): # TODO: Generalize the matrices to fit other counties
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(Conv2D(64, (20, 20), input_shape=(1, X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5)))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model
	
def main(county_number):
	
	if county_number is type(str):
		pass
	else:
		county_number = str(county_number)

	print("Processing County Number: " + county_number)
	
	ee.Initialize()

	area = (ee.FeatureCollection('ft:1QPasan0i6O9uUlcYkjqj91D7mbnhTZCmzS4t7t_g').filter(ee.Filter().eq('id', county_number)))

	# import the RS products

	modis = ee.ImageCollection('MCD43A4_NDVI') #

	years = [x for x in range(2000, 2016)]

	print("Fetching MODIS ========")
	finals_modis = []
	for year in years:
		finals_modis.append(aggregate(modis,year,area))
		print(str(year) + " complete")

	fin_modis_dfs = []
	for i in finals_modis:
		fin_modis_dfs.append(df_from_ee_object(i))

	arrs = []
	for i in fin_modis_dfs:
		arrs.append(array_from_df(i,"constant"))

	print ("====== COMPLETE ======")

	# Grab the training data 
	cwd = os.getcwd()
	y_dir = [os.path.join(cwd,x) for x in os.listdir(os.getcwd()) if "yields" in x][0]
	fn = [os.path.join(y_dir,x) for x in os.listdir(y_dir) if "107" in x][0]

	d = json.load(open(fn))
	yrs = [str(x) for x in years]
	training_dict = { year: d[year] for year in yrs }
	training = training_dict.values()

	labels = training_dict.copy()
	training = dict(zip(yrs,arrs))

	print ("Running Noodle Nets ======")

	for yr in yrs:
		print("Processing " + yr)
		x_train = [v for v in training.keys() if v!=yr]
		x_train = {key:training[key] for key in x_train}
		x_train = x_train.values()
		x_test = training[yr]
		y_train = [v for v in labels.keys() if v!=yr]
		y_train = {key:labels[key] for key in y_train}
		y_train = y_train.values()
		y_test = labels[yr]

		# Stack
		y_test = [y_test]
		y_test = np.array(y_test)
		x_train = np.stack(x_train)
		x_test = np.stack(x_test)
		y_train = np.stack(y_train)
		y_test = np.stack(y_test)

		#Tile 
		x_train = np.tile(x_train,(10,1,1))
		y_train = np.tile(y_train,(10))

		# Reshape
		X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]).astype('float32')
		X_test = x_test.reshape(1, 1, x_test.shape[0], x_test.shape[1]).astype('float32')

		# Run the noodle net
		model = baseline_model(X_train)
		model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=15, verbose=2)
		error=100*np.abs((model.predict(X_test)-y_test)/y_test)
		print("CNN Error for {} = {}".format(yr,error[0]))

		# Make the out dir and write files
		cwd = os.getcwd()
		outdir = os.path.join(cwd,"cnn_results")
		if os.path.exists(outdir):
			continue
		else:
			os.mkdir(outdir)

		county_dir = os.path.join(outdir,county_number)
		if os.path.exists(county_dir):
			continue
		else:
			os.mkdir(county_dir)

		with open(os.path.join(county_dir,'{}_summary.txt'.format(yr)),'w') as fh:
			model.summary(print_fn=lambda x: fh.write(x + '\n'))

		result = {}
		result['year'] = str(yr)
		result['predicted'] = str(model.predict(X_test)[0][0])
		result['actual'] = str(y_test[0])
		result['error'] = str(error[0][0])

		with open(os.path.join(county_dir,'{}_results.json'.format(yr)), 'w') as fp:
			json.dump(result, fp)

if __name__ == "__main__":
	main("033")