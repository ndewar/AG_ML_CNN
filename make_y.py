import geopandas as gp
import pandas as pd
import os
import numpy as np
import json

# Functions

def format_df_from_csv(df): # Takes df read from raw USDA CSV and kills off whitespace and other bs 
    df.columns = [t.strip() for t in list(df)]
    df.columns = df.columns.str.replace('\s+','')  # remove whitespace from headers
    df.columns = df.columns.str.lower()
    df['countycode'] = df['countycode'].apply(lambda x: str(x).zfill(3)) # Zpad the county codes 
    df.county = df.county.str.strip()
    if "commodity" in df.columns:
        df.rename(columns={'commodity': 'commoditycode'}, inplace = True)
    return df

def filter_by_county(df,county):
    if county is type(int):
        g = df[df.countycode == county]
    else:
        county = county.lower()
        df.county = df.county.str.replace('\s+','')
        df.county = df.county.str.lower()
        fin_df = df[df.county == county]
    return(fin_df)

def drop_crops(df):
    allcats = df.cropname
    badfields = ["HORSE","WOOL", "MILK", "LIVESTOCK", "CATTLE", "POULTRY", "PIG","MILK", "APIARY", "SERVICE", "BEES", "PASTURE", "SILAGE","SHEEP", "FEED","FLOWERS","NURSERY", "OSTRICH", "TURKEYS", "CHICKENS", "MANURE", "GOATS", "LAMBS", "BIRDS"]
    keep = [x for x in allcats if x.split() not in  badfields]
    result = [r for r in allcats if not any(z in r for z in badfields)]
    df = df.loc[df['cropname'].isin(result)]
    return df

def calc_prod(df):
    totals = df[df.countycode == 999]
    crops = df.cropname.unique()
    cdfs = []
    for c in crops:
        f = df.loc[df.cropname== c]
        
        f.production = pd.to_numeric(f.production, errors='coerce')
        f.value = pd.to_numeric(f.value, errors='coerce')
        f.countycode = pd.to_numeric(f.countycode, errors = "coerce")
        f.loc[f.production.isnull(), 'production'] = f.loc[f.production.isnull()].value.astype(np.float)*f.loc[f.production.isnull()].value.astype(np.float) / totals.loc[totals.cropname==c].value.astype(np.float)

        cdfs.append(f)
    
    fin_df = pd.concat(cdfs)
    return(fin_df)

def sum_production(df):
    result = {}
    year = df.year
    df = df.production.dropna(axis=0, how='all')
    crop_sum = []
    for i in df:
        try:
            crop_sum.append(float(i))
        except:
            continue    
            
    return sum(crop_sum)

def process_county(county):
    fdir = os.path.join(os.getcwd(), "csvs")
    files = [os.path.join(fdir,x) for x in os.listdir(fdir)]
    tables = [pd.read_csv(x) for x in files]
    dfs = []

    for t in tables:
        dfs.append(format_df_from_csv(t))

    by_county = []
    for i in dfs:
        by_county.append(filter_by_county(i,county))

    cleaned = []
    for i in by_county:
        cleaned.append(drop_crops(i))

    # Check for funky commoditites and remove them (some commodities are blank but still have production values)
    for i in cleaned:
        i = i[i.commoditycode > 100000] 

    filled = []
    for i in cleaned:
        filled.append(calc_prod(i))

    y = {}
    for i in filled:
        year = int(i.year.mode())
        year = str(year)
        y[year] = sum_production(i)

    outdir = os.path.join(os.getcwd(),"county_yields")

    if os.path.exists(outdir):
        pass
    else:
        os.mkdir(outdir)

    outfile = os.path.join(outdir,county+'.txt')

    with open(outfile, 'w') as file:
        file.write(json.dumps(y))


def main():
    cwd = os.getcwd()
    shpdir = os.path.join(cwd,"landuse")
    file = [os.path.join(shpdir,x) for x in os.listdir(shpdir) if x.endswith(".shp")][0]
    shp = gp.read_file(file)
    shp.columns=shp.columns.str.lower()
    counties = shp.county.str.lower().unique()
    for i in counties[:12]:
        i = i.replace(" ","")
        print("PROCESSING: " + i)
        try:
            y = process_county(i)
            print("PROCESSED " + i)
        except:
            print(i + " FAILED TO PROCESS ====================")
            continue

if __name__ == "__main__":
    main()
