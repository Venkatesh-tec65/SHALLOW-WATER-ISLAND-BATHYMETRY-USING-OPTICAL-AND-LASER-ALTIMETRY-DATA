import os, glob
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import pygeodesy
from pygeodesy.ellipsoidalKarney import LatLon

def geoidmodel(df):
    """
    Create an interpolation function for geoid values based on latitude and longitude.

    Parameters:
    df (DataFrame): A pandas DataFrame containing 'reference_photon_lon', 'reference_photon_lat', and 'geoid' columns.

    Returns:
    function: An interpolation function that takes longitude and latitude as inputs and returns the interpolated geoid value.
    """
    lon = df['reference_photon_lon'].values
    lat = df['reference_photon_lat'].values
    geoid = df['geoid'].values

    def interpolation_function(lon_val, lat_val):
        points = np.column_stack((lon, lat))
        return griddata(points, geoid, (lon_val, lat_val), method='linear')
    
    return interpolation_function
def getFiles(folderName,pattern):
    files=glob.glob(os.path.join(folderName,pattern),recursive=True)
    return files

def interpolate_geoid_column(df, interp_func):
    df['interpolated_geoid'] = df.apply(lambda row: interp_func((row['lat_ph'], row['lon_ph']))[0], axis=1)
    return df



def main(dataLoc,geoidLoc):
    ginterpolator = pygeodesy.GeoidKarney(geoidLoc)
    all_files_atl08 = getFiles(dataLoc,'*ATL03*ascii')
    df1, geoiddf1 = None,None
    # Iterate through ASCII tables: each table represents one groundtrack
    for k in range(len(all_files_atl08)):
        
        fn=all_files_atl08[k]
        print(fn)
        gtx = os.path.basename(fn).split('.')[0][-2:]
        with open(fn,'r') as f:
            data=f.read()
            #data=data.split('/gt1r/heights/delta_time /gt1r/heights/h_ph /gt1r/heights/lat_ph /gt1r/heights/lon_ph\n')
        data=data.split("\n\n")
        print(data)
        geoidData = data[0]
        geoidData=geoidData.split('\n')
        for j in range(1,len(geoidData)):
            # print("Raw:" , data[j])
            geoidData[j] = geoidData[j].split("  ")
            geoidData[j] = [x.strip() for x in geoidData[j] if x !=""]
        geoidData[0] = [x.strip().split("/")[-1] for x in geoidData[0].split(" ") if x !=""]
        tempgeoiddf1=pd.DataFrame(geoidData[1:],columns=geoidData[0])
        if len(tempgeoiddf1[tempgeoiddf1['geoid']!='<Fill_Value>'])==0:
            outCols = [(row['reference_photon_lat'], row['reference_photon_lon']) for _, row in tempgeoiddf1.iterrows()]
            outCols = [ginterpolator(LatLon(lat, lon)) for lat,lon in outCols]
            tempgeoiddf1['geoid']=outCols
        tempgeoiddf1 = tempgeoiddf1[['delta_time','reference_photon_lat', 'reference_photon_lon', 'geoid']]
        tempgeoiddf1.replace('<Fill_Value>', np.nan, inplace=True)
        tempgeoiddf1 = tempgeoiddf1.dropna()
        
        # print(tempgeoiddf1)
        geoiddf1 = pd.concat([geoiddf1,tempgeoiddf1], ignore_index=True)
        print(geoiddf1)
        X_train, X_test, y_train, y_test = train_test_split(geoiddf1[['reference_photon_lat', 'reference_photon_lon']], geoiddf1['geoid'], test_size=0.05)
        # print(geoiddf1)
        # Train the regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        data=data[1].split('\n')
        
        for j in range(1,len(data)):
            # print("Raw:" , data[j])
            data[j] = data[j].split("  ")
            data[j] = [x.strip() for x in data[j] if x !=""]
        data[0] = [x.strip().split("/")[-1] for x in data[0].split(" ") if x !=""]
        data[1:]=[x for x in data[1:] if len(x)==len(data[0])]
        # columns = ['datetime','h_ph','lat','lon']
        # print(data)
        tempdf=pd.DataFrame(data[1:],columns=data[0])
        tempdf.replace('<Fill_Value>', np.nan, inplace=True)
        if 'lat_ph' in tempdf.columns and 'lon_ph' in tempdf.columns:
            tempdf_renamed = tempdf.rename(columns={'lat_ph': 'reference_photon_lat', 'lon_ph': 'reference_photon_lon'})
            
            tempdf['geoid'] = model.predict(tempdf_renamed[['reference_photon_lat','reference_photon_lon']])
            # print(tempdf['h_ph'],tempdf['geoid'])
            tempdf['MSL'] = tempdf['h_ph'].astype(float) - tempdf['geoid']
            tempdf['gtx'] = gtx
            # print(tempdf)
            # input()
            df1 = pd.concat([df1,tempdf], ignore_index=True)  
        
        
    
    print(df1)
    df1.to_csv(os.path.join(dataLoc, f"data.csv"), index=False)
    geoiddf1.to_csv(os.path.join(dataLoc, f"geoid_data.csv"), index=False)
    return(df1)
if __name__=="__main__":
    if len(os.sys.argv) == 1:
        dataLoc = r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Andrott_002"
        geoidLoc=r"C:\Users\venki\OneDrive\Desktop\ISRO\geoids_EGM2008 - 1'\EGM2008 - 1'.pgm"
    main(dataLoc,geoidLoc)