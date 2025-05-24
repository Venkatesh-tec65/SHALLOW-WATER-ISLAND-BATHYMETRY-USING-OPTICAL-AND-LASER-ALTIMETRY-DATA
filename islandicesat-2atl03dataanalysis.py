import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from classesForIslandDataProcessing import Interactive2DScatter, Interactive3DScatter, InteractiveDBSCAN2D,InteractiveDBSCAN3D, InteractiveDataFramePlot
from matplotlib.widgets import TextBox
import os

def denseregions(x,z):
    # Combine x, y, z into a numpy array
    data = np.column_stack([x,z])

    # Normalize data (optional but recommended)
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)

    # Apply DBSCAN
    eps = 0.009  # distance threshold
    min_samples = 15  # minimum number of points in a cluster
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(normalized_data)

    # Retain points in dense clusters (clusters with label >= 0 are valid clusters)
    dense_points = data[clusters >= 0]

    # Plot original data and dense points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c='b', marker='.', label='Original Points', alpha=0.5)
    ax.scatter(dense_points[:, 0], dense_points[:, 1], c='r', marker='o', s=50, label='Dense Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_title('DBSCAN Clustering of 3D Points')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return dense_points
def dbscanforseafloor(df):
    # Assuming your dataframe is named df
    # Extract the relevant column
    X = df[['h_ph']]

    # Standardize the data (optional but recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.01, min_samples=25)  # You may need to adjust the eps and min_samples parameters
    dbscan.fit(X_scaled)

    # Add the cluster labels to the dataframe
    df['cluster'] = dbscan.labels_

    # Print the number of points in each cluster
    print(df['cluster'].value_counts())

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df['h_ph'], c=df['cluster'], cmap='plasma', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Height (h_ph)')
    plt.title('DBSCAN Clustering Based on h_ph')
    plt.colorbar(label='Cluster Label')
    plt.show()
def main():
    df = pd.read_csv(r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\data.csv")
    df['delta_time']=pd.to_datetime(df['delta_time'])
    df['date'] = df['delta_time'].dt.date 
    # dt=[['09-06-2019','3l'],['03-06-2022','3l'],['09-08-2020','3l'],['12-06-2021','3l'],['06-11-2019','3l'],['06-05-2022','3l'],['06-09-2020','3l'],['02-10-2020','3l'],['09-10-2019','3l'],[]
    boundsToGetPasses = "72.13008,11.494966,72.192133,11.609366"
    boundsToGetPasses = [float(x) for x in boundsToGetPasses.split(",")]
    dts=df[(df['lon_ph']>boundsToGetPasses[0])&(df['lon_ph']<boundsToGetPasses[2])&(df['lat_ph']>boundsToGetPasses[1])&(df['lat_ph']<boundsToGetPasses[3])][['date','gtx']]
    # print(df[(df['lon_ph']>boundsToGetPasses[0])&(df['lon_ph']<boundsToGetPasses[2])&(df['lat_ph']>boundsToGetPasses[1])&(df['lat_ph']<boundsToGetPasses[3])])
    # dts['delta_time']= dts['delta_time'].dt.date 
    dts = dts.drop_duplicates()
    print(dts)
    for i in range(len(dts)):
        print("{}/{} dates".format(i+1,len(dts)))
        eachDt = dts.iloc[[i]]
        csvName = r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\_{}_{}.csv".format(np.array(eachDt)[0][0], np.array(eachDt)[0][1])
        filteredDf = df.merge(eachDt, on=['date', 'gtx'], how='inner')
        if os.path.exists(csvName):
            continue
        # print(filteredDf)
        # print(df)
        # # os.sys.exit()
        # print(filteredDf)
        # print(filteredDf['lon_ph'].describe())
        ####################### Testing with image generation
        interactive_plot = InteractiveDataFramePlot(filteredDf, 'lon_ph', 'lat_ph')
        clicked_points = interactive_plot.get_clicked_points()
        lonMin,latMin = clicked_points[-1]
        filteredDf['distance'] = ((filteredDf['lon_ph']-lonMin)**2+(filteredDf['lat_ph']-latMin)**2)**(1/2)
        # input()
        # filteredDf.plot.scatter(x='distance',y='h_ph', marker='o')
        interactive_dbscan_2d = InteractiveDBSCAN2D(filteredDf)
        plt.show()
        filteredDf = interactive_dbscan_2d.get_clustered_points_df()
        validPointsDf = filteredDf[filteredDf['clustered_points']==1]
        validPointsDf["Class"]=None
        interactive_plot = Interactive2DScatter(validPointsDf)
        plt.show()
        oceanHt,bedHt = interactive_plot.get_final_params()
        # print(filteredDf['distance'].describe())
        # input()
        # continue








        ########################
        # interactive_dbscan_3d = InteractiveDBSCAN3D(filteredDf)
        # plt.show()
        # validPoints = interactive_dbscan_3d.get_clustered_points()
        # # validPoints = denseregions(filteredDf['lon_ph'], filteredDf['lat_ph'], filteredDf['h_ph'])
        # validPointsDf=pd.DataFrame(validPoints,columns = ['lon_ph','lat_ph','h_ph'])
        # validPointsDf["Class"]=None
        # interactive_plot = Interactive3DScatter(validPointsDf)
        # plt.show()
        # oceanHt,bedHt = interactive_plot.get_final_params()
        # # oceanHt = float(input("Enter the approximate elevation of ocean water level"))
        # # bedHt = float(input("Enter the approximate bed height"))
        validPointsDf["FromOceanSurface"]=abs(validPointsDf["MSL"]-oceanHt)
        validPointsDf["FromBed"]=abs(validPointsDf["MSL"]-bedHt)
        validPointsDf["Class"]=(validPointsDf['FromOceanSurface'] > validPointsDf["FromBed"]).astype(int)
        print(eachDt.date,eachDt.gtx,np.array(eachDt))
        validPointsDf.to_csv(csvName)
        print(validPointsDf.describe())
        # Plotting 3D scatter plot
        # Define colors based on 'new_column'
        colors = validPointsDf['Class'].map({0: 'blue', 1: 'red'})

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        sc = ax.scatter(validPointsDf['lon_ph'], validPointsDf['lat_ph'], validPointsDf['MSL'], c=colors)#, marker='o')
        # ax.scatter(validPointsDf['lon_ph'], validPointsDf['lat_ph'], validPointsDf['h_ph'], c=colors, marker='o')

        # Set labels and title
        ax.set_ylabel('Latitude (lat_ph)')
        ax.set_xlabel('Longitude (lon_ph)')
        ax.set_zlabel('Height (MSL)')
        ax.set_title('3D Scatter Plot of lat_ph, lon_ph, and MSL')
        # Display the plot
        plt.tight_layout()
        plt.show()
        # dbscanforseafloor(validPointsDf)
        # print(validPointsDf)

if __name__=='__main__':
    main()