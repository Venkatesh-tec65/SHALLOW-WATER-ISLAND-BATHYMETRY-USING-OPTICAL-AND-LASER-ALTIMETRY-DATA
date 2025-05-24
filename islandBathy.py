from osgeo import gdal, ogr
import numpy , pandas, os, glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor

def train_bagging_regression(data):
    """
    Train a Bagging Regressor model to predict 'h_ph' using 'B2', 'B3' columns from the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing columns 'B2', 'B3', and 'h_ph'.

    Returns:
    - model (sklearn.ensemble.BaggingRegressor): Trained Bagging Regressor model.
    - metrics (dict): Performance metrics of the model on the test set.
    """

    # Define columns
    feature_columns = ['B2', 'B3', 'B4', 'B8', 'index1', 'index2', 'index3']
    # feature_columns = ['B2', 'B3', 'B4', 'index1', 'index2']
    target_column = ['Corrected_MSL']

    # Extract features (B2, B3) and target (h_ph)
    X = data[feature_columns]
    y = data[target_column]
    y=y.values.ravel()
    # print(X,y)
    # input()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test =pandas.DataFrame(X_test, columns = X_train.columns)

    # Initialize the base regressor
    base_regressor = LinearRegression()

    # Initialize the Bagging Regressor with the base regressor
    model = BaggingRegressor(estimator=base_regressor, n_estimators=100)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print(y_test,y_pred)

    metrics = {
        'Mean Squared Error': mse,
        'R-squared': r2
    }

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    # import matplotlib.pyplot as plt

    # coefs = numpy.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # print(coefs)
    # coefs.plot(kind="barh", figsize=(9, 7))
    # plt.title("Ridge model")
    # plt.axvline(x=0, color=".5")
    # plt.subplots_adjust(left=0.3)

    return model, metrics

def train_ann_regression(data):
    """
    Train an Artificial Neural Network (ANN) model to predict 'h_ph' using 'B2', 'B3' columns from the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing columns 'B2', 'B3', and 'h_ph'.

    Returns:
    - model (sklearn.neural_network.MLPRegressor): Trained ANN model.
    - metrics (dict): Performance metrics of the model on the test set.
    """

    # Define columns
    feature_columns = ['B2', 'B3', 'B4', 'B8', 'index1', 'index2', 'index3']
    target_column = 'h_ph'

    # Extract features (B2, B3) and target (h_ph)
    X = data[feature_columns]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the ANN model
    model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=100000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Mean Squared Error': mse,
        'R-squared': r2
    }

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return model, metrics

def remove_outliers(data, columns):
    """
    Remove outliers from a DataFrame based on the IQR method for specified columns.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - columns (list of str): List of column names to check for outliers.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        print(column, lower_bound,upper_bound)
    return data

def train_linear_regression(data):
    """
    Train a Linear Regression model to predict 'h_ph' using 'B2', 'B3', 'B4', 'B8' columns from the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing columns 'B2', 'B3', 'B4', 'B8', and 'h_ph'.

    Returns:
    - model (sklearn.linear_model.LinearRegression): Trained Linear Regression model.
    - metrics (dict): Performance metrics of the model on the test set.
    """

    # Define columns
    feature_columns = ['B2', 'B3', 'B4', 'B8', 'index1', 'index2', 'index3']
    # feature_columns = ['B2', 'B3']
    target_column = 'h_ph'

    # Remove outliers
    # data = remove_outliers(data, [target_column])

    # Extract features (B2, B3, B4, B8) and target (h_ph)
    X = data[feature_columns]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Mean Squared Error': mse,
        'R-squared': r2
    }

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return model, metrics

def main(csvs,rasterFile):
    # n1 = 1.00029
    # n2 = 1.34116
    ndwiTh = 0.17
    dfs=[]
    for i in range(len(csvs)):
        tempdf = pandas.read_csv(csvs[i])
        if len(tempdf)>0:
            med = tempdf[tempdf['Class']==0]['MSL'].median()
            tempdf['Corrected_MSL'] = med - ((med - tempdf['MSL'])*1.00029/1.34116)
            dfs.append(tempdf)
            # # Separate the data based on class
            # class_0 = tempdf[tempdf['Class'] == 0]
            # class_1 = tempdf[tempdf['Class'] == 1]

            # # Create a scatter plot
            # plt.figure(figsize=(10, 6))

            # # Plot distance vs. h_ph for Class 0
            # plt.scatter(class_0['distance'], class_0['h_ph'], color='blue', label='Ocean Surface', marker='o',s=25, alpha=0.3)

            # # Plot distance vs. h_ph for Class 1
            # plt.scatter(class_1['distance'], class_1['h_ph'], color='red', label='Apparant Bed Elevation', marker='o',s=25, alpha=0.3)

            # # Plot distance vs. Corrected_Elevation for Class 1
            # plt.scatter(class_1['distance'], class_1['Corrected_Elevation'], color='green', label='Corrected Bed Elevation', marker='x',s=25, alpha=0.3)

            # # Add labels and legend
            # plt.xlabel('Distance')
            # plt.ylabel('h_ph / Corrected Elevation')
            # plt.title('Scatter Plot of Distance vs. h_ph and Corrected Elevation')
            # plt.legend()
            # plt.grid(True)

            # # Show the plot
            # plt.show()
        
        #print(tempdf)
        #input()
    df=pandas.concat(dfs)
    dfs=None
    df=df.replace(-9999,numpy.nan).dropna()
    raster=gdal.Open(rasterFile)
    rasterGTs=raster.GetGeoTransform()
    # validPoints = df[df["Class"]==1]
    data=[]
    rasterArray=raster.ReadAsArray()/10000
    columnNames=df.columns.tolist()
    additionalColumns = ['B2','B3','B4','B8']
    columnNames+=additionalColumns
    for i in range(0,df.shape[0]):
        # print(df,list(df.iloc[i]))
        # input()
        vals = list(df.iloc[i])
        if not vals[11]==1:
            continue
        if vals[9]>0:
            
            c,r = int((vals[4]-rasterGTs[0])/rasterGTs[1]),int((vals[3]-rasterGTs[3])/rasterGTs[5])
            vals+=[rasterArray[1,r,c],rasterArray[2,r,c],rasterArray[3,r,c],rasterArray[7,r,c]]
        else:
            vals+=[numpy.nan]*len(additionalColumns)
            # print(rasterArray[1,r,c],rasterArray[2,r,c],rasterArray[3,r,c],rasterArray[7,r,c])
        # print(vals,r,c,(vals[0]-rasterGTs[0])/rasterGTs[1],(vals[1]-rasterGTs[3])/rasterGTs[5],rasterArray[1,r,c],rasterArray[2,r,c],rasterArray[3,r,c],rasterArray[7,r,c])
        data.append(vals)
    # print(data)
    df=pandas.DataFrame(data=data,columns=columnNames)
    
    df['index1']=(df['B2']-df['B8'])/(df['B2'] + df['B8'] + 0.0000001)
    df['index2']=(df['B3']-df['B8'])/(df['B3'] + df['B8'] + 0.0000001)
    df['index3']=(df['B4']-df['B8'])/(df['B4'] + df['B8'] + 0.0000001)
    df.to_csv(r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\data_new.csv")
    df=df[df['index2']>=ndwiTh-0.02]
    
    # trained_model,metrics = train_linear_regression(df)
    trained_model,metrics = train_bagging_regression(df)
    # df.plot('lat_ph',y='h_ph')
    # plt.show()
    
    toPredict = rasterArray.transpose(1,2,0).reshape([rasterArray.shape[1]*rasterArray.shape[2],rasterArray.shape[0]])[:,[1,2,3,7]]
    toPredict =pandas.DataFrame(data=toPredict,columns= ['B2', 'B3', 'B4', 'B8'])
    toPredict['index1']=(toPredict['B2']-toPredict['B8'])/(toPredict['B2'] + toPredict['B8']+0.00000001)
    toPredict['index2']=(toPredict['B3']-toPredict['B8'])/(toPredict['B3'] + toPredict['B8']+0.00000001)
    toPredict['index3']=(toPredict['B4']-toPredict['B8'])/(toPredict['B4'] + toPredict['B8']+0.00000001)
    # Define columns
    feature_columns = ['B2', 'B3', 'B4', 'B8', 'index1', 'index2', 'index3']
    # feature_columns = ['B2', 'B3', 'B4', 'index1', 'index2']
    print(toPredict)
    input()
    predictions = trained_model.predict(numpy.array(toPredict[feature_columns]))
    predictions[toPredict['index2']<ndwiTh]=-9999
    predictions = predictions.reshape(rasterArray.shape[1],rasterArray.shape[2])
    
    print(predictions)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\bitra_30(new17).tif", predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(predictions)
    out_ds.SetGeoTransform(rasterGTs)
    out_ds.SetProjection(raster.GetProjection())
    out_band.SetNoDataValue(-9999)
    # toPredict=numpy.array([rasterArray[1,:,:],rasterArray[2,:,:],rasterArray[3,:,:],rasterArray[7,:,:],rasterArray[1,:,:]/rasterArray[7,:,:],rasterArray[2,:,:]/rasterArray[7,:,:],rasterArray[3,:,:]/rasterArray[7,:,:]])
    # print(toPredict[:,0, 0])
    

if __name__=="__main__":
    #AllCSVs for an island
    inputCSVs = glob.glob(r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\_20*.csv")
    if inputCSVs:
        print(f"Files found: {inputCSVs}")
    else:
        print("No files found matching the pattern. Please check the file path and pattern.")
    # inputCSVs=["/media/bharath/DATA/TDP/TDP-SurfaceWaterTrends/data/islands/kadamat/validPoints.csv"]
    #S2RasterFile
    inputRaster=(r"C:\Users\venki\OneDrive\Desktop\ISRO\Islands_new\islands\Bitra_01\bitra_30.tif")
    main(inputCSVs,inputRaster) 