# SHALLOW-WATER-ISLAND-BATHYMETRY-USING-OPTICAL-AND-LASER-ALTIMETRY-DATA
This repository provides a modular pipeline for analyzing ICESat-2 ATL03 data to classify ocean surface and bathymetry features around islands using machine learning and remote sensing techniques.

🛰️ Project Overview The workflow includes:

ICESat-2 ATL03 Data Download using NASA EarthData services.

Data Preprocessing & Geoid Correction to compute mean sea level.

Interactive Clustering and Classification of ocean and bed surfaces.

Raster Integration & Feature Extraction using Sentinel-2 multispectral imagery.

Machine Learning Modeling for bathymetry prediction using regression techniques (Linear, Bagging, ANN).

🗂️ Repository Structure Python scrip clone.py - Automated download of ATL03 ICESat-2 data from NASA EarthData.

dataLoader_03.py - Parses and processes ATL03 ASCII data, applies geoid correction, and computes MSL.

classesForIslandDataProcessing.py - Contains interactive classes for data visualization and clustering (2D/3D scatter, DBSCAN).

islandicesat-2atl03dataanalysis.py - Interactive classification and data preparation for modeling.

islandBathy.py - Uses multispectral imagery (e.g., Sentinel-2) for bathymetry regression modeling and prediction export as GeoTIFF.

🧰 Dependencies Install the required packages:

bash Copy Edit pip install numpy pandas matplotlib scikit-learn seaborn pygeodesy gdal Note: GDAL might require additional system configuration. Refer to GDAL installation guide.

🔧 Usage Download ATL03 Data

Edit bounding box and ID in Python scrip clone.py:

python Copy Edit bounding_box='72.130081,11.494966,72.192133,11.609366' id = 'Bitra' Run:

bash Copy Edit python "Python scrip clone.py" Preprocess ICESat Data

Set the data and geoid paths in dataLoader_03.py, then run:

bash Copy Edit python "dataLoader_03.py" Visual Interactive Classification

Run interactive session to classify ocean/bed points:

bash Copy Edit python "islandicesat-2atl03dataanalysis.py" Bathymetry Modeling

Edit CSV and raster paths in islandBathy.py, then run:

bash Copy Edit python "islandBathy.py" GeoTIFF output is saved with predicted shallow water depth.

📊 Output CSVs of classified points with ocean/bed labels.

.tif raster predictions of shallow water bathymetry.

🧠 Algorithms Used DBSCAN clustering for spatial segmentation.

Linear, Bagging & ANN regression for elevation prediction.

NDWI-based filtering for water detection using Sentinel-2.

🏝️ Use Case Originally applied to Indian Ocean islands like Bitra, this pipeline is adaptable to other regions with appropriate bounding boxes and imagery.

📌 Credits Developed as part of remote sensing analysis for shallow water feature mapping using ICESat-2 ATL03 data
