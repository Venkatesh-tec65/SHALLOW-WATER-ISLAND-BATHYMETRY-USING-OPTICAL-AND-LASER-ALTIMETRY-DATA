  # for variable subsetting read this documentation
# https://nsidc.org/sites/nsidc.org/files/technical-references/ICESat2_ATL13_data_dict_v002.pdf


# imports
import requests
import json
import zipfile
import io
import math
import os
import shutil
import pprint
import re
import time
from datetime import date
from xml.etree import ElementTree as ET

# for now define bounding box manually
# bounding_box = '79.62725862,9.02919945,79.84802485,9.16063830'


# bbox2 = "-16.405662539,16.625480289,-16.319783598,16.777654898"
# bounding_box='8.329269146,7.786776546,8.365740747,7.823068483'

def icesat_profiles_download(bounding_box, id):
    # ------------------------------------------------------------------
    # bounding box
    bbox = bounding_box

    # ------------------------------------------------------------------
    # Input temporal range: maximum range available
    start_date = '2018-10-01'
    # start_date = '2020-05-06'
    start_time = '00:00:00'
    end_date = str(date.today())
    end_time = '00:00:00'
    temporal = start_date + 'T' + start_time + 'Z' + ',' + end_date + 'T' + end_time + 'Z'
    time_var = start_date + 'T' + start_time + ',' + end_date + 'T' + end_time

    # ------------------------------------------------------------------
    # EARTHDATA LOGIN
    email = 'email'
    uid = 'geointellect'
    pswd = 'Icesentinal@12345'
    email = 'venkateshvankadavth9@gmail.com'

    # identify operating system
    if os.name == 'nt':
      delimiter = "\\"
    else: # Mac or Linux
      delimiter = "/"

    # ------------------------------------------------------------------
    # Input data set short name (ATL08) of interest here.
    # ------------------------------------------------------------------
    short_name = 'ATL03'
    # Get json response from CMR collection metadata
    params = {'short_name': short_name}
    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'

    # EXCEPTION HANDLING 1
    # try 3 times (if any error occurs)
    # time.sleep and as often as necessary if Timeout error occurs
    attempts = 3
    while attempts > 0:
        try:
            response = requests.get(cmr_collections_url, params=params)
        except requests.exceptions.Timeout:
            time.sleep(1)
            attempts -= 1
        except requests.exceptions.RequestException:
            attempts -= 1
        # no error occurred -> continue
        break

    results = json.loads(response.content)
    # Find all instances of 'version_id' in metadata and print most recent version number
    versions = [el['version_id'] for el in results['feed']['entry']]
    latest_version = max(versions)
    print('The most recent version of ', short_name, ' is ', latest_version)

    # reformatting
    reformat = 'TABULAR_ASCII'
    # no projection parameters supported for this format
    projection = ''
    projection_parameters = ''

    # ---------------------------------------------------------------
    # VARIABLE SELECTION
    # for each of the groundtracks the same variables are available
    gtx = ["/gt1l", "/gt1r", "/gt2l", "/gt2r", "/gt3l", "/gt3r"]
    # var_subset = ["/land_segments/terrain/h_te_best_fit", "/land_segments/longitude", "/land_segments/latitude","/land_segments/terrain/h_te_interp","/land_segments/terrain/terrain_slope","/land_segments/segment_id_beg","/land_segments/segment_id_end"]
    # var_subset = ["/land_segments/terrain/h_te_best_fit", "/land_segments/longitude", "/land_segments/latitude"]
    var_subset = ["/heights/lat_ph","/heights/lon_ph","/heights/h_ph", "/geophys_corr/geoid", "/geophys_corr/tide_ocean" ,"/geophys_corr/tide_earth", "/geophys_corr/tide_equilibrium"]
    # "/land_segments/terrain/h_te_best_fit", "/land_segments/longitude", "/land_segments/latitude",
    # var_subset = ["/land_segments/longitude_20m", "/land_segments/latitude_20m",
                  # "/land_segments/terrain/h_te_best_fit_20m", "/land_segments/terrain/h_te_uncertainty"]
    #  , "/land_segments/terrain/h_te_uncertainty","/land_segments/segment_id_beg"
    # ---------------------------------------------------------------
    # GRANULE SEARCH BASED ON BOUNDING BOX
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    # bounding box input
    search_params = {
        'short_name': short_name,
        'version': latest_version,
        'temporal': temporal,
        'page_size': 100,
        'page_num': 1,
        'bounding_box': bounding_box}

    granules = []
    headers = {'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            # Handle empty case
            # Out of results, so break out of loop
            break

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    print('There are', len(granules), 'granules of', short_name, 'version', latest_version,
          'over my area and time of interest.')
    # --------------------------------------------------------------------------------
    # DO ALL THE FOLLOWING STEPS FOR EACH GROUNDTRACK SEPARATELY
    coverage = ""

    for gt in gtx:
        print(gt)
        coverage = ""
        coverage = coverage + "/orbit_info/orbit_number,"
        for var in var_subset:
            coverage = coverage + gt + var + ","
        # remove the last comma
        coverage = coverage[:-1]

        # -----------------------------------------------------------
        # Set NSIDC data access base URL and define how many orders are necessary
        base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
        # Set the request mode to asynchronous if the number of granules is over 100, otherwise synchronous is enabled by default
        if len(granules) > 100:
            request_mode = 'async'
            page_size = 2000
        else:
            page_size = 100
            request_mode = 'stream'
        # Determine number of orders needed for requests over 2000 granules.
        page_num = math.ceil(len(granules) / page_size)
        # print('There will be', page_num, 'total order(s) processed for this', short_name, 'request.')

        # not sure what the agent is
        agent = ''

        # ---------------------------------------------------------------
        # Query service capability URL
        from xml.etree import ElementTree as ET

        capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{latest_version}.xml'

        # Create session to store cookie and pass credentials to capabilities url

        session = requests.session()
        attempts = 3
        while attempts > 0:
            # EXCEPTION HANDLING 2
            print(attempts)
            try:
                s = session.get(capability_url)
                response = session.get(s.url, auth=(uid, pswd))
                # no error occurred
                break
            except requests.exceptions.Timeout:
                time.sleep(1)
                attempts -= 1
            except requests.exceptions.RequestException:
                attempts -= 1
        root = ET.fromstring(response.content)
        print('Done')

        # -----------------------------------------------------
        # API request formation
        # -----------------------------------------------------
        param_dict = {'short_name': short_name,
                      'version': latest_version,
                      'temporal': temporal,
                      'time': time_var,
                      'bounding_box': bounding_box,
                      'bbox': bbox,
                      'format': reformat,
                      'projection': projection,
                      'projection_parameters': projection_parameters,
                      'Coverage': coverage,
                      'page_size': page_size,
                      'request_mode': request_mode,
                      'agent': agent,
                      'email': email, }

        # Remove blank key-value-pairs
        param_dict = {k: v for k, v in param_dict.items() if v != ''}

        # Convert to string
        param_string = '&'.join("{!s}={!r}".format(k, v) for (k, v) in param_dict.items())
        param_string = param_string.replace("'", "")

        # Print API base URL + request parameters
        endpoint_list = []
        for i in range(page_num):
            page_val = i + 1
            API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
            endpoint_list.append(API_request)

        # print(*endpoint_list, sep = "\n")

        # -----------------------------------------------------
        # API request formation
        # -----------------------------------------------------
        path = str(os.getcwd() + delimiter + id)
        # print(path)
        if not os.path.exists(path):
            print('generate directory')
            os.mkdir(path)

        if request_mode == 'async':
            # Request data service for each page number, and unzip outputs
            for i in range(page_num):
                page_val = i + 1
                print('Order: ', page_val)

                # For all requests other than spatial file upload, use get function
                request = session.get(base_url, params=param_dict)

                print('Request HTTP response: ', request.status_code)

                # Raise bad request: Loop will stop for bad response code.
                request.raise_for_status()
                print('Order request URL: ', request.url)
                esir_root = ET.fromstring(request.content)
                print('Order request response XML content: ', request.content)

                # Look up order ID
                orderlist = []
                for order in esir_root.findall("./order/"):
                    orderlist.append(order.text)
                orderID = orderlist[0]
                print('order ID: ', orderID)

                # Create status URL
                statusURL = base_url + '/' + orderID
                print('status URL: ', statusURL)

                # Find order status
                request_response = session.get(statusURL)
                print('HTTP response from order response URL: ', request_response.status_code)

                # Raise bad request: Loop will stop for bad response code.
                request_response.raise_for_status()
                request_root = ET.fromstring(request_response.content)
                statuslist = []
                for status in request_root.findall("./requestStatus/"):
                    statuslist.append(status.text)
                status = statuslist[0]
                print('Data request ', page_val, ' is submitting...')
                print('Initial request status is ', status)

                # Continue loop while request is still processing
                while status == 'pending' or status == 'processing':
                    print('Status is not complete. Trying again.')
                    time.sleep(1)
                    loop_response = session.get(statusURL)

                    # Raise bad request: Loop will stop for bad response code.
                    loop_response.raise_for_status()
                    loop_root = ET.fromstring(loop_response.content)

                    # find status
                    statuslist = []
                    for status in loop_root.findall("./requestStatus/"):
                        statuslist.append(status.text)
                    status = statuslist[0]
                    print('Retry request status is: ', status)
                    if status == 'pending' or status == 'processing':
                        continue

                # Order can either complete, complete_with_errors, or fail:
                # Provide complete_with_errors error message:
                if status == 'complete_with_errors' or status == 'failed':
                    messagelist = []
                    for message in loop_root.findall("./processInfo/"):
                        messagelist.append(message.text)
                    print('error messages:')
                    pprint.pprint(messagelist)
                try:
                # Download zipped order if status is complete or complete_with_errors
                    if status == 'complete' or status == 'complete_with_errors':
                        downloadURL = 'https://n5eil02u.ecs.nsidc.org/esir/' + orderID + '.zip'
                        print('Zip download URL: ', downloadURL)
                        print('Beginning download of zipped output...')
                        zip_response = session.get(downloadURL)
                        # Raise bad request: Loop will stop for bad response code.
                        zip_response.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
                            z.extractall(path)
                        print('Data request', page_val, 'is complete.')
                    else:
                        print('Request failed.')
                except requests.exceptions.HTTPError as e:
                    estatus_code = e.response.status_code
                    print("Error HTTP:", estatus_code)
                    pass
        else:
            for i in range(page_num):
                page_val = i + 1
                # print('Order: ', page_val)
                print('Requesting...')

                # EXCEPTION HANDLING 3
                # try 3 times (if any error occurs)
                # time.sleep and as often as necessary if Timeout error occurs
                attempts = 3
                while attempts > 0:
                    # print(attempts)
                    try:
                        request = session.get(base_url, params=param_dict)
                    except requests.exceptions.Timeout:
                        time.sleep(1)
                        attempts -= 1
                    except requests.exceptions.RequestException:
                        attempts -= 1
                    # no error occurred -> continue
                    break

                print('HTTP response from order response URL: ', request.status_code)
                # proceed only if https request result is 200
                # if request.status_code == 201:
                #     print('Request status code is 201.')
                #     print(request)
                #     location = request.headers['Location']
                #     print(location)
                #     #resource = location.get
                #     #resource = location.get
                if request.status_code == 200:
                    request.raise_for_status()
                    d = request.headers['content-disposition']
                    fname = re.findall('filename=(.+)', d)
                    dirname = os.path.join(path, fname[0].strip('\"'))
                    print('Downloading...')
                    open(dirname, 'wb').write(request.content)
                    print('Data request', page_val, 'is complete.')

                    # Unzip outputs
                    for z in os.listdir(path):
                        print(z)
                        if z.endswith('.zip'):
                            zip_name = path + delimiter + z
                            zip_ref = zipfile.ZipFile(zip_name)
                            zip_ref.extractall(path)
                            zip_ref.close()
                            os.remove(zip_name)



                    # Clean up Outputs folder by removing individual granule folders
                    for root, dirs, files in os.walk(path, topdown=False):
                        for file in files:
                            new_name = file[:-6] + gt[1:] + ".ascii"
                            try:
                                shutil.move(os.path.join(root, file), path)
                            except OSError:
                                pass
                            print("privet")
                            # only for the files which do not have gt info yet
                            if ("gt" in file) == False:
                                try:
                                    os.rename(os.path.join('r', path, file), os.path.join('r', path, new_name))
                                except FileExistsError:
                                    os.remove(os.path.join('r', path, new_name))
                                    os.rename(os.path.join('r', path, file), os.path.join('r', path, new_name))
                                    print("File Exist Error! Existed file deleted and created new one")
                                except FileNotFoundError:
                                    print("File Not Found Error")
                                    pass
                        for name in dirs:
                             os.rmdir(os.path.join(root, name))

        # Unzip outputs
        for z in os.listdir(path):
            print(z)
            if z.endswith('.zip'):
                zip_name = path + delimiter + z
                zip_ref = zipfile.ZipFile(zip_name)
                zip_ref.extractall(path)
                zip_ref.close()
                os.remove(zip_name)



        # Clean up Outputs folder by removing individual granule folders
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                new_name = file[:-6] + gt[1:] + ".ascii"
                try:
                    shutil.move(os.path.join(root, file), path)
                except OSError:
                    pass
                # only for the files which do not have gt info yet
                if ("gt" in file) == False:
                    try:
                        os.rename(os.path.join('r', path, file), os.path.join('r', path, new_name))
                    except FileExistsError:
                        os.remove(os.path.join('r', path, new_name))
                        os.rename(os.path.join('r', path, file), os.path.join('r', path, new_name))
                        print("File Exist Error! Existed file deleted and created new one")
                    except FileNotFoundError:
                        print("File Not Found Error")
                        pass

            for name in dirs:
                 os.rmdir(os.path.join(root, name))
        print('Next gtx')
    print('all gtx through')
    return None

if __name__ == "__main__":
   bounding_box='73.650316,10.794872,73.712868,10.832079'
   id = 'Andrott_002'
   icesat_profiles_download(bounding_box,id)  
