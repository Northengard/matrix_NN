import pandas as pd
import geopandas as gpd

import numpy as np
from time import sleep
from geopy.geocoders import Nominatim, ArcGIS
from scipy.spatial.distance import pdist
from shapely.geometry.point import Point

# globals
LOCATOR = Nominatim(user_agent='myGeocoder')


def get_centers(poligon_coords):
    long, lat = poligon_coords.centroid.xy
    coordinates = list([np.mean(lat.tolist()), np.mean(long.tolist())])
    return coordinates


def get_address(coordinates):
    location = LOCATOR.reverse(coordinates)
    return location[0]


if __name__ == '__main__':
    edges = gpd.read_file('nn_roads.json')
    buildings = gpd.read_file('nn_buildings.json')
    nodes = gpd.read_file('nn_nodes.json')
    area = gpd.read_file('nn_area.json')

    # according to https://wiki.openstreetmap.org/wiki/Key:building
    building_types = buildings.building.value_counts().index.values
    buildings = buildings[buildings.building != 'no']

    living_building_types = dict.fromkeys(building_types, False)
    living_building_types['detached'] = True
    living_building_types['apartments'] = True
    living_building_types['house'] = True
    living_building_types['dormitory'] = True
    living_building_types['residential'] = True
    living_building_types['apartments'] = True

    work_buildings_types = dict.fromkeys(building_types, False)
    work_buildings_types['commercial'] = True
    work_buildings_types['industrial'] = True
    work_buildings_types['service'] = True
    work_buildings_types['school'] = True
    work_buildings_types['retail'] = True
    work_buildings_types['kiosk'] = True
    work_buildings_types['greenhouse'] = True
    work_buildings_types['hangar'] = True
    work_buildings_types['hospital'] = True
    work_buildings_types['college'] = True
    work_buildings_types['stadium'] = True
    work_buildings_types['train_station'] = True
    work_buildings_types['cafe'] = True
    work_buildings_types['supermarket'] = True
    work_buildings_types['transportation'] = True
    work_buildings_types['shed'] = True
    work_buildings_types['hut'] = True
    work_buildings_types['construction'] = True
    work_buildings_types['kindergarten'] = True
    work_buildings_types['roof'] = True
    work_buildings_types['garages'] = True
    work_buildings_types['garage'] = True
    work_buildings_types['university'] = True
    work_buildings_types['manufacture'] = True

    is_living = buildings.building.apply(lambda x: living_building_types[x])
    buildings['is_living'] = is_living
    is_working = buildings.building.apply(lambda x: work_buildings_types[x])
    buildings['is_working'] = is_working

    # drop redundant columns
    nan_part = buildings.isna().sum() / buildings.shape[0]
    nan_part = nan_part[nan_part < 0.7]
    buildings = buildings[nan_part.index.values]

    # get building centers
    centers = buildings.geometry.apply(lambda x: get_centers(x))
    centers = centers.apply(lambda x: np.round(x, decimals=4))
    centers = pd.DataFrame(np.vstack(centers.values))
    centers.drop_duplicates(inplace=True)
    print(centers.shape)
    buildings = buildings.loc[buildings.index.isin(centers.index.values)]
    centers = pd.Series(data=np.split(centers.values, centers.values.shape[0]), index=centers.index)
    buildings['building_coords'] = centers

    # rows with missed addreses
    missed = buildings[buildings['addr:street'].isna() | buildings['addr:street'].isna()]

    addresses = dict()
    for idx, center_coords in missed.building_coords.items():
        try:
            addresses[idx] = (get_address(center_coords))
        except:
            sleep(2)
            addresses[idx] = None
    #         addresses.append(get_address(center_coords))
    addresses = pd.Series(addresses).dropna()
    # dump files
    pd.Series(addresses).to_csv('missed_buildings_addresses.csv')

    addresses_split = addresses.str.split(',')
    addresses_full = addresses_split[addresses_split.apply(lambda x: x[0][0].isnumeric())]
    print(len(addresses_split))

    addresses_split = pd.Series(addresses_split)
    addresses_weak = addresses_split[~addresses_split.apply(lambda x: x[0][0].isnumeric())]
    addresses_weak = addresses_weak[addresses_weak.apply(lambda x: False if 'Нижний Новгород' == x[0] else True)]
    # sequence:
    # building type, house number, street, what_is_it,
    # city region, city, city district, country region, country district, postcode, country
    good_addresses = addresses_weak[addresses_weak.apply(lambda x: x[1][1].isnumeric())]
    addresses_weak = addresses_weak[~addresses_weak.apply(lambda x: x[1][1].isnumeric())]

    company_names_mask_1 = addresses_weak.apply(lambda x: True if 'район' == x[0].split(' ')[-1] else False)
    company_names_mask_2 = addresses_weak.apply(lambda x: True if 'проспект' in x[0] else False)
    company_names_mask_3 = addresses_weak.apply(lambda x: True if 'улица' in x[0] else False)
    company_names_mask_4 = addresses_weak.apply(lambda x: True if 'совхоз Доскино' == x[0] else False)
    company_names_mask_5 = addresses_weak.apply(lambda x: True if 'общежитие' in x[1] else False)

    mask = company_names_mask_1 | company_names_mask_2
    mask = mask | company_names_mask_3 | company_names_mask_4 | company_names_mask_5
    company_name = addresses_weak[~mask]
    print(len(good_addresses), len(addresses_full), len(company_name))

    com_name = pd.concat([company_name.apply(lambda x: x[0]), good_addresses.apply(lambda x: x[0])])
    com_addr = pd.concat([company_name.apply(lambda x: x[1]),
                          good_addresses.apply(lambda x: x[2]),
                          addresses_full.apply(lambda x: x[1])])
    com_house_num = pd.concat([good_addresses.apply(lambda x: x[1]),
                               addresses_full.apply(lambda x: x[0])])

    com_addr.sort_index(inplace=True)
    com_name.sort_index(inplace=True)
    com_house_num.sort_index(inplace=True)

    missed['company_name'] = com_name
    missed['addr:street'] = com_addr
    missed['addr:housenumber'] = com_house_num

    # drop other values
    buildings.loc[buildings.index.isin(missed.index)] = missed
    buildings = buildings.loc[buildings[['addr:housenumber', 'addr:street']].dropna().index]

    buildings_data = pd.read_csv('highrise_building_data_processed.csv', index_col=0)
    buildings_data.living_number = buildings_data.living_number.str.replace(' ', '').astype('int')
    buildings_data.floor_num = buildings_data.floor_num.apply(
        lambda x: x.replace('14-16', '14') if type(x) == str else x).astype(int)

    apartments = buildings[buildings.building == 'apartments']
    apartments = pd.merge(apartments, buildings_data, on=['addr:housenumber', 'addr:street'], how='left')

    buildings['living_number'] = None
    buildings['floor_num'] = None

    buildings.loc[apartments.index] = apartments

    buildings.floor_num[buildings.floor_num.isna()] = buildings['building:levels'][buildings.floor_num.isna()]
    buildings['floor_num'] = buildings['floor_num'].apply(
        lambda x: x.replace('14-16', '14') if type(x) == str else x).astype(float)
    buildings.drop(columns=['building:levels'], inplace=True)
    buildings.living_number[((buildings['building'] == 'detached') | (buildings['building'] == 'house'))] = 1

    mean_lv_num = dict()
    for fl_num in buildings_data.floor_num.unique():
        if fl_num == -1:
            continue
        mean_lv_num[fl_num] = np.round(buildings_data.living_number[(buildings_data.floor_num == fl_num) & (
                buildings_data.living_number != -1)].mean()).astype(int)
    mean_lv_num.pop(22)  # dirt because it is NaN


    def flour_to_ln(fn):
        if fn in mean_lv_num.keys():
            val = mean_lv_num[fn]
        else:
            val = None
        return val


    # buildings['living_number'] =
    mean_ln = buildings['floor_num'].apply(flour_to_ln)
    buildings['living_number'].loc[buildings['living_number'].isna()] = \
        mean_ln.loc[buildings['living_number'].isna()].values

    # drop na
    buildings = buildings[~buildings.index.isin(
        buildings[buildings.is_living][buildings[buildings.is_living]['living_number'].isna()].index)]

    desired_locations = {'theater': ['улица Белинского, 59, Нижний Новгород, Нижегородская обл., 603006',
                                     'улица Эльтонская, 30, Нижний Новгород, Нижегородская обл., 603146',
                                     'улица Максима Горького, 145, Нижний Новгород, Нижегородская обл., 603006',
                                     'улица Максима Горького, 145, Нижний Новгород, Нижегородская обл., 603006',
                                     'улица Варварская, 32, Нижний Новгород, Нижегородская обл., 603006',
                                     'улица Грузинская, 23, Нижний Новгород, Нижегородская обл., 603000',
                                     'Большая Покровская улица, 39Б, Нижний Новгород, Нижегородская обл., 603000',
                                     'Большая Покровская улица, 43, Нижний Новгород, Нижегородская обл., 603000',
                                     'Большая Покровская улица, 13, Нижний Новгород, Нижегородская обл., 603000',
                                     'Большая Покровская улица, 4а, Нижний Новгород, Нижегородская обл., 603005',
                                     'улица Рождественская, 24В, Нижний Новгород, Нижегородская обл., 603001',
                                     'Революционная улица, 20, Нижний Новгород, Нижегородская обл., 603002',
                                     'Канавинская улица, 2, Нижний Новгород, Нижегородская обл., 603002',
                                     'улица Июльских Дней, 21/96, Нижний Новгород, Нижегородская обл., 603011'],
                         'hospital': ['улица Нестерова, 34, Нижний Новгород, Нижегородская обл., 603005',
                                      'Верхневолжская набережная, 21, Нижний Новгород, Нижегородская обл., 603155',
                                      'улица Ульянова, 41, Нижний Новгород, Нижегородская обл., 603155',
                                      'улица Костина, 5б, Нижний Новгород, Нижегородская обл., 603000',
                                      'Республиканская улица, 47, Нижний Новгород, Нижегородская обл., 603089',
                                      'улица Ильинская, 11, Нижний Новгород, Нижегородская обл., 603000',
                                      'улица Чернышевского, 22, Нижний Новгород, Нижегородская обл., 603000',
                                      'Нижневолжская наб., 2, Нижний Новгород, Нижегородская обл., 603005',
                                      'улица Минина, 20Е, Нижний Новгород, Нижегородская обл., 603155',
                                      'улица Семашко, 22, Нижний Новгород, Нижегородская обл., 603155',
                                      'улица Адмирала Васюнина, 2А, Нижний Новгород, Нижегородская обл., 603106',
                                      'улица Ивлиева генерала, 32/1, Нижний Новгород, Нижегородская обл., 603122',
                                      'улица Верхне-Печерская, 6, Нижний Новгород, Нижегородская обл., 603163',
                                      'улица Германа Лопатина, 2, Нижний Новгород, Нижегородская обл., 603163',
                                      'улица Донецкая, 4, Нижний Новгород, Нижегородская обл., 603163',
                                      'улица Ошарская, 88, Нижний Новгород, Нижегородская обл., 603105',
                                      'улица Бекетова, 39, Нижний Новгород, Нижегородская обл., 603146',
                                      'улица Нестерова, 34 А, Нижний Новгород, Нижегородская обл., 603005',
                                      'улица Ломоносова, 13, Нижний Новгород, Нижегородская обл., 603105',
                                      'улица Заярская, 4, Нижний Новгород, Нижегородская обл., 603146',
                                      'улица Тимирязева, 5, Нижний Новгород, Нижегородская обл., 603022',
                                      'Ашхабадская улица, 8, Нижний Новгород, Нижегородская обл., 603000',
                                      'улица Грузинская, 10, Нижний Новгород, Нижегородская обл., 603000',
                                      'улица Ильинская, 78, Нижний Новгород, Нижегородская обл., 603000',
                                      'улица Окский Съезд, 2А, Нижний Новгород, Нижегородская обл., 603022',
                                      'Студенческая улица, 21, Нижний Новгород, Нижегородская обл., 603022',
                                      'улица Бекетова, 8А, Нижний Новгород, Нижегородская обл., 603057',
                                      'улица С. Есенина, д. 27, Нижний Новгород, Нижегородская обл., 603070',
                                      'Портовый пер., д. 8, Нижний Новгород, Нижегородская обл., 603070',
                                      'улица Приокская, 14, Нижний Новгород, Нижегородская обл., 603002',
                                      'проспект Ленина, 16а, Нижний Новгород, Нижегородская обл., 603140',
                                      'улица Норильская, 55, Нижний Новгород, Нижегородская обл., 603135',
                                      'улица Саврасова, 10, Нижний Новгород, Нижегородская обл., 603146',
                                      'улица Бекетова, 48, Нижний Новгород, Нижегородская обл., 603146',
                                      'улица Бекетова, 13, Нижний Новгород, Нижегородская обл., 603057',
                                      'проспект Гагарина, 50к15, Нижний Новгород, Нижегородская обл., 603057',
                                      'улица Белинского, 58/60, Нижний Новгород, Нижегородская обл., 603086',
                                      'улица Новая, 28, Нижний Новгород, 603000',
                                      'Большая Покровская улица, 62/5, Нижний Новгород, Нижегородская обл., 603000',
                                      'Провиантская улица, 47, Нижний Новгород, Нижегородская обл., 603006',
                                      'улица Володарского, 56, Нижний Новгород, Нижегородская обл., 603006',
                                      'улица Пискунова, 21/2, Нижний Новгород, Нижегородская обл., 603005',
                                      'Большая Покровская улица, 16 а, г. Н. Новгород, Нижегородская обл., 603005',
                                      'Большая Покровская улица, 23, Нижний Новгород, Нижегородская обл., 603005',
                                      'Сергиевская, 8, Нижний Новгород, Нижегородская обл., 603000',
                                      'улица Бетанкура, 3, Нижний Новгород, Нижегородская обл., 603086',
                                      'улица Октябрьской Революции, 43, Нижний Новгород, Нижегородская обл., 603011',
                                      'Трамвайный пер., 2, Нижний Новгород, Нижегородская обл., 603140',
                                      'проспект Ленина, 93, 2 этаж, Нижний Новгород, Нижегородская обл., 603064'],
                         'Institute': ['улица Грузинская, 44, Нижний Новгород, Нижегородская обл., 603005',
                                       'улица Ошарская, 8 д, г. Н. Новгород, Нижегородская обл., 603006',
                                       'улица Ульянова, 10, Нижний Новгород, Нижегородская обл., 603005',
                                       'Верхневолжская набережная, 18, Нижний Новгород, Нижегородская обл., 603155',
                                       'улица Семашко, 20, Нижний Новгород, Нижегородская обл., 603005',
                                       'улица Семашко, 22, Нижний Новгород, Нижегородская обл., 603155',
                                       'Ванеева улица, 127, Нижний Новгород, Нижегородская обл., 603105',
                                       'улица Салганская, 10, Нижний Новгород, Нижегородская обл., 603105',
                                       'проспект Гагарина, 23, к.5, Нижний Новгород, Нижегородская обл., 603022',
                                       'улица Костина, 4, Нижний Новгород, Нижегородская обл., 603000',
                                       'Гагарина проспект 23/6, Нижний Новгород, Нижегородская обл., 603022',
                                       'улица Медицинская, 5А, Нижний Новгород, Нижегородская обл., 603104',
                                       'улица Героя Шапошникова, 5, Нижний Новгород, Нижегородская обл., 603009',
                                       'улица Маршала Голованова, 21, Нижний Новгород, Нижегородская обл., 603107',
                                       'улица Журова, 2, Нижний Новгород, Нижегородская обл., 603011',
                                       'Московское шоссе, 31, Нижний Новгород, Нижегородская обл., 603116',
                                       'улица Марата, 51, Нижний Новгород, Нижегородская обл., 603002',
                                       'проспект Гагарина, 176, Нижний Новгород, Нижегородская обл., 603009',
                                       'улица Нартова, 6к6, Нижний Новгород, Нижегородская обл., 603104', ]}

    geolocator = ArcGIS(user_agent="specify_your_app_name_here")
    received_locations = dict.fromkeys(desired_locations.keys())
    for wt in desired_locations.keys():
        addresses = desired_locations[wt]
        received_locations[wt] = dict.fromkeys(addresses)
        for adr in addresses:
            location = geolocator.geocode(adr)
            received_locations[wt][adr] = [location.latitude, location.longitude]


    def create_row(received_loc, build_type, row_name):
        human_addr = received_loc[0].split(',')
        coords = received_loc[1]
        row = pd.Series(index=buildings.columns, data=None)
        row['addr:housenumber'] = human_addr[1].strip()
        row['addr:street'] = human_addr[0].strip()
        row['building_coords'] = coords
        row['is_living'] = False
        row['is_working'] = True
        row['building'] = build_type
        row['geometry'] = Point(coords[::-1])
        row.name = row_name
        return row


    last_id = buildings.shape[0]
    for wt in received_locations.keys():
        for it in received_locations[wt].items():
            buildings = buildings.append(create_row(it, wt, last_id))
            last_id += 1

    buildings['is_living'] = buildings['is_living'].fillna(0)
    buildings['is_living'] = buildings['is_living'].astype(bool)
    buildings.drop_duplicates(
        subset=['addr:housenumber', 'addr:street', 'building', 'floor_num', 'id', 'is_living', 'is_working',
                'living_number'], inplace=True)
    buildings['building_coords'] = buildings['building_coords'].apply(
        lambda x: x.tolist() if type(x) == np.ndarray else x)
    buildings.index = pd.RangeIndex(buildings.shape[0])
    with open('nn_buildings_last_modified.json', 'w') as f:
        f.write(buildings.to_json())


    def lat_long_to_distance(lat_log_1, lat_log_2):
        '''returns distance in km'''
        lat_log_diff = lat_log_1 - lat_log_2
        kmeters_in_grad_lat = 111.134861111
        kmetters_in_grad_long = np.cos(56.2866538296335)  # NN mean latitude is 56.28... according to the given data
        lat, long = lat_log_diff
        return np.linalg.norm([kmeters_in_grad_lat * lat, kmetters_in_grad_long * long])


    dist_values = pdist(np.stack(buildings.loc[(buildings['is_living'] == True), 'building_coords'].values),
                        metric=lat_long_to_distance)
    np.save('living_distance_values.npy', dist_values)

    distance_matrix = np.zeros(21684)
    tr_idxs = np.triu_indices(21684, 1)
    distance_matrix[tr_idxs] = dist_values
    distance_matrix[tr_idxs[1], tr_idxs[0]] = dist_values
    np.save('living_distance_matr.npy', distance_matrix)

    buildings.loc[buildings['addr:street'].apply(lambda x: 'Коновалова' in x), 'is_working'] = ~ \
        buildings[buildings['addr:street'].apply(lambda x: 'Коновалова' in x)]['is_living']
    buildings.loc[buildings['addr:street'].apply(lambda x: 'Коновалова' in x) & (
            buildings.is_working == True), 'building'] = 'industrial'
    buildings.loc[buildings['addr:street'].apply(lambda x: 'Федосеенко' in x) & (buildings.is_living == False) & (
            buildings.building == 'yes'), 'is_working'] = True
    buildings_new = np.random.randint(low=0, high=6, size=262)
    types = {0: 'garage',
             1: 'industrial',
             2: 'retail',
             3: 'manufacture',
             4: 'service',
             5: 'office'}
    pd.Series(buildings_new).value_counts()
    buildings.loc[buildings['addr:street'].apply(lambda x: 'Федосеенко' in x) & (buildings.is_living == False) & (
            buildings.building == 'yes'), 'building'] = list(map(lambda x: types[x], buildings_new))
    buildings.loc[buildings['addr:street'].apply(lambda x: 'Интернациональная' in x) & (
            buildings.is_living == False), 'is_working'] = True
    buildings_new = np.random.randint(low=0, high=6, size=63)
    types = {0: 'garage',
             1: 'industrial',
             2: 'retail',
             3: 'manufacture',
             4: 'service',
             5: 'office'}
    pd.Series(buildings_new).value_counts()
    buildings.loc[
        buildings['addr:street'].apply(lambda x: 'Интернациональная' in x) & (buildings.is_working == True) & (
                buildings.building == 'yes'), 'building'] = list(map(lambda x: types[x], buildings_new))
    buildings.loc[
        buildings['addr:street'].apply(lambda x: 'Ленина' in x) & (buildings.is_living == False), 'is_working'] = True
    buildings_new = np.random.randint(low=0, high=6, size=188)
    types = {0: 'garage',
             1: 'industrial',
             2: 'retail',
             3: 'manufacture',
             4: 'service',
             5: 'office',
             6: 'transportation',
             7: 'warehouse',
             8: 'carport'}
    pd.Series(buildings_new).value_counts()
    buildings.loc[buildings['addr:street'].apply(lambda x: 'Ленина' in x) & (buildings.is_working == True) & (
            buildings.building == 'yes'), 'building'] = list(map(lambda x: types[x], buildings_new))
    building_mask = buildings.building_coords.apply(lambda x: (56.28 < x[0] < 56.35) & (43.97 < x[1] < 44.1))
    buildings.loc[building_mask & (buildings['is_living'] == False), 'is_working'] = True
    # have to get real addresses and numbers if future work is required
    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=130), 'building'] = 'supermarket'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=300), 'building'] = 'cafe'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=25), 'building'] = 'government'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=100), 'building'] = 'public'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=9), 'building'] = 'fire_station'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=1309), 'building'] = 'civic'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=458), 'building'] = 'service'
    num_b = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].shape[0]
    buildings_new = np.random.randint(low=0, high=4, size=num_b)
    types = {0: 'kiosk',
             1: 'retail',
             2: 'office',
             3: 'commercial'}
    pd.Series(buildings_new).value_counts()
    buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes'), 'building'] = list(
        map(lambda x: types[x], buildings_new))
    building_mask = buildings.building_coords.apply(lambda x: (56.25 < x[0]) & (x[1] < 43.95))
    buildings.loc[building_mask & (buildings['is_living'] == False), 'is_working'] = True
    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=110), 'building'] = 'supermarket'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=300), 'building'] = 'cafe'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=30), 'building'] = 'government'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=125), 'building'] = 'public'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=12), 'building'] = 'fire_station'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=1439), 'building'] = 'civic'

    idxs = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].index
    buildings.loc[np.random.choice(idxs, size=38), 'building'] = 'industrial'
    num_b = buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes')].shape[0]
    buildings_new = np.random.randint(low=0, high=4, size=num_b)
    types = {0: 'kiosk',
             1: 'retail',
             2: 'office',
             3: 'commercial'}
    pd.Series(buildings_new).value_counts()
    buildings.loc[building_mask & (buildings['is_working'] == True) & (buildings.building == 'yes'), 'building'] = list(
        map(lambda x: types[x], buildings_new))

    with open('nn_buildings_last_modified_180520.json', 'w') as f:
        f.write(buildings.to_json())
