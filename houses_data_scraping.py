import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_buildings_ids(request_text):
    def handle_row(row_data):
        code_phrase = 'nizhniy-novgorod/'
        code_phrase_len = len(code_phrase)
        start = row_data.find(code_phrase) + code_phrase_len
        end_bias = row_data[start:].find('''">''')
        building_id = row_data[start:start + end_bias]
        return building_id

    building_ids = list()

    ad_tag_index = request_text.find(
        '''<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>''')
    table_bias = request_text[ad_tag_index:].find(
        '''<table id="grid-data" class="table table-condensed table-hover table-striped">''')
    end_table_bias = request_text[ad_tag_index + table_bias:].find('''</table>''')
    table_html = request_text[ad_tag_index + table_bias:ad_tag_index + table_bias + end_table_bias]

    rows = table_html.split('td')
    # drop header
    rows.pop(0)
    rows = [row if 'nizhniy-novgorod/' in row else '' for row in rows]
    rows.sort(reverse=True)
    rows = rows[:rows.index('')]
    building_ids.extend(list(map(handle_row, rows)))

    return building_ids


def get_data_by_index(index):
    start = building_page.text.find(index)
    if start > 0:
        start = building_page.text[start:].find('<dd>') + 4 + start
        end = building_page.text[start:].find('''</dd>''') + start
        value = building_page.text[start:end]
        if value == 'Не заполнено':
            value = -1
    else:
        value = -1
    return value


if __name__ == '__main__':
    bulid_ids = list()
    base_url = 'http://dom.mingkh.ru/nizhegorodskaya-oblast/nizhniy-novgorod/houses?page='
    for page_id in range(1, 108):
        print('handle', page_id)
        table_page_url = base_url + str(page_id)
        table_page = requests.get(table_page_url)
        bulid_ids.extend(get_buildings_ids(table_page.text))

    base_url = 'http://dom.mingkh.ru/nizhegorodskaya-oblast/nizhniy-novgorod/'
    buildings_data = dict()
    for bulding_id in bulid_ids:
        print('handle', bulding_id)
        building_page_url = base_url + bulding_id
        building_page = requests.get(building_page_url)

        parsed = BeautifulSoup(building_page.text)
        address = parsed.title.text[parsed.title.text.find('г.'):]
        print(address)

        living_number = get_data_by_index('Жилых помещений')
        floor_num = get_data_by_index('Количество этажей')

        buildings_data[bulding_id] = {'address': address,
                                      'living_number': living_number,
                                      'floor_num': floor_num}

    pd.DataFrame.from_dict(buildings_data).transpose().to_csv('highrise_building_data.csv')