#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:12:49 2022

@author: eliorland
"""

'''
Orignal script used to make the product info file. It remains unchanged 
since its initial run and is here to show how the file was first created
'''


import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

archive_url = 'https://www.taylorstitch.com/collections/mens-archive'
shirts_url = 'https://www.taylorstitch.com/collections/mens-shirts?sorted=best-selling-sales-count'
base_url = 'https://www.taylorstitch.com'

url_list = [archive_url, shirts_url]

product_urls = [] # list of all product pages to get later

for url in url_list:
    site = requests.get(url)

    soup = BeautifulSoup(site.text, 'html.parser')

    products = soup.find_all('ul',{'class':'product matrix'})
    products = products[0].find_all('a',href=True) # a tags hold products here

    for product in products:
        product_urls.append(base_url+product['href'])
        
# look at each product, pull relevent info.
# store all info in lists, which will be converted to pandas df later
product_title = []
product_description = []
product_material = []


for product in product_urls:

    product_page = requests.get(product)
    product_soup = BeautifulSoup(product_page.text, 'html.parser')
    
    title_info = product_soup.find('h1')['data-title']
    
    description_info = product_soup.find_all('div',
                                             {'id':'collapsible-description'})
    material_info = product_soup.find_all('div',
                                             {'id':'collapsible-material'})
    #if len(description_info)==0:
    #   continue
    try: 
        description = description_info[0].find('p').text
        material = material_info[0].find('p').text
    
    except: 
        continue
        
    product_title.append(title_info)
    product_description.append(description)
    product_material.append(material)

    
all_info = pd.DataFrame(list(zip(product_title, 
                                 product_description,
                                 product_material)),
               columns =['Name', 'Description', 'Material'])

all_info.to_csv('taylor_stitch_info.csv',index=True,header=True)

