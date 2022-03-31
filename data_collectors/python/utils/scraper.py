import requests
from bs4 import BeautifulSoup


def call_url_get_bs4(url, cookies={}, headers={}):
    response = requests.get(
        url, verify=False, headers=headers, cookies=cookies, timeout=3
    )
    soup = BeautifulSoup(response.content, "html.parser")

    return soup
