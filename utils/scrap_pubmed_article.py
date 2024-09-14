"""
    Nice References
    - https://stackoverflow.com/questions/72190902/can-biopython-entrez-pull-full-pubmed-articles-from-a-list-of-pmids
    - https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/
    or directly crawl the HTML
    - sample: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8822225/
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import pprint

def scrap_article(pmcid="PMC8822225"):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    response = requests.get(url)
    
    article_json = json.loads(response.content)[0]

    raw_text = ""
    for passage in article_json['documents'][0]['passages']:
        # passage is for each paragraph I presume
        # ['passages'][0] is for document information
        # ['passages'][1] is for document abstract
        # ['passages'][2] starts for Intro
        raw_text += passage['text'] + "\n"
        if passage['text'] in ["ACKNOWLEDGEMENTS", "CONSENT FOR PUBLICATION", "FUNDING", "CONFLICT OF INTEREST"]:
            break
    
    return raw_text

def scrap_article_bs4(pmcid="PMC8822225"):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    headers = {
        ":authority": "www.ncbi.nlm.nih.gov",
        ":method": "GET",
        ":path": "/pmc/articles/PMC8822225/",
        ":scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Cookie": "ncbi_sid=5917493A6C45CDC3_8188SID; pmc-frontend-csrftoken=OkthmmJjFM1C3590dCzJ7W1NZCqYdvGo; books.article.report=; _ga_7147EPK006=GS1.1.1726131071.1.0.1726131072.0.0.0; _ga_P1FPTH9PL4=GS1.1.1726131072.1.0.1726131072.0.0.0; _ce.irv=new; cebs=1; cebsp_=1; _ce.s=v~ae680f253ff5c34e22cdf9462ff420b9f2164389~lcw~1726132287862~lva~1726131074131~vpv~0~v11.cs~156325~v11.s~2ab9f8a0-70e4-11ef-be13-1399c07553bb~v11.sla~1726132370847~lcw~1726132370847; _ga=GA1.1.591917809.1724148341; ncbi_pinger=N4IgDgTgpgbg+mAFgSwCYgFwgMwCZcDCuALAIIBsAHACICs5u1AYrQAzuvYCcptljXAEJcCrAHQBGMQFs4uEABoQAVwB2AGwD2AQ1SqoADwAumUPKxhpAYwC0AMwibVRqKvRLsmcNcUhiX7QgjZCt1KF9aLwB6QODQqABnKIAFAFkCSn58XFoo3wl/LGgjCGRYKHcQXFYvPEISChp6RhYOTh4+AWFRSRk5X1wJL0tbBycXNwwRjFiQsIwYoLnElPTM7Jy8pXwvAHd9sVUrACNkQ/VpQ+REMQBzTRgBri8JbHJyX2warAB2Kk+hlhXu9PuYQD9sJRPp4sHZtOoEuEPIUQCVlEicFCsJ9nlhMtgChJaEMlMRviBMsRsLQflwaqSYSBxGSxJ5SSi1FpdPpjL5iJEsPSQLRGa8chEUUShfQXpkPkoaV4heRAeCviAAL4aoA=; _ga_DP2X732JSX=GS1.1.1726301792.21.0.1726301792.0.0.0; _ga_CSLL4ZEK4L=GS1.1.1726301792.21.0.1726301792.0.0.0",
        "Priority": "u=0, i",
        "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Opera GX";v="112"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "Windows",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": 1,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.226 Safari/537.36",
    }
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Cookie": "ncbi_sid=5917493A6C45CDC3_8188SID; pmc-frontend-csrftoken=OkthmmJjFM1C3590dCzJ7W1NZCqYdvGo; books.article.report=; _ga_7147EPK006=GS1.1.1726131071.1.0.1726131072.0.0.0; _ga_P1FPTH9PL4=GS1.1.1726131072.1.0.1726131072.0.0.0; _ce.irv=new; cebs=1; cebsp_=1; _ce.s=v~ae680f253ff5c34e22cdf9462ff420b9f2164389~lcw~1726132287862~lva~1726131074131~vpv~0~v11.cs~156325~v11.s~2ab9f8a0-70e4-11ef-be13-1399c07553bb~v11.sla~1726132370847~lcw~1726132370847; _ga=GA1.1.591917809.1724148341; ncbi_pinger=N4IgDgTgpgbg+mAFgSwCYgFwgMwCZcDCuALAIIBsAHACICs5u1AYrQAzuvYCcptljXAEJcCrAHQBGMQFs4uEABoQAVwB2AGwD2AQ1SqoADwAumUPKxhpAYwC0AMwibVRqKvRLsmcNcUhiX7QgjZCt1KF9aLwB6QODQqABnKIAFAFkCSn58XFoo3wl/LGgjCGRYKHcQXFYvPEISChp6RhYOTh4+AWFRSRk5X1wJL0tbBycXNwwRjFiQsIwYoLnElPTM7Jy8pXwvAHd9sVUrACNkQ/VpQ+REMQBzTRgBri8JbHJyX2warAB2Kk+hlhXu9PuYQD9sJRPp4sHZtOoEuEPIUQCVlEicFCsJ9nlhMtgChJaEMlMRviBMsRsLQflwaqSYSBxGSxJ5SSi1FpdPpjL5iJEsPSQLRGa8chEUUShfQXpkPkoaV4heRAeCviAAL4aoA=; _ga_DP2X732JSX=GS1.1.1726301792.21.0.1726301792.0.0.0; _ga_CSLL4ZEK4L=GS1.1.1726301792.21.0.1726301792.0.0.0",
        "Priority": "u=0, i",
        "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Opera GX";v="112"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "Windows",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.226 Safari/537.36",
    }

    response = requests.get(url, headers=headers)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract and display the HTML string nicely
    html_string = soup.prettify()

    # # Print the HTML content
    # pprint.pprint(html_string)
    
    raw_text = ""
    # # Print the extracted content
    # for num in range(1, 20): # 20 just random number. lets just assume the pubmed full article sections don't reach or surpass that
    #     # Extract the div with id="sec1"
    #     sec_div = soup.find("div", id=f"sec{num}")
    #     if sec_div:
    #         # decompose table and img
    #         for table in sec_div.find_all("table"):
    #             table.decompose()
    #         # Find and remove all tags with class name "table-wrap"
    #         for tag in sec_div.find_all(class_="table-wrap"):
    #             tag.decompose()
    #         for img in sec_div.find_all("img"):
    #             img.decompose()
    #         for tag in sec_div.find_all(class_="fig"):
    #             tag.decompose()
            
    #         title = sec_div.find(id=f"sec{num}title").get_text()
    #         raw_text += title + "\n"
    #         raw_text += sec_div.get_text().replace(title, "") + "\n\n"
    #         print()
    #         print()
    #     else:
    #         break
    
    sections = soup.find_all(class_="tsec sec")
    for sec_div in sections:
        # decompose table and img
        for table in sec_div.find_all("table"):
            table.decompose()
        # Find and remove all tags with class name "table-wrap"
        for tag in sec_div.find_all(class_="table-wrap"):
            tag.decompose()
        for img in sec_div.find_all("img"):
            img.decompose()
        for tag in sec_div.find_all(class_="fig"):
            tag.decompose()
        
        title = sec_div.find(class_="head").get_text()
        raw_text += title + "\n"
        if title.lower() != "references":
            raw_text += sec_div.get_text().replace(title, "") + "\n\n"
        else:
            ref_sec = sec_div.find_all(class_="ref-cit-blk")
            for ref in ref_sec:
                raw_text += ref.get_text() + "\n"
    
    return raw_text

def scrap_article_pdf_original_publisher(pmcid="PMC8822225"):
    # convert pmcid to pmid
    # from: https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/
    url_converter = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=rinogrego1212@gmail.com&ids={pmcid}&format=json"
    response = requests.get(url_converter)
    text = json.loads(response.text)
    if response.status_code == 200:
        print("doi      :", text["records"][0]["doi"])
        print("pmid     :", text["records"][0]["pmid"])
        print("pmcid    :", text["records"][0]["pmcid"])
        pmid = text["records"][0]["pmid"]
    else:
        return "No PMID Retrieved"
    
    # retrieve pdf if available
    from metapub import FindIt
    src = FindIt(pmid)
    
    # src.pma contains the PubMedArticle
    print("Title    :", src.pma.title)
    print("Abstract :", src.pma.abstract)

    # URL, if available, will be fulltext PDF
    if src.url:
        # insert your downloader of choice here, e.g. requests.get(url)
        print("URL      :", src.url)
    else:
        # if no URL, reason is one of "PAYWALL", "TXERROR", or "NOFORMAT"
        print("Reason   :", src.reason)
        return "Failed to download the PDF directly"
    
    # use a PDF reader to extract the fulltext from here.
    # Send the GET request
    response = requests.get(src.url)
    
    # parse title
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip().replace(" ", "_")
        filename = f"downloads/{title}.pdf"
    except:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"downloads/{timestamp}.pdf"

    # Save the PDF to a file
    with open(filename, "wb") as file:
        file.write(response.content)
    
    
    raw_txt = ""
    return raw_txt


if __name__ == "__main__":
    
    # raw_text = scrap_article()
    
    pmcid = "PMC1790863"
    pmcid = "PMC7442218"
    # pmcid = "PMC8822225"
    raw_text = scrap_article_bs4(pmcid=pmcid)
        
    print(raw_text)