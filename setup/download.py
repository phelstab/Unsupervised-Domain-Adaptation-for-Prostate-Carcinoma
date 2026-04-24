"""
Source: https://pi-cai.grand-challenge.org
Source: https://zenodo.org/records/6624726
"""
# %%
from tqdm import tqdm
import requests
import os
import zipfile


#%%

current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
download_dir = os.path.join(current_dir, 'input/images')
os.makedirs(download_dir, exist_ok=True)

headers = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
  'Accept-Language': 'en-US,en;q=0.9',
  'Connection': 'keep-alive',
  'Referer': 'https://zenodo.org/records/6624726',
  'Sec-Fetch-Dest': 'document',
  'Sec-Fetch-Mode': 'navigate', 
  'Sec-Fetch-Site': 'same-origin',
  'Sec-Fetch-User': '?1',
  'Upgrade-Insecure-Requests': '1',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
}

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        mininterval=1.0
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_and_extract_picai():
    """Download and extract the PI-CAI Challenge dataset"""
    # First get the dataset metadata
    api_url = 'https://zenodo.org/api/records/6624726'
    print("Fetching dataset metadata...")
    response = requests.get(api_url)
    data = response.json()
    
    # Download each file
    for file_info in data.get('files', []):
        filename = file_info.get('key')
        download_url = file_info.get('links', {}).get('self')
        
        if not download_url:
            continue
            
        output_path = os.path.join(download_dir, filename)
        
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping download")
            continue
            
        print(f"Downloading {filename}...")
        download_file(download_url, output_path)
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(download_dir, os.path.splitext(filename)[0]))
    
    print("PI-CAI dataset download completed")

# %%
download_and_extract_picai()

# %%
