def unpickle(file, encoding='bytes'):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding=encoding)
    return data_dict

def download_and_extract(url, root='data'):
    import requests, tarfile, os, gzip
    local_filename = url.split('/')[-1] 
    file_path = os.path.join(root, local_filename)

    if os.path.exists(file_path): return
    os.makedirs(root, exist_ok=True)

    print(f'Downloading {url} to {file_path}...')
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)

    if file_path.endswith('tar.gz'):
        with tarfile.open(file_path) as tar:
            tar.extractall(path=root)
            file_path = tar.name
    elif file_path.endswith('gz'):
        with open(file_path[:-3], 'wb') as f:
            f.write(gzip.decompress(r.content))