import hashlib


def get_library_hash(file_list=None):
    if file_list is None:
        file_list = ['ibeoCSVImporter.py', 'SequenceWrangler.py']
    hash_value = 0
    for full_path in file_list:
        hash_value += abs(hash(hashlib.md5(open(full_path, 'rb').read()).hexdigest()))
    return hash_value
