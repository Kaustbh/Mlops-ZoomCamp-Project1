import os.path
from zipfile import ZipFile

ZIP_FILE = '/home/kaustubh/mlops_zoomcamp/final_project/project_1/data/extended_crab_age_pred.zip'
RAW_FOLDER = '/home/kaustubh/mlops_zoomcamp/final_project/project_1/data'

def unzip_raw_data(zip_file=ZIP_FILE, extract_to=RAW_FOLDER):
    # Check if the zip file exists
    if os.path.isfile(zip_file):
        zipfile = ZipFile(zip_file)
        zipfile.extractall(path=extract_to)

# Call the function
# unzip_raw_data()

if __name__ == '__main__':
    unzip_raw_data()
    
