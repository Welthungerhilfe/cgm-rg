import requests


url = "http://localhost:5001"
file_endpoint = '/api/scan/files'

path = '/home/nikhil/cgm-all/cgm-rg/data/scans/59560ba2-33e1-11eb-af63-4b01606d9610/img/5850e04c-33e1-11eb-af63-4f5622046249_blur.jpg'

type_ = 'image/jpeg'
files = {}
files['file'] = (open(path, 'rb'), type_)
files['filename'] = path
#files['content_type'] ='multipart/form-data'
print('\nFile name to post : ', files['filename'])

#print(files)
response = requests.post(url + file_endpoint, files=files, headers={'content_type':'multipart/form-data'})
#response = requests.post(url + file_endpoint, data=files)

print("\nResponse status code: ", response.status_code)

file_id  = response.content.decode('utf-8')
print("\nFile Id from post of test.jpg: ", file_id, '\n')