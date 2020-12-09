import os
import requests

headers = {}

path = 'tests/rgb_1583438117-71v1y4z0gd_1592711198959_101_74947.76209955901.jpg'

type_ = 'image/jpeg'

files = {}
files['file'] = (open(path, 'rb'), type_)
files['filename'] = path.split('/')[-1]

print('\nFile name to post : ', files['filename'])
headers['content_type'] = 'multipart/form-data'
#headers['content-type'] = 'multipart/form-data'
#headers['Content-Type'] = 'multipart/form-data'

print(headers)

url = "http://localhost:5001"
file_endpoint = '/api/scan/files'


# print(files)
response = requests.post(url + file_endpoint, files=files, headers=headers)


#response = requests.post(url + file_endpoint, data=files, headers=headers)

print("\nResponse status code: ", response.status_code)

file_id = response.content.decode('utf-8')
print("\nFile Id from post of test.jpg: ", file_id, '\n')
