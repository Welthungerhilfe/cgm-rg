import json
import requests
import shutil

url = "http://localhost:5001"
file_endpoint = '/api/scan/files'

file_id = '5850e04c-33e1-11eb-af63-4f5622046249'

#response = requests.get(url+file_endpoint+ file_id)


image_path = 'sample_image.jpeg'


response = requests.get('http://localhost:5001/api/scan/files/1dd223a8-34b9-11eb-9913-4b06875f87c3')



#response = requests.get('http://localhost:5001/api/scan/files/5850e04c-33e1-11eb-af63-4f5622046249')
#response = requests.get('http://localhost:5001/api/scan/files/5850e04c-33e1-11eb-af63-4f5622046249', stream=True)


print("\nStatus code: ", response.status_code)
print("\nContent : ", response.content)

print("\nResponse raw: ", response.raw)

#print("\n Response json : ", response.json())

#print("\nResponse data: ", response.data)
#print("\nResponse file: ", response.file)

#with open(image_path, 'wb') as out_file:
#    shutil.copyfileobj(response.raw, out_file)


'''
if response.status_code == 200:
    with open(image_path, 'wb') as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)    
'''

with open(image_path, 'wb') as f:
    f.write(response.content)

#print("Success")