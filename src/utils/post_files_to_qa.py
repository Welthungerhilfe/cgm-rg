import requests

resource = "https%3A%2F%2Fcgm-be-ci-qa-scanner-api.azurewebsites.net"
url = 'https://cgm-be-ci-qa-scanner-api.azurewebsites.net'

headers = {
    'Metadata': 'true',
}

response_one = requests.get(
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=' + resource,
    headers=headers)

print("\nresponse_one status code: ", response_one.status_code)

token = response_one.json()
print("\ntoken : ", token)

access_token = token['access_token']
print("\naccess_token: ", access_token)

data = {"access_token": access_token}

response_two = requests.post(
    url + '/.auth/login/aad',
    json=data)
print("\response_two status code: ", response_two.status_code)

auth_token_json = response_two.json()
print("\nauth_token_json : ", auth_token_json)

auth_token = auth_token_json['authenticationToken']
print("\nauth_token: ", auth_token)

headers = {
    'X-ZUMO-AUTH': auth_token,
}

openapi_response = requests.get(
    url + '/api/scan/openapi.json',
    headers=headers)
print("\nOpenAPI response\n")
print(openapi_response.json())


path = '/app/tests/rgb_1583438117-71v1y4z0gd_1592711198959_101_74947.76209955901.jpg'
type_ = 'image/jpeg'

files = {}
files['file'] = (open(path, 'rb'), type_)
files['filename'] = path.split('/')[-1]

print('\nFile name to post : ', files['filename'])

headers['content_type'] = 'multipart/form-data'

post_file_response = requests.post(
    url + '/api/scan/files',
    files=files,
    headers=headers)

print("\npost_file_response status code: ", post_file_response.status_code)
print("\nPost File response\n")
print(post_file_response.content)

file_id = post_file_response.content.decode('utf-8')
print("\nFile Id from post of test.jpg: ", file_id, '\n')
