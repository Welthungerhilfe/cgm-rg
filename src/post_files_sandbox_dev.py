import requests

headers = {}

resource = "https%3A%2F%2Fcgmb2csandbox.onmicrosoft.com%2F98e9e1be-53fb-47f4-b53a-5842aeb869d5"

headers = {
    'Metadata': 'true',
}

response_one = requests.get(
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=' +
    resource,
    headers=headers)
print("\nresponse_one status code: ", response_one.status_code)

token = response_one.json()
print("\ntoken : ", token)

access_token = token['access_token']
print("\naccess_token: ", access_token)

data = {"access_token": access_token}

response_two = requests.post(
    'https://cgm-be-ci-dev-scanner-api.azurewebsites.net/.auth/login/aad',
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
    'https://cgm-be-ci-dev-scanner-api.azurewebsites.net/api/scan/openapi.json',
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
# headers['content-type'] = 'multipart/form-data'
# headers['Content-Type'] = 'multipart/form-data'


# url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
# file_endpoint = '/api/scan/files'


# content_type = 'multipart/form-data'

post_file_response = requests.post(
    'https://cgm-be-ci-dev-scanner-api.azurewebsites.net/api/scan/files',
    files=files,
    headers=headers)
# post_file_response = requests.post('https://cgm-be-ci-dev-scanner-api.azurewebsites.net/api/scan/files', data=files, headers=headers)
# post_file_response = requests.post('https://cgm-be-ci-dev-scanner-api.azurewebsites.net/api/scan/files', content_type=content_type, files=files, headers=headers)


print(r"\post_file_response status code: ", post_file_response.status_code)
print("\nPost File response\n")
print(post_file_response.content)
# print(post_file_response.json())

# print("\nResponse status code: ", post_file_response.status_code)

file_id = post_file_response.content.decode('utf-8')
print("\nFile Id from post of test.jpg: ", file_id, '\n')
