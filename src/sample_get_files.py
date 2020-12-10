import requests


def get_token():
    '''
    Retuns a session token used to authenticate to the api
    '''
    # Make the call to AD with resource query param
    headers = {
        'Metadata': 'true',
    }

    params = (
        ('api-version', '2018-02-01'),
        ('resource', '$resource'),
    )

    response = requests.get(
        'http://169.254.169.254/metadata/identity/oauth2/token',
        headers=headers,
        params=params)

    token_json = response.json()
    access_token = token_json['access_token']

    # Validate the token against the API to get a session token
    data = {"access_token": token}
    data = str(data)
    response = requests.post(
        'https://cgm-be-ci-dev-scanner-api.azurewebsites.net/.auth/login/aad',
        data=data)
    session_token_json = response.json()
    session_token = session_token_json['authenticationToken']

    print(session_token_json)
    print("\nAuthentication Token: ", session_token)

    return session_token


def get_auth_headers():
    session_token = get_token()
    headers = {'X-ZUMO-AUTH': session_token}
    return headers


if __name__ == "__main__":
    url = "http://localhost:5001"
    file_endpoint = '/api/scan/files'

    file_id = '5850e04c-33e1-11eb-af63-4f5622046249'

    image_path = 'sample_image.jpeg'

    response = requests.get(
        'http://localhost:5001/api/scan/files/1dd223a8-34b9-11eb-9913-4b06875f87c3')

    print("\nStatus code: ", response.status_code)
    print("\nContent : ", response.content)
    print("\nResponse raw: ", response.raw)

    with open(image_path, 'wb') as f:
        f.write(response.content)

    # print("Success")
