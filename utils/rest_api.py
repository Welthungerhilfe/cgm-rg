from os import getenv
from typing import Dict
import datetime

import anyio
import aiohttp
from utils.retry_decorator import retry


files_api_semaphore = anyio.Semaphore(10)


class RestApi():
    def __init__(self, base_url: str, api_key: str):
        self.url = base_url
        self.headers = {'X-API-Key': api_key}
        self.session = aiohttp.ClientSession()

    async def get_json(self, path: str, params: Dict | None = None):
        async with self.session.get(self.url + path, headers=self.headers, params= params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data, resp.status

    async def get_binary(self, path: str):
        async with self.session.get(self.url + path, headers=self.headers) as resp:
            resp.raise_for_status()
            data = await resp.read()
            return data, resp.status

    async def post_json(self, path: str, json: Dict | None = None):
        async with self.session.post(self.url + path, headers=self.headers, json=json) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data, resp.status

    async def post_binary(self, path, files):
        async with self.session.post(self.url + path, headers=self.headers, data=files) as resp:
            resp.raise_for_status()
            data = await resp.read()
            return data.decode('utf-8'), resp.status

    async def put_json(self, path: str, json: Dict | None = None):
        async with self.session.put(self.url + path, headers=self.headers, json=json) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data, resp.status


class CgmApi(RestApi):
    def __init__(self):
        super().__init__(getenv("APP_URL"),getenv("API_KEY"))

    @retry(retries=3, delay=2)
    async def get_files(self, file_id):
        async with files_api_semaphore:
            return await self.get_binary(f"/api/files/{file_id}")

    async def get_basic_person_info(self, person_id):
        basic_info, status_code = await self.get_json(f'/api/persons/{person_id}/basic')
        return basic_info['date_of_birth'], basic_info['sex']

    async def get_manual_measures(self, person_id, scan_date):
        us_manual_measures, status_code = await self.get_json(f'/api/persons/{person_id}/measurement')
        manual_measures = sorted(us_manual_measures['measurements'], key=lambda m: m['measured'])

        mm_keys_keys = ['height', 'weight', 'muac', 'head_cir', 'oedema' ,'location']
        mms = []
        for mm in manual_measures:
            if scan_date == datetime.datetime.strptime(mm['measured'], '%Y-%m-%dT%H:%M:%SZ').date():
                target_dict = dict((k, mm[k]) for k in mm_keys_keys if k in mm)
                mms.append(target_dict)
        return mms

    @retry(retries=3, delay=2)
    async def get_scan_metadata(self, scan_id):
        scan_metadata, status_code = await self.get_json(f"/api/scans/{scan_id}")
        return scan_metadata['scan']

    @retry(retries=3, delay=2)
    async def get_workflows(self):
        workflows, status_code = await self.get_json('/api/workflows')
        return workflows['workflows']

    @retry(retries=3, delay=2)
    async def post_results(self, results):
        data, status_code = await self.post_json('/api/results', json=results)
        return status_code

    @retry(retries=3, delay=2)
    async def post_files(self, bin_file, file_format) -> str:
        async with files_api_semaphore:
            if file_format == 'rgb':
                filename = 'test.jpg'
            elif file_format == 'depth':
                filename = 'test.depth'
            else:
                raise ValueError(f"Unrecognized file format: {file_format}.")

            files = {
                'file': bin_file,
                'filename': filename
            }
            file_id, status_code = await self.post_binary('/api/files', files)
            if status_code != 201:
                print(f"file upload failed with {status_code}")
            return file_id

    async def get_workflow_id(self, workflow_name, workflow_version):
        workflows = await self.get_workflows()
        workflow = [workflow for workflow in workflows if workflow['name'] == workflow_name and workflow['version'] == workflow_version]

        return workflow[0]['id']

    async def get_results_for_workflow(self, scan_id, workflow_id):
        params = {
            'scan_id': scan_id,
            'workflow_id': workflow_id,
            'show_results': True
        }

        results, status_code = await self.get_json('/api/scans', params=params)
        return results['scans'][0]['results']

    @retry(retries=3, delay=2)
    async def put_child_visit_result(self, child_visit_id, result_data):
        results, status_code = await self.put_json(f'/api/child-visits/{child_visit_id}/result-data', json={"result_data": result_data})
        return status_code
