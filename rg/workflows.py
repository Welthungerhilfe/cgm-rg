import anyio
import asyncer
from asyncer import asyncify

from utils.inference import call_face_api, call_pose_api


async def run_pose_workflow(session, encoded_images):
    async with asyncer.create_task_group() as task_group:
        soon_values = [task_group.soonify(call_pose_api)(session, image) for image in encoded_images]
    return [soon.value for soon in soon_values]


async def run_face_workflow(session, encoded_images):
    async with asyncer.create_task_group() as task_group:
        soon_values = [task_group.soonify(call_face_api)(session, image) for image in encoded_images]
    return [soon.value for soon in soon_values]
