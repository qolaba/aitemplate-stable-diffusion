from diffusers import EulerDiscreteScheduler
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
from typing import List, Optional, Union
import io,os
import zipfile
import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import subprocess as sp
import gc
import numpy as np
from multiprocessing import Process, Queue, Array
import multiprocessing as mp

mp.set_start_method('forkserver', force=True)

def gen_img(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size,q):
    new_work_dir= "tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size)+'/'

    StableDiffusionAITPipeline.workdir=new_work_dir
    model_id = "stabilityai/stable-diffusion-2-1"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe=StableDiffusionAITPipeline.from_pretrained(
                        model_id,
                        scheduler=scheduler,
                        revision="fp16",
                        torch_dtype=torch.float16,
                        use_auth_token="").to("cuda")
    with torch.autocast("cuda"):
        image = pipe(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt).images
        q.put(image)
    #os.system('sudo sh -c \'echo 3 >  /proc/sys/vm/drop_caches\'')


class aitemplate_sd:
    def __init__(self):
        pass
        # self.pipe_nsd = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        # self.pipe_nsd.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_nsd.scheduler.config)
        # self.pipe_nsd = self.pipe_nsd.to("cuda")


    def get_image(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size):
        q = Queue()
        process = Process(target=gen_img, args=(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size,q))
        process.start()
        image=q.get()
        #process.join()
        
        return image


        

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    
    def getlistoftmp(self):
        listdir=[i for i in os.listdir('.') if 'tmp_' in i]
        return listdir
    
    def get_image_buffer(self,image):
        img_buffer = io.BytesIO()
        image.save(img_buffer, 'PNG')
        img_buffer.seek(0)
        
        return img_buffer

    def get_zip_buffer(self,list_of_tuples):
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, data in list_of_tuples:
                zip_file.writestr(file_name, data.read())

        zip_buffer.seek(0)
        return zip_buffer
    
    





