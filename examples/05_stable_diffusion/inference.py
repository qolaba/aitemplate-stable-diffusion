from fastapi import FastAPI,BackgroundTasks, Request
from typing import List, Optional, Union
import io
from fastapi.responses import StreamingResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from compile import compile_diffusers
import time
from aithelper import aitemplate_sd
import typing
import asyncio
import logging 
import urllib.parse
import threading 
import gc
import torch

lock = threading.Lock()


app = FastAPI()
helper=aitemplate_sd()
app.POOL: ThreadPoolExecutor = None


@app.on_event("startup")
def startup_event():
    app.POOL = ThreadPoolExecutor(max_workers=1)

@app.on_event("shutdown")
def shutdown_event():
    app.POOL.shutdown(wait=False)



@app.post("/getimage")
def get_image(
    #prompt: Union[str, List[str]],
    prompt: Optional[str] = "dog",
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = None,):

    if(type(prompt)==str):
        batch_size=1
    else:
        batch_size=len(prompt)
        
    listdir=helper.getlistoftmp()
    if(not ("tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size) in listdir)):
        return {"msg" : "you need to compile this configuration",
                "height" : height,
                "width" : width,
                "batch_size" : batch_size}
    else:
        lock.acquire()
        
        pipe = helper.get_pipe(height, width, batch_size)
        lock.release()
        
        image = app.POOL.submit(pipe,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt).result().images
        # if('pipe' in locals()):
        #     del pipe
        #     gc.collect()
        #     torch.cuda.empty_cache()
        
        
        
        if(len(image)==1):
            filtered_image = io.BytesIO()
            image[0].save(filtered_image, "JPEG")
            filtered_image.seek(0)
            return StreamingResponse(filtered_image, media_type="image/jpeg")
        else:
            list_of_tuples=[]
            for i in range(0,len(image)):
                list_of_tuples.append((str(i)+'.png', helper.get_image_buffer(image[i])))
            buff=helper.get_zip_buffer(list_of_tuples) 
            response = StreamingResponse(buff, media_type="application/zip")
            response.headers["Content-Disposition"] = "attachment; filename=images.zip"
            return response


@app.get("/compile_model")
def compile_model(
    height: int,
    width: int,
    batch_size: int,
    background_tasks: BackgroundTasks):

    listdir=helper.getlistoftmp()
    if("tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size) in listdir):
        return {"msg" : "tmp folder is already available",
                "height" : height,
                "width" : width,
                "batch_size" : batch_size}


    background_tasks.add_task(compile_diffusers,None, width, height, batch_size)

    return {"msg" : "Task_Submitted",
                "height" : height,
                "width" : width,
                "batch_size" : batch_size}

@app.get("/list_of_configuration")
def list_of_configuration():
    listdir=helper.getlistoftmp()
    jsondata={}
    for i in range(0,len(listdir)):
        data=listdir[i].split("_")
        jsondata[i]={"height" : data[2],
            "width" : data[1],
            "batch_size" : data[3]}
    return jsondata


@app.post("/getimage_nsd")
def get_image_nsd(
    #prompt: Union[str, List[str]],
    prompt: Optional[str] = "dog",
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = None,):

    if(type(prompt)==str):
        batch_size=1
    else:
        batch_size=len(prompt)
        

    image = app.POOL.submit(helper.pipe_nsd,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt).result().images
    if(len(image)==1):
        filtered_image = io.BytesIO()
        image[0].save(filtered_image, "JPEG")
        filtered_image.seek(0)
        return StreamingResponse(filtered_image, media_type="image/jpeg")
    else:
        list_of_tuples=[]
        for i in range(0,len(image)):
            list_of_tuples.append((str(i)+'.png', helper.get_image_buffer(image[i])))
        buff=helper.get_zip_buffer(list_of_tuples) 
        response = StreamingResponse(buff, media_type="application/zip")
        response.headers["Content-Disposition"] = "attachment; filename=images.zip"
        return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)    