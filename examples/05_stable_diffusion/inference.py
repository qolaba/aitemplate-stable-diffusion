from fastapi import FastAPI,BackgroundTasks
import click
import torch
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from diffusers import EulerDiscreteScheduler
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
from typing import List, Optional, Union
import io,os
from fastapi.responses import StreamingResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from compile import compile_diffusers
import zipfile

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()
model_id = "stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

app.POOL: ThreadPoolExecutor = None

def getlistoftmp():
    listdir=[i for i in os.listdir('.') if 'tmp_' in i]
    return listdir

def get_image_buffer(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, 'PNG')
    img_buffer.seek(0)
    
    return img_buffer

def get_zip_buffer(list_of_tuples):
    zip_buffer = io.BytesIO()
    
    # https://stackoverflow.com/a/44946732 <3   
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in list_of_tuples:
            zip_file.writestr(file_name, data.read())

    zip_buffer.seek(0)
    return zip_buffer

StableDiffusionAITPipeline.workdir="tmp_512_512_1/"
pipe = StableDiffusionAITPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token="").to("cuda")



    # FastAPI will automatically parse the HTTP request for us.
@app.on_event("startup")
def startup_event():
    app.POOL = ThreadPoolExecutor(max_workers=5)

@app.on_event("shutdown")
def shutdown_event():
    app.POOL.shutdown(wait=False)



@app.get("/getimage")
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

    global pipe
        
    listdir=getlistoftmp()
    if(not ("tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size) in listdir)):
        return {"msg" : "you need to compile this configuration",
                "height" : height,
                "width" : width,
                "batch_size" : batch_size}
    elif(not ("tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size)+'/'==StableDiffusionAITPipeline.workdir)):
        print("tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size),StableDiffusionAITPipeline.workdir)
        StableDiffusionAITPipeline.workdir="tmp_"+str(width)+"_"+str(height)+"_"+str(batch_size)
        pipe = StableDiffusionAITPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token="").to("cuda")


    image = app.POOL.submit(pipe,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt).result().images
    if(len(image)==1):
        filtered_image = io.BytesIO()
        image[0].save(filtered_image, "JPEG")
        filtered_image.seek(0)
        return StreamingResponse(filtered_image, media_type="image/jpeg")
    else:
        list_of_tuples=[]
        for i in range(0,len(image)):
            list_of_tuples.append((str(i)+'.png', get_image_buffer(image[i])))
        buff=get_zip_buffer(list_of_tuples) 
        response = StreamingResponse(buff, media_type="application/zip")
        response.headers["Content-Disposition"] = "attachment; filename=images.zip"
        return response


@app.get("/compile_model")
def compile_model(
    height: int,
    width: int,
    batch_size: int,
    background_tasks: BackgroundTasks):

    listdir=getlistoftmp()
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
    listdir=getlistoftmp()
    jsondata={}
    for i in range(0,len(listdir)):
        data=listdir[i].split("_")
        jsondata[i]={"height" : data[2],
            "width" : data[1],
            "batch_size" : data[3]}
    return jsondata

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)    











