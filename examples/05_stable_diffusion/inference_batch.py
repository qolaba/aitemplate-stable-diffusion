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

Scope = typing.MutableMapping[str, typing.Any]
Message = typing.MutableMapping[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Message], typing.Awaitable[None]]
RequestTuple = typing.Tuple[Scope, Receive, Send]


async def very_heavy_lifting(requests: dict[int,RequestTuple], batch_no) -> dict[int, RequestTuple]:
    processed_requests: dict[int,RequestTuple] = {}
    request_check=[]
    prompt_list=[]
    neg_prompt_list=[]
    first_Request=[]
    for id, request in requests.items():
        params=['prompt', 'height', 'width', 'num_inference_steps', 'guidance_scale', 'negative_prompt']
        params_check=[x.lower() in request[0]['query_string'].decode().lower() for x in params]
        if(not all(params_check)):
            request[0]["result"] = "invalid data"
            request_check.append(False)
        else:
            request[0]["query"]=[i.split("=") for i in urllib.parse.unquote(request[0]['query_string']).split("&")]
            request[0]["query"]=[i[1] for i in request[0]["query"]]
            if(first_Request==[]):
                first_Request=request[0]["query"]
                request_check.append(True)
                prompt_list.append(request[0]["query"][0])
                neg_prompt_list.append(request[0]["query"][-1])
            else:
                if(first_Request[1:-1]==request[0]["query"][1:-1]):
                    request_check.append(True)
                    prompt_list.append(request[0]["query"][0])
                    neg_prompt_list.append(request[0]["query"][-1])
                else:
                    request_check.append(False)
    samebatch=False
    if(all(request_check)==True):
        samebatch=True
    if(len(requests)!=5 or samebatch==False):
        listdir=helper.getlistoftmp()
        for id, request in requests.items():
            if("query" in request[0]):
                if(not ("tmp_"+str(request[0]["query"][2])+"_"+str(request[0]["query"][1])+"_"+str(1) in listdir)):
                    request[0]["result"] = "invalid data"
                else:
                    pipe = helper.get_pipe(int(request[0]["query"][1]),int(request[0]["query"][2]),1)
                
                    image = pipe(request[0]["query"][0],int(request[0]["query"][1]),int(request[0]["query"][2]),int(request[0]["query"][3]),float(request[0]["query"][4]),request[0]["query"][5]).images
                    request[0]["result"] = image[0]
            else:
                request[0]["result"] = 'invalid data'
            processed_requests[id] = (request[0], request[1], request[2])
    else:
        
        image = helper.pipe_512_512_5(prompt_list,int(first_Request[1]),int(first_Request[2]),int(first_Request[3]),float(first_Request[4]),neg_prompt_list).images
        i=0
        for id, request in requests.items():                
            request[0]["result"] = image[i]
            processed_requests[id] = (request[0], request[1], request[2])
            i=i+1
    return processed_requests

class Batcher():
    def __init__(self, batch_max_size: int = 5, batch_max_seconds: int = 0.0001) -> None:
        self.batch_max_size = batch_max_size
        self.batch_max_seconds = batch_max_seconds
        self.to_process: dict[int, RequestTuple] = {}
        self.processing: dict[int, RequestTuple] = {}
        self.processed: dict[int, RequestTuple] = {}
        self.batch_no = 1

    def start_batcher(self):
        _ = asyncio.get_event_loop()
        self.batcher_task = asyncio.create_task(self._batcher())

    async def _batcher(self):
        while True:
            time_out = time.time() + self.batch_max_seconds
            while time.time() < time_out:
                if len(self.to_process) >= self.batch_max_size:
                    self.batch_no += 1
                    await self.process_requests(self.batch_no)

                    break
                await asyncio.sleep(0)
            else:
                if len(self.to_process)>0:
                    self.batch_no += 1
                    await self.process_requests(self.batch_no)
            await asyncio.sleep(0)

    async def process_requests(self, batch_no: int):
        for id, request in self.to_process.items():
            self.processing[id] = request
        self.to_process = {}
        processed_requests  = await very_heavy_lifting(self.processing, batch_no)
        self.processed = processed_requests
        self.processing = {}

batcher = Batcher() 

class InterceptorMiddleware():
    def __init__(self, app) -> None:
        self.app = app
        self.request_id: int = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return
        if(scope['path']=='/compile_model'):
            await self.app(scope, receive, send)
            return 
        if(scope['path']=='/list_of_configuration'):
            await self.app(scope, receive, send)
            return 

        self.request_id += 1
        current_id = self.request_id
        batcher.to_process[self.request_id] = (scope, receive, send)
        while True:
            request = batcher.processed.get(current_id, None)
            if not request:
                await asyncio.sleep(0)
            else:
                batcher.processed.pop(current_id)
                await self.app(request[0], request[1], request[2])
                await asyncio.sleep(0)



app = FastAPI()

#app.POOL: ThreadPoolExecutor = None

helper=aitemplate_sd()




@app.on_event("startup")
def startup_event():
    #app.POOL = ThreadPoolExecutor(max_workers=4)
    batcher.start_batcher()
    return

app.add_middleware(InterceptorMiddleware)


# @app.on_event("shutdown")
# def shutdown_event():
#     app.POOL.shutdown(wait=False)



@app.get("/getimage")
def get_image(request: Request,
    #prompt: Union[str, List[str]],
    prompt: Optional[str] = "Cute dog",
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = " ",):
    if(request["result"]=="invalid data"):
        return "Either this configuration is not available or data is invalid"
    else:
        
        filtered_image = io.BytesIO()
        request["result"].save(filtered_image, "PNG")
        filtered_image.seek(0)
        #img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return StreamingResponse(filtered_image, media_type="image/jpeg")
        #return img_str
    #return {"Return value": request["heavy_lifting_result"]}
    


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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    











