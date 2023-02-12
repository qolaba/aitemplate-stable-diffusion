from fastapi import FastAPI, Request
import typing
import asyncio
import time 
import logging 

Scope = typing.MutableMapping[str, typing.Any]
Message = typing.MutableMapping[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Message], typing.Awaitable[None]]
RequestTuple = typing.Tuple[Scope, Receive, Send]

logger = logging.getLogger("uvicorn")

async def very_heavy_lifting(requests: dict[int,RequestTuple], batch_no) -> dict[int, RequestTuple]:
    #This mimics a heavy lifting function, takes a whole 3 seconds to process this batch
    logger.info(f"Heavy lifting for batch {batch_no} with {len(requests.keys())} requests")
    await asyncio.sleep(3)
    processed_requests: dict[int,RequestTuple] = {}
    for id, request in requests.items():
        request[0]["heavy_lifting_result"] = f"result of request {id} in batch {batch_no}"
        processed_requests[id] = (request[0], request[1], request[2])
    return processed_requests

class Batcher():
    def __init__(self, batch_max_size: int = 5, batch_max_seconds: int = 3) -> None:
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
                    logger.info(f"Batch {self.batch_no} is full \
                        (requests: {len(self.to_process.keys())}, max allowed: {self.batch_max_size})")
                    self.batch_no += 1
                    await self.process_requests(self.batch_no)

                    break
                await asyncio.sleep(0)
            else:
                if len(self.to_process)>0:
                    logger.info(f"Batch {self.batch_no} is over timelimit (requests: {len(self.to_process.keys())})")
                    self.batch_no += 1
                    await self.process_requests(self.batch_no)
            await asyncio.sleep(0)

    async def process_requests(self, batch_no: int):
        logger.info(f"Start of processing batch {batch_no}...")
        for id, request in self.to_process.items():
            self.processing[id] = request
        self.to_process = {}
        processed_requests  = await very_heavy_lifting(self.processing, batch_no)
        self.processed = processed_requests
        self.processing = {}
        logger.info(f"Finished processing batch {batch_no}")

batcher = Batcher() 

class InterceptorMiddleware():
    def __init__(self, app) -> None:
        self.app = app
        self.request_id: int = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        self.request_id += 1
        current_id = self.request_id
        batcher.to_process[self.request_id] = (scope, receive, send)
        logger.info(f"Added request {current_id} to batch {batcher.batch_no}.")
        while True:
            request = batcher.processed.get(current_id, None)
            if not request:
                await asyncio.sleep(0.5)
            else:
                logger.info(f"Request {current_id} was processed, forwarding to FastAPI endpoint..")
                batcher.processed.pop(current_id)
                await self.app(request[0], request[1], request[2])
                await asyncio.sleep(0)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    batcher.start_batcher()
    return

app.add_middleware(InterceptorMiddleware)

@app.get("/")
async def root(request: Request):
    return {"Return value": request["heavy_lifting_result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)