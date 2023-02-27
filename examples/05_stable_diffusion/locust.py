from locust import HttpUser, task
import random
import urllib

class HelloWorldUser(HttpUser):
    host="http://127.0.0.1:9000"
    @task(1)

    def hello_world(self):
        h_list=[512,768,1024]
        height=random.sample(h_list, 1)
        url="/getimage?prompt=dog&height="+str(height[0])+"&width="+str(height[0])+"&num_inference_steps=50&guidance_scale=7.5&negative_prompt=%20"
        b=urllib.parse.quote_plus(url)
        self.client.post(url)
