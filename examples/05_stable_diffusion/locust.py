from locust import HttpUser, task

class HelloWorldUser(HttpUser):
    host="http://127.0.0.1:8000"
    @task(1)
    def hello_world(self):
        self.client.post("/getimage")
