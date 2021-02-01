import requests 

ENDPOINT_URL = 'http://localhost:5000/infer2'
 
def infer(): 
    img_path = 'cat.1.jpg'
    data = { 'img_path': img_path } 
    response = requests.post(ENDPOINT_URL, json = data) 
    response.raise_for_status() 
    print(response.content) 
  
if __name__ =="__main__":
    infer() 
