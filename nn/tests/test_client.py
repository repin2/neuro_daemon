import requests
import os

pizza_path = os.path.join(os.path.dirname(__file__), 'tests_data', 'pizza.jpg')

with open(pizza_path, 'rb') as pizza_file:
    pizza_bytes = pizza_file.read()

response = requests.post('http://localhost:80', json={'img': pizza_bytes})


assert 1
