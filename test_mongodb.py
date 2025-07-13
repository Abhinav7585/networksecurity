
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://abhinavreddy1804:Mar1234@cluster0.1mwqlpj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)



# The above code is used to connect to MongoDB database
