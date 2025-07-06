from pymongo import MongoClient
from datetime import datetime

# MongoDB connection string
MONGO_URI = "mongodb+srv://sandeshvarma:sandeshvarma@cluster0.mx0wqlt.mongodb.net/"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)

# Create/Get database
db = client['sentiment_analysis_db']

# Create/Get collections
users_collection = db['users']
searches_collection = db['searches']

def create_user_indexes():
    # Create unique index on email
    users_collection.create_index("email", unique=True)
    # Create index on username
    users_collection.create_index("username", unique=True)

def create_search_indexes():
    # Create index on user_id for faster queries
    searches_collection.create_index("user_id")
    # Create index on timestamp for time-based queries
    searches_collection.create_index("timestamp")

# Initialize indexes
create_user_indexes()
create_search_indexes() 