from pymilvus import connections, FieldSchema, CollectionSchema, Collection, utility

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema
fields = [
    FieldSchema(name="id", dtype="INT64", is_primary=True),
    FieldSchema(
        name="embedding", dtype="FLOAT_VECTOR", dim=768
    ),  # Adjust dim as per model output
]
schema = CollectionSchema(fields, description="Sample RAG collection")

# Create a collection
collection_name = "rag_collection"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)


from googleapiclient.discovery import build

# Initialize the Gemini API client
gemini_api = build("gemini", "v1", developerKey="YOUR_API_KEY")
