import fastapi
from models.src.generation.lstm import lstm_generate_description # Assuming you have this import
from fastapi.middleware.cors import CORSMiddleware # Import the middleware

app = fastapi.FastAPI()

# Add CORS middleware *before* defining routes
# Temporarily allow all origins for debugging
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the wildcard
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (including POST and OPTIONS)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/lstm_job_description")
async def lstm_job_description(request: fastapi.Request):
    data = await request.json()
    title = data["title"]
    description = lstm_generate_description(title)
    return {"description": description}
