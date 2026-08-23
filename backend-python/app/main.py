from fastapi import FastAPI
from app.routers import farmer

app = FastAPI(
    title="ResidueLink API",
    description="Backend for the Stubble-to-Biomass Marketplace",
    version="1.0.0"
)

app.include_router(farmer.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the ResidueLink API. ML Models are active."}