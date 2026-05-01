"""
API routes for the prediction endpoint.
Main business logic for food recognition and recipe recommendation.
Bridged to Colab Microservice via ngrok.
"""

import base64
import requests
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PredictionRequestSchema, PredictionResponseSchema
from app.services.calorie_service import CalorieService
from app.services.recipe_service import RecipeService
from app.config import ML_CONFIG

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponseSchema)
async def predict_food(request: PredictionRequestSchema) -> PredictionResponseSchema:
    """
    Main prediction endpoint.
    Receives base64 image from React, forwards as multipart file to Sam's Colab tunnel,
    fetches nutritional info, and returns matching recipes.
    """
    try:
        # Step 1: Predict the food
        if request.food_name:
            # Deterministic testing mode (skips ML)
            predicted_food = request.food_name.lower()
            confidence = 1.0
        else:
            # LIVE ML MICROSERVICE MODE
            ngrok_url = ML_CONFIG.get("ngrok_url")
            
            if not ngrok_url:
                raise HTTPException(
                    status_code=500, 
                    detail="ML_NGROK_URL is missing from the .env file. Ask Sam for the active link!"
                )
                
            if not request.image_base64:
                raise HTTPException(
                    status_code=400, 
                    detail="No image or food_name provided in the request."
                )

            # Clean the base64 string if React sent a data URI prefix
            b64_data = request.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]

            # Decode the base64 string into raw image bytes
            try:
                image_bytes = base64.b64decode(b64_data)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 image data.")

            # Forward to Sam's Colab tunnel as a multipart/form-data file
            try:
                # requests automatically sets the multipart headers when using 'files'
                files = {"file": ("food_image.jpg", image_bytes, "image/jpeg")}
                
                # Sam specifically asked for a POST to [NGROK_URL]/predict
                colab_response = requests.post(
                    f"{ngrok_url}/predict", 
                    files=files, 
                    timeout=20  # Giving Colab 20 seconds to process the heavy YOLO/MiDaS models
                )
                colab_response.raise_for_status()
                
                # Parse Sam's JSON answer
                ml_data = colab_response.json()
                
                # Extract the food name (Fallback to 'apple' if the key name doesn't match exactly)
                predicted_food = ml_data.get("food", ml_data.get("class", "apple")).lower()
                confidence = float(ml_data.get("confidence", 0.95))

            except requests.exceptions.ConnectionError:
                raise HTTPException(
                    status_code=502, 
                    detail="Could not connect to the ML engine. Sam's ngrok tunnel might be offline or the URL changed."
                )
            except requests.exceptions.Timeout:
                raise HTTPException(
                    status_code=504, 
                    detail="The Colab ML engine took too long to respond. The T4 GPU might be overloaded."
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502, 
                    detail=f"Error reading from Colab microservice: {str(e)}"
                )

        # Step 2: Fetch food information from MongoDB
        food_info = CalorieService.get_food_by_name(predicted_food)
        
        if not food_info:
            raise HTTPException(
                status_code=404,
                detail=f"Food '{predicted_food}' not found in database"
            )
        
        # Step 3: Fetch and filter recipes
        matching_recipes = RecipeService.get_filtered_recipes(
            predicted_food,
            available_appliances=request.available_appliances if request.available_appliances else None,
            health_constraints=request.health_constraints if request.health_constraints else None
        )
        
        # Step 4: Build and return the final payload to React
        response = PredictionResponseSchema(
            detected_food=predicted_food,
            confidence=confidence,
            food_info=food_info,
            matching_recipes=matching_recipes
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction pipeline failed: {str(e)}"
        )


@router.get("/foods/{food_name}", response_model=dict)
async def get_food_info(food_name: str):
    """Get detailed information about a specific food."""
    try:
        food_info = CalorieService.get_food_by_name(food_name)
        if not food_info:
            raise HTTPException(status_code=404, detail=f"Food '{food_name}' not found")
        return food_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching food info: {str(e)}")


@router.get("/recipes/{food_name}", response_model=list)
async def get_food_recipes(food_name: str, appliances: str = None, health_tags: str = None):
    """Get recipes for a specific food with optional filtering."""
    try:
        available_appliances = appliances.split(",") if appliances else []
        health_constraints = health_tags.split(",") if health_tags else []
        
        recipes = RecipeService.get_filtered_recipes(
            food_name,
            available_appliances=available_appliances,
            health_constraints=health_constraints
        )
        return recipes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recipes: {str(e)}")


@router.get("/foods", response_model=list)
async def list_all_foods():
    """Get list of all available foods."""
    try:
        return CalorieService.get_all_foods()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching foods: {str(e)}")


@router.get("/recipes", response_model=list)
async def list_all_recipes():
    """Get list of all available recipes."""
    try:
        return RecipeService.get_all_recipes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recipes: {str(e)}")


@router.get("/low-calorie", response_model=list)
async def get_low_calorie_foods(max_calories: float = 50):
    """Get foods with calories below a threshold."""
    try:
        return CalorieService.get_low_calorie_foods(max_calories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching low-calorie foods: {str(e)}")


@router.get("/low-sugar", response_model=list)
async def get_low_sugar_foods():
    """Get foods suitable for diabetic diets."""
    try:
        return CalorieService.get_low_sugar_foods()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching low-sugar foods: {str(e)}")