from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Manual CORS handling (alternative to flask-cors)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ===== CROP PRICE TABLE (Price per metric ton in currency units) =====
CROP_PRICES = {
    "Rice": 25000,
    "Maize": 18000,
    "Jute": 35000,
    "Cotton": 45000,
    "Coconut": 30000,
    "Papaya": 15000,
    "Orange": 22000,
    "Apple": 60000,
    "Muskmelon": 20000,
    "Watermelon": 12000,
    "Grapes": 50000,
    "Mango": 35000,
    "Banana": 18000,
    "Pomegranate": 55000,
    "Lentil": 40000,
    "Blackgram": 42000,
    "Mungbean": 45000,
    "Mothbeans": 38000,
    "Pigeonpeas": 43000,
    "Kidneybeans": 48000,
    "Chickpea": 41000,
    "Coffee": 120000
}

def calculate_revenue(crop, yield_per_hectare, area_hectares):
    """
    Calculate total revenue for a crop

    Args:
        crop: Name of the crop
        yield_per_hectare: Predicted yield in metric tons per hectare
        area_hectares: Total cultivation area in hectares

    Returns:
        dict: Revenue calculations
    """
    if crop not in CROP_PRICES:
        return None

    price_per_ton = CROP_PRICES[crop]
    total_yield = yield_per_hectare * area_hectares
    total_revenue = total_yield * price_per_ton
    revenue_per_hectare = yield_per_hectare * price_per_ton

    return {
        'crop': crop,
        'price_per_ton': price_per_ton,
        'yield_per_hectare': round(yield_per_hectare, 2),
        'revenue_per_hectare': round(revenue_per_hectare, 2),
        'total_area_hectares': area_hectares,
        'total_yield_tons': round(total_yield, 2),
        'total_revenue': round(total_revenue, 2),
        'currency': 'INR'  # Change to your currency
    }

# ===== ROUTE: Full Analysis =====
@app.route('/api/full_analysis', methods=['POST'])
def full_analysis():
    """
    Full analysis endpoint that:
    1. Predicts top 5 best crops using model_first based on soil/climate parameters
    2. Gets state directly from request
    3. Gets season directly from request
    4. Tests each crop for yield prediction and selects the one with HIGHEST positive yield
    5. Calculates revenue if area is provided
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided in request body'
            }), 400

        # ===== STEP 1: Load models for crop recommendation =====
        try:
            model_first = pickle.load(open('model_first.pkl', 'rb'))
            sc = pickle.load(open('standscaler.pkl', 'rb'))
            ms = pickle.load(open('minmaxscaler_first.pkl', 'rb'))
        except FileNotFoundError as e:
            return jsonify({
                'success': False,
                'error': f'Crop recommendation model files not found: {str(e)}'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error loading crop recommendation models: {str(e)}'
            }), 500

        # ===== STEP 2: Get soil and climate parameters for crop prediction =====
        try:
            # Handle both spellings for Phosphorus and pH
            N = float(data.get('Nitrogen', 0))
            P = float(data.get('Phosphorus', data.get('Phosporus', 0)))
            K = float(data.get('Potassium', 0))
            temp = float(data.get('Temperature', 0))
            humidity = float(data.get('Humidity', 0))
            ph = float(data.get('Ph', data.get('pH', 0)))
            # Make Rainfall optional: parse it if provided, otherwise leave as None
            raw_rainfall = data.get('Rainfall', None)
            if raw_rainfall in (None, ''):
                rainfall = None
            else:
                rainfall = float(raw_rainfall)
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'All soil and climate parameters must be valid numbers: {str(e)}'
            }), 400

        # Validate input parameters (Rainfall is optional)
        if not all([N, P, K, temp, humidity, ph]):
            return jsonify({
                'success': False,
                'error': 'All parameters are required: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph'
            }), 400

        # ===== STEP 3: Get TOP 5 crop predictions using model_first =====
        # Use a fallback numeric value for the crop-recommendation model if rainfall is missing
        rainfall_for_model = float(rainfall) if rainfall is not None else 0.0
        feature_list = [N, P, K, temp, humidity, ph, rainfall_for_model]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Get prediction probabilities for all crops
        if hasattr(model_first, 'predict_proba'):
            # For classifiers with probability prediction
            probabilities = model_first.predict_proba(final_features)[0]
            # Get indices of top 5 predictions
            top_5_indices = np.argsort(probabilities)[-5:][::-1]
            top_5_crops_encoded = top_5_indices + 1  # Add 1 because crop_dict starts at 1
        elif hasattr(model_first, 'decision_function'):
            # For classifiers with decision function (like SVM)
            decision_scores = model_first.decision_function(final_features)[0]
            top_5_indices = np.argsort(decision_scores)[-5:][::-1]
            top_5_crops_encoded = top_5_indices + 1
        else:
            # Fallback: just get the single prediction and try related crops
            prediction = model_first.predict(final_features)
            top_5_crops_encoded = [prediction[0]]
            # Add nearby predictions as fallback
            for i in range(1, 5):
                next_crop = prediction[0] + i
                if next_crop <= 22:  # Max crop ID
                    top_5_crops_encoded.append(next_crop)

        # Crop dictionary mapping
        crop_dict = {
            1: "Rice",
            2: "Maize",
            3: "Jute",
            4: "Cotton",
            5: "Coconut",
            6: "Papaya",
            7: "Orange",
            8: "Apple",
            9: "Muskmelon",
            10: "Watermelon",
            11: "Grapes",
            12: "Mango",
            13: "Banana",
            14: "Pomegranate",
            15: "Lentil",
            16: "Blackgram",
            17: "Mungbean",
            18: "Mothbeans",
            19: "Pigeonpeas",
            20: "Kidneybeans",
            21: "Chickpea",
            22: "Coffee"
        }

        # Convert encoded predictions to crop names
        top_5_crops = []
        for crop_id in top_5_crops_encoded:
            if crop_id in crop_dict:
                top_5_crops.append(crop_dict[crop_id])

        if not top_5_crops:
            return jsonify({
                'success': False,
                'error': 'Could not determine suitable crops with the provided data'
            }), 400

        # ===== STEP 4: Load models for yield prediction =====
        try:
            # Try loading as tuple first
            loaded_data = joblib.load('best_model_augmented.pkl')
            if isinstance(loaded_data, tuple):
                best_model, feature_columns = loaded_data
            else:
                # If it's just the model, load feature columns separately or define them
                best_model = loaded_data
                # Define feature columns based on your training code
                feature_columns = ['Crop', 'State', 'Season', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

            label_encoders = joblib.load('label_encoders.pkl')
            median_values = joblib.load('median_values.pkl')
            unique_values = joblib.load('unique_values.pkl')
        except FileNotFoundError as e:
            return jsonify({
                'success': False,
                'error': f'Yield prediction model files not found: {str(e)}'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error loading yield prediction model files: {str(e)}'
            }), 500

        # ===== STEP 5: Get state from request =====
        state = data.get('state', '').strip().title()

        if not state:
            return jsonify({
                'success': False,
                'error': 'State is required'
            }), 400

        # Validate state - handle both dictionary and list formats
        try:
            available_states = unique_values.get('states', label_encoders['State'].classes_)
            if state not in available_states:
                return jsonify({
                    'success': False,
                    'error': f'Invalid state: {state}. Available states: {", ".join(sorted(available_states))}'
                }), 400
        except KeyError:
            return jsonify({
                'success': False,
                'error': 'State encoder not found in model files'
            }), 500

        # ===== STEP 6: Get season from request =====
        season = data.get('season', '').strip()

        if not season:
            return jsonify({
                'success': False,
                'error': 'Season is required'
            }), 400

        # Validate season - handle both dictionary and list formats
        try:
            available_seasons = unique_values.get('seasons', label_encoders['Season'].classes_)
            if season not in available_seasons:
                return jsonify({
                    'success': False,
                    'error': f'Invalid season: {season}. Available seasons: {", ".join(sorted(available_seasons))}'
                }), 400
        except KeyError:
            return jsonify({
                'success': False,
                'error': 'Season encoder not found in model files'
            }), 500

        # Get numerical parameters with defaults
        try:
            crop_year = float(data.get('crop_year', median_values.get('Crop_Year', 2020)))
            # Map incoming 'Rainfall' to 'annual_rainfall' when the latter isn't provided
            if data.get('annual_rainfall') not in (None, ""):
                annual_rainfall = float(data.get('annual_rainfall'))
            else:
                # use the previously parsed `rainfall` (from 'Rainfall' key) or median if not available
                try:
                    annual_rainfall = float(rainfall)
                except Exception:
                    annual_rainfall = float(median_values.get('Annual_Rainfall', 1000))

            fertilizer = float(data.get('fertilizer', median_values.get('Fertilizer', 100)))
            pesticide = float(data.get('pesticide', median_values.get('Pesticide', 100)))
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Crop year, rainfall, fertilizer, and pesticide must be valid numbers: {str(e)}'
            }), 400

        # Get area for revenue calculation (optional)
        area_hectares = data.get('area_hectares', None)
        if area_hectares is not None:
            try:
                area_hectares = float(area_hectares)
                if area_hectares <= 0:
                    area_hectares = None
            except (ValueError, TypeError):
                area_hectares = None

        # ===== STEP 7: Test each crop and find the one with HIGHEST positive yield =====
        try:
            state_encoded = label_encoders['State'].transform([state])[0]
            season_encoded = label_encoders['Season'].transform([season])[0]
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error encoding state or season: {str(e)}'
            }), 500

        best_crop = None
        best_yield = -999  # Initialize with a very low value
        # Track best crop by revenue (uses area_hectares if provided, otherwise per-hectare)
        best_crop_by_revenue = None
        best_revenue = -float('inf')
        all_crop_predictions = []

        for crop in top_5_crops:
            # Check if crop is in yield model
            if crop not in label_encoders['Crop'].classes_:
                crop_info = {
                    'crop': crop,
                    'available_in_yield_model': False,
                    'predicted_yield': None,
                    'reason': 'Not available in yield prediction model'
                }
                all_crop_predictions.append(crop_info)
                continue

            # Encode the crop
            try:
                crop_encoded = label_encoders['Crop'].transform([crop])[0]
            except Exception as e:
                crop_info = {
                    'crop': crop,
                    'available_in_yield_model': True,
                    'predicted_yield': None,
                    'error': f'Error encoding crop: {str(e)}'
                }
                all_crop_predictions.append(crop_info)
                continue

            # Create input for yield prediction
            input_dict = {}
            for col in feature_columns:
                if col == 'Crop':
                    input_dict[col] = [crop_encoded]
                elif col == 'State':
                    input_dict[col] = [state_encoded]
                elif col == 'Season':
                    input_dict[col] = [season_encoded]
                elif col == 'Crop_Year':
                    input_dict[col] = [crop_year]
                elif col == 'Annual_Rainfall':
                    input_dict[col] = [annual_rainfall]
                elif col == 'Fertilizer':
                    input_dict[col] = [fertilizer]
                elif col == 'Pesticide':
                    input_dict[col] = [pesticide]

            # Create DataFrame for prediction
            input_data = pd.DataFrame(input_dict)

            # Make yield prediction
            try:
                prediction_yield = best_model.predict(input_data)
                predicted_yield = float(prediction_yield[0])

                price_per_ton = CROP_PRICES.get(crop, 0)
                revenue_per_hectare = predicted_yield * price_per_ton
                # If the caller provided an area, compute total revenue for that area; otherwise use 1 hectare
                area_for_calc = area_hectares if area_hectares is not None else 1
                total_revenue = revenue_per_hectare * area_for_calc

                crop_info = {
                    'crop': crop,
                    'available_in_yield_model': True,
                    'predicted_yield': round(predicted_yield, 2),
                    'is_positive': predicted_yield > 0,
                    'price_per_ton': price_per_ton,
                    'revenue_per_hectare': round(revenue_per_hectare, 2),
                    'area_used_for_revenue': area_for_calc,
                    'total_revenue_for_area': round(total_revenue, 2)
                }

                # Also include detailed revenue calculation object when an explicit area was given
                if area_hectares is not None and predicted_yield > 0:
                    revenue_calc = calculate_revenue(crop, predicted_yield, area_hectares)
                    if revenue_calc:
                        crop_info['revenue_calculation'] = revenue_calc

                all_crop_predictions.append(crop_info)

                # DEBUG: Print current crop and yield for troubleshooting
                print(f"Crop: {crop}, Yield: {predicted_yield}, Best so far: {best_crop} with {best_yield}")

                # FIXED: Properly select crop with HIGHEST positive yield
                if predicted_yield > 0:
                    if best_crop is None or predicted_yield > best_yield:
                        best_crop = crop
                        best_yield = predicted_yield
                        print(f"NEW BEST: {best_crop} with yield {best_yield}")

                # Track best crop by total revenue (or per-hectare when area not provided)
                try:
                    if total_revenue > best_revenue:
                        best_crop_by_revenue = crop
                        best_revenue = total_revenue
                        print(f"NEW BEST BY REVENUE: {best_crop_by_revenue} with revenue {best_revenue}")
                except Exception:
                    # In case revenue comparison fails, skip
                    pass

            except Exception as e:
                all_crop_predictions.append({
                    'crop': crop,
                    'available_in_yield_model': True,
                    'predicted_yield': None,
                    'error': str(e)
                })

        # DEBUG: Print final selection
        print(f"FINAL SELECTION: {best_crop} with yield {best_yield}")

        # Sort alternative crops by yield (highest first)
        all_crop_predictions.sort(key=lambda x: x.get('predicted_yield', -999) if x.get('predicted_yield') is not None else -999, reverse=True)

        # FIXED: If no crop with positive yield, select the one with highest yield overall
        if best_crop is None and all_crop_predictions:
            # Find the crop with the highest yield (even if negative)
            valid_predictions = [c for c in all_crop_predictions if c.get('predicted_yield') is not None]
            if valid_predictions:
                # Prefer highest revenue if available, otherwise fall back to highest yield
                # Ensure revenue fields exist
                revenue_available = any('total_revenue_for_area' in c for c in valid_predictions)
                if revenue_available:
                    valid_predictions.sort(key=lambda x: x.get('total_revenue_for_area', -float('inf')), reverse=True)
                    best_crop_by_revenue = valid_predictions[0]['crop']
                    best_revenue = valid_predictions[0].get('total_revenue_for_area', best_revenue)
                    # Also set best_crop/best_yield for backward compatibility
                    best_crop = valid_predictions[0]['crop']
                    best_yield = valid_predictions[0].get('predicted_yield', best_yield)
                    print(f"FALLBACK SELECTION BY REVENUE: {best_crop_by_revenue} with revenue {best_revenue}")
                else:
                    valid_predictions.sort(key=lambda x: x.get('predicted_yield', -999), reverse=True)
                    best_crop = valid_predictions[0]['crop']
                    best_yield = valid_predictions[0]['predicted_yield']
                    print(f"FALLBACK SELECTION: {best_crop} with yield {best_yield}")

        # ===== STEP 8: Prepare response =====
        if best_crop is None:
            # No crop predictions available at all
            return jsonify({
                'success': True,
                'crop_recommendation': {
                    'recommended_crop': top_5_crops[0],
                    'recommendation_basis': 'Soil and climate analysis',
                    'warning': 'No yield predictions available for any recommended crops'
                },
                'yield_prediction': {
                    'available': False,
                    'reason': 'No yield predictions generated for any crops',
                    'all_tested_crops': all_crop_predictions
                },
                'soil_parameters': {
                    'Nitrogen': N,
                    'Phosphorus': P,
                    'Potassium': K,
                    'Temperature': temp,
                    'Humidity': humidity,
                    'pH': ph,
                    'Rainfall': rainfall
                },
                'suggestions': [
                    'Consider adjusting fertilizer and pesticide usage',
                    'Try different seasons for cultivation',
                    'Consult with local agricultural experts for region-specific advice'
                ]
            }), 200

        # ===== STEP 9: Calculate revenue for recommended crop =====
        revenue_info = None
        if area_hectares and best_yield > 0:
            revenue_info = calculate_revenue(best_crop, best_yield, area_hectares)

        # Get count of crops with valid predictions
        crops_with_predictions = len([c for c in all_crop_predictions if c.get('predicted_yield') is not None])

        # Create yield comparison string
        yield_comparison = []
        for crop_info in all_crop_predictions:
            if crop_info.get('predicted_yield') is not None:
                yield_comparison.append(f"{crop_info['crop']}: {crop_info['predicted_yield']} tons/ha")

        # ===== STEP 10: Return successful prediction =====
        # Prefer recommendation by revenue when available
        recommended_crop = best_crop_by_revenue if best_crop_by_revenue is not None else best_crop

        # Find revenue details for the recommended-by-revenue crop (if any)
        best_rev_info = None
        if best_crop_by_revenue is not None:
            for c in all_crop_predictions:
                if c.get('crop') == best_crop_by_revenue:
                    best_rev_info = c
                    break

        response = {
            'success': True,
            'crop_recommendation': {
                'recommended_crop': recommended_crop,
                'recommendation_basis': 'Highest expected revenue for given area' if best_crop_by_revenue is not None else 'Highest positive yield from soil and climate analysis',
                'suitable_for': f'{state} region during {season} season',
                'ranking': f'Best yield among {crops_with_predictions} tested crops',
                'selection_criteria': 'Crop with maximum expected revenue' if best_crop_by_revenue is not None else 'Crop with maximum predicted yield',
                'yield_comparison': ', '.join(yield_comparison)
            },
            'yield_prediction': {
                'available': True,
                'crop': best_crop,
                'state': state,
                'season': season,
                'predicted_yield': round(best_yield, 2),
                'unit': 'metric ton per hectare',
                'interpretation': f'Expected yield of {round(best_yield, 2)} metric tons per hectare'
            },
            'alternative_crops': all_crop_predictions,
            'soil_parameters': {
                'Nitrogen': N,
                'Phosphorus': P,
                'Potassium': K,
                'Temperature': temp,
                'Humidity': humidity,
                'pH': ph,
                'Rainfall': rainfall
            },
            'agricultural_parameters': {
                'crop_year': crop_year,
                'annual_rainfall': annual_rainfall,
                'fertilizer': fertilizer,
                'pesticide': pesticide
            },
            'model_info': {
                'crop_recommendation_model': 'model_first',
                'yield_prediction_model': best_model.__class__.__name__,
                'crops_evaluated': len(all_crop_predictions),
                'features_used': feature_columns
            }
        }

        # Attach best-by-revenue summary if available
        if best_rev_info:
            response['best_by_revenue'] = {
                'crop': best_rev_info.get('crop'),
                'predicted_yield': best_rev_info.get('predicted_yield'),
                'revenue_per_hectare': best_rev_info.get('revenue_per_hectare'),
                'area_used_for_revenue': best_rev_info.get('area_used_for_revenue'),
                'total_revenue_for_area': best_rev_info.get('total_revenue_for_area'),
                'currency': 'INR'
            }

        # Add revenue calculation if available
        if revenue_info:
            response['revenue_calculation'] = revenue_info

        return jsonify(response), 200

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


# ===== ROUTE: Calculate Revenue =====
@app.route('/api/calculate_revenue', methods=['POST'])
def calculate_revenue_endpoint():
    """
    Calculate revenue for a specific crop, yield, and area
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        crop = data.get('crop', '').strip().title()
        yield_per_hectare = data.get('yield_per_hectare')
        area_hectares = data.get('area_hectares')

        # Validate inputs
        if not crop:
            return jsonify({
                'success': False,
                'error': 'Crop name is required'
            }), 400

        if crop not in CROP_PRICES:
            return jsonify({
                'success': False,
                'error': f'Crop "{crop}" not found in price table. Available crops: {", ".join(sorted(CROP_PRICES.keys()))}'
            }), 400

        try:
            yield_per_hectare = float(yield_per_hectare)
            area_hectares = float(area_hectares)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'yield_per_hectare and area_hectares must be valid numbers'
            }), 400

        if yield_per_hectare <= 0 or area_hectares <= 0:
            return jsonify({
                'success': False,
                'error': 'yield_per_hectare and area_hectares must be positive numbers'
            }), 400

        # Calculate revenue
        revenue_info = calculate_revenue(crop, yield_per_hectare, area_hectares)

        return jsonify({
            'success': True,
            'data': revenue_info
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


# ===== ROUTE: Get Crop Prices =====
@app.route('/api/crop_prices', methods=['GET'])
def get_crop_prices():
    """
    Returns the price table for all crops
    """
    return jsonify({
        'success': True,
        'data': {
            'prices': CROP_PRICES,
            'currency': 'INR',
            'unit': 'per metric ton'
        }
    }), 200


# ===== ROUTE: Get Available Options =====
@app.route('/api/options', methods=['GET'])
def get_options():
    """
    Returns available options for crops, states, and seasons
    """
    try:
        unique_values = joblib.load('unique_values.pkl')

        return jsonify({
            'success': True,
            'data': {
                'crops': unique_values.get('crops', []),
                'states': unique_values.get('states', []),
                'seasons': unique_values.get('seasons', [])
            }
        }), 200

    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error': 'unique_values.pkl file not found'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


# ===== ROUTE: Health Check =====
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return jsonify({
        'success': True,
        'message': 'API is running',
        'endpoints': {
            'full_analysis': '/api/full_analysis [POST]',
            'calculate_revenue': '/api/calculate_revenue [POST]',
            'crop_prices': '/api/crop_prices [GET]',
            'options': '/api/options [GET]',
            'health': '/api/health [GET]'
        }
    }), 200


# ===== ROUTE: Home =====
@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint with API documentation
    """
    return jsonify({
        'message': 'Crop Recommendation and Yield Prediction API',
        'version': '2.2.0',
        'description': 'Smart crop recommendation system that selects highest yielding crop with revenue calculation',
        'endpoints': {
            'full_analysis': {
                'url': '/api/full_analysis',
                'method': 'POST',
                'description': 'Get crop recommendation with highest yield validation and revenue calculation',
                'required_parameters': [
                    'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature',
                    'Humidity', 'Ph', 'Rainfall', 'state', 'season'
                ],
                'optional_parameters': [
                    'crop_year', 'annual_rainfall', 'fertilizer', 'pesticide', 'area_hectares'
                ]
            },
            'calculate_revenue': {
                'url': '/api/calculate_revenue',
                'method': 'POST',
                'description': 'Calculate revenue for specific crop, yield, and area',
                'required_parameters': ['crop', 'yield_per_hectare', 'area_hectares']
            },
            'crop_prices': {
                'url': '/api/crop_prices',
                'method': 'GET',
                'description': 'Get price table for all crops'
            },
            'options': {
                'url': '/api/options',
                'method': 'GET',
                'description': 'Get available crops, states, and seasons'
            },
            'health': {
                'url': '/api/health',
                'method': 'GET',
                'description': 'Health check endpoint'
            }
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
