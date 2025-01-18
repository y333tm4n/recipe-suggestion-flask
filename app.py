from flask import Flask, request, jsonify
import psycopg2
from psycopg2 import pool
import os
import json
import google.generativeai as genai 
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Database connection pool
db_pool = pool.SimpleConnectionPool(1, 10, 
                                    host=os.getenv('DATABASE_HOST'),
                                    user=os.getenv('DATABASE_USER'),
                                    password=os.getenv('DATABASE_PASSWORD'),
                                    database=os.getenv('DATABASE_NAME'))

# Initialize Google Generative AI
try:
    # Initialize Generative AI using the API key
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
except Exception as ai_init_error:
    app.logger.error('Error initializing Google AI Platform: %s', ai_init_error)

@app.route('/suggest-cuisines', methods=['GET'])
def suggest_cuisines():
    conn = None
    try:
        # Fetch database connection
        conn = db_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT "inventoryID", food_name, food_type, quantity
                FROM public.food_inventory
                WHERE quantity > %s;
            """, (0,))
            inventory = cursor.fetchall()

        # Prepare a string of the inventory items
        inventory_list = ', '.join(
            f"{item[1]} ({item[2]}): {item[3]}" for item in inventory
        )

        # Formulate the prompt for the AI model
        prompt = f"""
        I have the following food inventory: {inventory_list}. 
        Suggest 5 traditional Filipino cuisine recipes that can be made using these ingredients. 
        IMPORTANT: Your ENTIRE response must be a VALID JSON matching this exact structure:
        {{"recipes": [... JSON structure ...]}}
        Do NOT include any text outside of the JSON.
        """

        # Generate content using Google Generative AI
        try:
            # Create the generation configuration and model
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",  # Use a valid model name
                generation_config=generation_config,
            )

            # Start a chat session
            chat_session = model.start_chat(history=[])

            # Send the prompt as a message
            response = chat_session.send_message(prompt)

            # Extract the response text
            recipe_text = response.text.strip() if hasattr(response, 'text') else ''
            if not recipe_text:
                raise ValueError('Empty response from AI model')
        except Exception as model_error:
            app.logger.error('Error with AI model: %s', model_error)
            return jsonify({'message': 'Model prediction error', 'error': str(model_error)}), 500

        # Clean the response (remove unnecessary backticks and escape characters)
        try:
            cleaned_response = recipe_text.replace("```json\n", "").replace("\n```", "")
            parsed_recipes = json.loads(cleaned_response)
            if not parsed_recipes.get('recipes') or not isinstance(parsed_recipes['recipes'], list):
                raise ValueError('Invalid recipe format')
        except (json.JSONDecodeError, ValueError) as parse_error:
            app.logger.error('JSON Parsing Error: %s', parse_error)
            app.logger.error('Problematic Response: %s', cleaned_response)
            return jsonify({
                'message': 'Failed to parse recipe recommendations',
                'error': str(parse_error),
                'rawResponse': cleaned_response
            }), 500

        # Return the parsed recipes in a response
        return jsonify(parsed_recipes['recipes']), 200

    except psycopg2.Error as db_error:
        app.logger.error('Database Error: %s', db_error)
        return jsonify({'message': 'Database error', 'error': str(db_error)}), 500
    except Exception as error:
        app.logger.error('Error generating recipe recommendations: %s', error)
        return jsonify({'message': 'Internal server error', 'error': str(error)}), 500
    finally:
        if conn:
            db_pool.putconn(conn)

if __name__ == '__main__':
    debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    app.run(debug=debug_mode)
