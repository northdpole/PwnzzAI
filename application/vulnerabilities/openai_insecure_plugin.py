import sqlite3

from application.llm_provider import run_pizza_price_tool_conversation


def get_pizza_price(pizza_type):
    """
    VULNERABLE: Get the price for a specific pizza type using raw SQL query
    This function is intentionally vulnerable to SQL injection attacks
    """
    conn = None
    try:
        # Connect to the database (assuming SQLite for this demo)
        conn = sqlite3.connect('instance/pizza_shop.db')
        cursor = conn.cursor()
        print("DEBUG: Connected to database")

        # Test database connection by listing all pizzas
        cursor.execute("SELECT name, price FROM pizza")
        all_pizzas = cursor.fetchall()
        print(f"DEBUG: All pizzas in database: {all_pizzas}")
        # VULNERABLE: Direct string concatenation in SQL query
        # This allows SQL injection through the pizza_type parameter
        pizza = pizza_type.lower().replace("pizza", "").strip()

        # DANGEROUS: Building SQL query with user input without parameterization
        query = f"SELECT name, price FROM pizza WHERE LOWER(name) LIKE '%{pizza}%'"

        print(f"DEBUG: Executing SQL query: {query}")  # For demonstration purposes

        cursor.execute(query)
        result = cursor.fetchone()

        print(f"DEBUG: Query result: {result}")  # Debug output

        if result:
            _, price = result  # We only need the price, but must unpack both values
            result_msg = f"${price}"
            print(f"DEBUG: Returning price: {result_msg}")
            return result_msg
        else:
            print(f"DEBUG: No pizza found")
            return "Pizza not found in our menu"

    except sqlite3.Error as e:
        # Return database error - this can leak sensitive information
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()

# Function definition for OpenAI
price_function = {
    "name": "get_pizza_price",
    "description": "Get the price for a specific pizza type.",
    "parameters": {
        "type": "object",
        "properties": {
            "pizza_type": {
                "type": "string",
                "description": "The type of pizza you want to know the price for"
            },
        },
        "required": ["pizza_type"],
    },
}

def chat_with_openai(user_input, api_key):

    try:
        return run_pizza_price_tool_conversation(
            user_input,
            openai_api_key=api_key,
            price_tool_spec=price_function,
            get_pizza_price=get_pizza_price,
        )

    except Exception as e:
        # Check for common OpenAI API errors
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            return "Error: Invalid OpenAI API key. Please provide a valid API key."
        elif "rate limit" in str(e).lower():
            return "Error: OpenAI API rate limit exceeded. Please try again later."
        else:
            return f"Error: {str(e)}"
