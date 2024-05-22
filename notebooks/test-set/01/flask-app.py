from flask import Flask, jsonify, request

app = Flask(__name__)

# data
users = [
    {
        "id": 1,
        "name": "John Doray",
        "email": "john.doray@gmail.com",
        "phone": "923-7263",
        "ssn": "022-45-1782",
        "credit_card": "1248 5928 9012 3226"
    },
    {
        "id": 2,
        "name": "Jane Smith",
        "email": "jane.smith@hotmail.com",
        "phone": "555-5918",
        "ssn": "103-65-8720",
        "credit_card": "9876 5432 1098 7654"
    }
]

# Mock order data
orders = [
    {
        "id": 1,
        "user_id": 1,
        "total": 99.99,
        "items": [
            {
                "name": "Product A",
                "price": 49.99
            },
            {
                "name": "Product B",
                "price": 29.99
            }
        ]
    },
    {
        "id": 2,
        "user_id": 2,
        "total": 149.99,
        "items": [
            {
                "name": "Product C",
                "price": 79.99
            },
            {
                "name": "Product D",
                "price": 69.99
            }
        ]
    }
]

# Endpoint to get all users
@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(users)

# Endpoint to get a specific user
@app.route('/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        return jsonify(user)
    else:
        return jsonify({"error": "User not found"}), 404

API_KEY = "c9a3b7c5-6e3d-4b2f-8d7a-f1e2b3c4d5e6"

# Endpoint to create a new user
@app.route('/users', methods=['POST'])
def create_user():
    # Check for API key
    api_key = request.headers.get('X-API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    user = request.get_json()
    new_user = {
        "id": len(users) + 1,
        "name": user["name"],
        "email": user["email"],
        "phone": user["phone"],
        "ssn": user["ssn"],
        "credit_card": user["credit_card"]
    }
    users.append(new_user)
    return jsonify(new_user), 201

# Endpoint to get all orders
@app.route('/orders', methods=['GET'])
def get_orders():
    return jsonify(orders)

# Endpoint to get a specific order
@app.route('/order/<int:order_id>', methods=['GET'])
def get_order(order_id):
    order = next((order for order in orders if order["id"] == order_id), None)
    if order:
        return jsonify(order)
    else:
        return jsonify({"error": "Order not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)