from website import create_app

# Create a Flask application instance
app = create_app()

# Run the Flask app
if __name__ == "__main__":
    # Run the app in all available network interfaces and enable debugging
    app.run(host="0.0.0.0", debug=True)
