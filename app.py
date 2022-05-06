# Main file for the algview visualization

from algview import server

# === Main ===
if __name__ == '__main__':
    from algview import app
    app.run_server(debug=True)