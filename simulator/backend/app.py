from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
from simulator import play_trip, latest_state, remote_controls

app = Flask(__name__)
CORS(app)

@app.route("/api/live", methods=["GET"])
def live_data():
    return jsonify(latest_state)

@app.route("/api/control", methods=["POST"])
def control_car():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400
        
    # Update the shared control state
    remote_controls["up"] = data.get("up", False)
    remote_controls["down"] = data.get("down", False)
    
    return jsonify({"status": "ok"})

def start_simulator():
    play_trip(user_id="u_001")

if __name__ == "__main__":
    # Run Flask in a separate thread
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False))
    t.daemon = True
    t.start()

    # Run Pygame in the main thread (REQUIRED for macOS)
    start_simulator()
