# 🚗 Intelligent Driver Profiling & Usage-Based Insurance (UBI) System

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI](https://img.shields.io/badge/AI-TensorFlow%20%7C%20LSTM-orange)
![Database](https://img.shields.io/badge/Database-MongoDB%20Atlas-green)
![Framework](https://img.shields.io/badge/Frontend-Flask%20%7C%20Streamlit-red)

## 📖 Project Overview

The **Intelligent Driver Profiling System** is an end-to-end IoT and AI solution designed to modernize vehicle insurance. Unlike traditional flat-rate insurance models, this system utilizes **telemetry data** (speed, acceleration, braking patterns) to calculate a **Dynamic Risk Score** for every driver.

The system features a **custom-built web simulator** for data generation, a **cloud-based pipeline** for real-time logging, and a **Long Short-Term Memory (LSTM)** Deep Learning model that analyzes driving behavior to determine personalized insurance premiums.

---

## 🚀 Key Features

### 1. 🎮 Web-Based Physics Simulator

* **Headless Physics Engine:** A custom Python backend (`app.py`) simulating vehicle mass, drag, friction, and engine force.
* **Real-Time Dashboard:** A Cyberpunk-themed web UI using **Flask & Socket.IO** for lag-free, bi-directional control.
* **Data Logging:** Captures 20Hz telemetry (Speed, Throttle, Brake, G-Force) and automatically uploads logs to the cloud upon trip completion.
* **Pause & Save Logic:** Features a "Pause Menu" to save partial trips or force-quit sessions.

### 2. 🧠 AI Risk Engine (LSTM)

* **Temporal Analysis:** Uses an LSTM neural network to detect patterns in time-series driving data.
* **Anomaly Detection:** Identifies specific "Risk Events" such as hard braking, rapid acceleration, and speeding violations.
* **Explainable AI (XAI):** Visualizes *where* and *why* a driver was flagged, using interactive velocity graphs.

### 3. ☁️ Cloud & Analytics Dashboard

* **MongoDB Atlas Integration:** Stores user profiles, policy configurations, and trip logs securely in the cloud.
* **Streamlit Admin Panel:** A professional dashboard for insurers to:
  * Monitor real-time trip feeds.
  * Analyze historical driving data.
  * **Calculate Premiums:** Automatically generates bills based on risk scores, distance caps, and dynamic pricing formulas.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Simulator Frontend** | HTML5, CSS3 (Cyberpunk UI), JavaScript (Socket.IO) |
| **Simulator Backend** | Python, Flask, Pygame (Headless Physics), Eventlet |
| **Database** | MongoDB Atlas (NoSQL Cloud DB) |
| **AI Model** | TensorFlow, Keras, Scikit-Learn (LSTM/RNN) |
| **Admin Dashboard** | Streamlit, Pandas, Altair (Data Viz) |

---

## 📂 Project Structure

```bash
├── app.py                  # Flask Web Simulator (The Physics Engine & Web Server)
├── admin_dashboard.py      # Streamlit Analytics Dashboard (The Admin UI)
├── train_model.py          # Script to train the LSTM model
├── requirements.txt        # List of Python dependencies
├── .env                    # Environment variables (Mongo URI)
├── models/
│   ├── driver_model.h5     # Pre-trained LSTM Model file
│   └── scaler.pkl          # Data Scaler for normalization
├── templates/
│   └── index.html          # Frontend HTML/JS for the Simulator
└── data/                   # Local storage for trip logs (backup)
```

---

## ⚙️ Installation & Setup

1. Clone the Repository

```bash
git clone https://github.com/your-username/driver-profiling-ubi.git
cd driver-profiling-ubi
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Configure Environment

Create a `.env` file in the root directory and add your MongoDB connection string:

```bash
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
```

## 🖥️ How to Run

### Step 1: Start the Driving Simulator

This runs the web server for the vehicle simulation.

```bash
python app.py
```

Open your browser and go to http://localhost:5000.

Enter a User ID (e.g., `u_001`) and click Start Drive.

Use W (Gas) and S (Brake) to drive.

Drive for the duration of the timer. The data will upload automatically when finished.

Press ESC to pause, save logs early, or quit.

### Step 2: Launch the Admin Dashboard

Open a new terminal window to run the analytics platform.

```bash
streamlit run admin_dashboard.py
```

Navigate to the URL provided (usually http://localhost:8501).

Use the Sidebar to select the user (`u_001`).

Go to "New Trip Analysis" to process the pending trip.

Click "Run AI Analysis" to generate a risk score and premium bill.

## 📊 Premium Calculation Logic

The system uses a Pay-How-You-Drive (PHYD) model to ensure fair pricing.

$$
P_{total} = P_{base} + (Distance \times Rate \times (1 + RiskScore^2))
$$

Where:

* `P_{base}`: Fixed monthly fee.
* `Distance`: Kilometers driven.
* `Rate`: Cost per km (increases if Distance Cap is exceeded).
* `RiskScore`: AI-derived score between 0.0 (Safe) and 1.0 (Risky).

Impact:

* Safe Drivers (Score < 0.3): Pay standard rates + earn "Low Usage" discounts.
* Risky Drivers (Score > 0.7): Pay exponentially higher rates per km.

## 🔮 Future Enhancements

- [ ] Integration with real OBD-II hardware scanners.
- [ ] Mobile App for drivers to view their own scores.
- [ ] GPS map integration for route visualization.
- [ ] Multi-agent simulation for traffic scenarios.

## 👨‍💻 Contributors

Niranjan S Kaithota - Lead Developer & AI Engineer

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
