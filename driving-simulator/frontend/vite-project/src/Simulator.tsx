import { useEffect, useState } from "react";
import { socket } from "./socket";

type Telemetry = {
  speed?: number;
  speed_limit?: number;
  throttle?: number;
  brake?: number;
};

export default function Simulator() {
  const [data, setData] = useState<Telemetry>({});
  const [throttle, setThrottle] = useState(0);
  const [brake, setBrake] = useState(0);

  // Receive data from Python
  useEffect(() => {
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setData(payload);
    };
  }, []);

  // Keyboard controls
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp") setThrottle(1);
      if (e.key === "ArrowDown") setBrake(1);
    };

    const up = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp") setThrottle(0);
      if (e.key === "ArrowDown") setBrake(0);
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);

    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // Send inputs to Python
  useEffect(() => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          throttle,
          brake
        })
      );
    }
  }, [throttle, brake]);

  return (
    <div style={{ padding: 40, fontFamily: "monospace" }}>
      <h1>🚗 Driving Simulator</h1>

      <p>Speed: {data.speed ?? "--"} km/h</p>
      <p>Speed Limit: {data.speed_limit ?? "--"} km/h</p>

      <p>Throttle: {data.throttle ?? 0}</p>
      <p>Brake: {data.brake ?? 0}</p>

      <p style={{ color: data.speed && data.speed > (data.speed_limit ?? 0) ? "red" : "white" }}>
        {data.speed && data.speed_limit && data.speed > data.speed_limit
          ? "⚠️ OVER SPEEDING"
          : "OK"}
      </p>

      <p style={{ opacity: 0.6 }}>
        Controls: ↑ Throttle | ↓ Brake
      </p>
    </div>
  );
}
