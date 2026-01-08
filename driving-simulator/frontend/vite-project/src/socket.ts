export const socket = new WebSocket("ws://localhost:8000/drive");

socket.onopen = () => {
  console.log("✅ Connected to backend");
};

socket.onerror = (e) => {
  console.error("❌ WebSocket error", e);
};
