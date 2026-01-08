import { useEffect, useState } from "react";
import { socket } from "../socket";
import Speedometer from "../components/Speedometer";
import Pedals from "../components/Pedals";

export default function Simulator() {
  const [data, setData] = useState({});
  const [throttle, setThrottle] = useState(0);
  const [brake, setBrake] = useState(0);

  useEffect(() => {
    socket.onmessage = e => setData(JSON.parse(e.data));
  }, []);

  useEffect(() => {
    const down = e => {
      if (e.key === "ArrowUp") setThrottle(1);
      if (e.key === "ArrowDown") setBrake(1);
    };
    const up = () => {
      setThrottle(0);
      setBrake(0);
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  useEffect(() => {
    socket.send({ throttle, brake });
  }, [throttle, brake]);

  return (
    <>
      <Speedometer speed={data.speed} limit={data.speed_limit} />
      <Pedals throttle={data.throttle} brake={data.brake} />
    </>
  );
}
