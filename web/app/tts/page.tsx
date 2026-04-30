import { Suspense } from "react";
import { TTSDashboard } from "./tts-dashboard";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <TTSDashboard />
    </Suspense>
  );
}
