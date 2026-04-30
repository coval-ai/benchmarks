import { Suspense } from "react";
import { STTDashboard } from "./stt-dashboard";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <STTDashboard />
    </Suspense>
  );
}
