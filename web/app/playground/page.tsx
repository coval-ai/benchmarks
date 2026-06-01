import { Suspense } from "react";
import { PlaygroundPageClient } from "./playground-page-client";

function PlaygroundFallback() {
  return (
    <div className="min-h-screen bg-surface-primary text-text-primary pt-32 px-8">
      <div className="max-w-[1600px] mx-auto text-center text-text-secondary text-sm">
        Loading playground…
      </div>
    </div>
  );
}

export default function PlaygroundPage() {
  return (
    <Suspense fallback={<PlaygroundFallback />}>
      <PlaygroundPageClient />
    </Suspense>
  );
}
