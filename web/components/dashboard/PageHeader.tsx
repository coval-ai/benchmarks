"use client";

import React from "react";
import { useDashboard } from "@/contexts/DashboardContext";

const PageHeader: React.FC = () => {
  const { pageTitle, pageSubtitle } = useDashboard();

  return (
    <div className="mb-16 text-center">
      <h1 className="text-4xl font-bold mb-4">{pageTitle}</h1>
      <p className="text-text-primary text-default mb-6 max-w-3xl mx-auto">
        {pageSubtitle}
      </p>
      <p className="text-text-secondary">
        Run evals for your Voice AI on{" "}
        <a
          href="https://coval.dev"
          target="_blank"
          rel="noopener noreferrer"
          className="text-text-secondary hover:text-text-primary transition-colors font-medium"
        >
          Coval.dev
        </a>
      </p>
    </div>
  );
};

export default PageHeader;
