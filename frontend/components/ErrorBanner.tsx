"use client";

import { AlertCircle, X } from "lucide-react";

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-start gap-3 px-4 py-3 bg-red-50 border border-red-200/80 rounded-xl text-sm shadow-sm fade-in-up">
      <AlertCircle size={15} className="text-red-500 shrink-0 mt-0.5" />
      <p className="flex-1 text-red-700 leading-snug">{message}</p>
      <button
        onClick={onDismiss}
        className="shrink-0 p-0.5 text-red-300 hover:text-red-500 transition-colors"
      >
        <X size={13} />
      </button>
    </div>
  );
}
