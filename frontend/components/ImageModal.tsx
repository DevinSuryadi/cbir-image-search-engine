"use client";

import { useEffect, useCallback } from "react";
import { SearchResultItem } from "@/lib/types";
import { X, RefreshCw, ExternalLink } from "lucide-react";

interface ImageModalProps {
  item: SearchResultItem | null;
  onClose: () => void;
  onMoreLikeThis: (item: SearchResultItem) => void;
}

export default function ImageModal({
  item,
  onClose,
  onMoreLikeThis,
}: ImageModalProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); },
    [onClose]
  );

  useEffect(() => {
    if (!item) return;
    document.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [item, handleKeyDown]);

  if (!item) return null;

  const similarityPercent = (item.score * 100).toFixed(1);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-8 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-2xl overflow-hidden w-full max-w-2xl max-h-[90vh] flex flex-col fade-in-up"
        style={{ animationDelay: "0ms" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-slate-100">
          <div>
            <h3 className="text-sm font-bold text-slate-800 capitalize leading-tight">
              {item.category.replace(/_/g, " ")}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Similarity &mdash;{" "}
              <span className="font-semibold text-slate-600">
                {similarityPercent}%
              </span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors shrink-0"
          >
            <X size={17} />
          </button>
        </div>

        {/* Image */}
        <div className="flex-1 overflow-hidden bg-slate-50 flex items-center justify-center min-h-0 px-4 py-4">
          <img
            src={item.image_url}
            alt={item.category}
            className="max-w-full max-h-[55vh] object-contain rounded-xl"
          />
        </div>

        {/* Footer */}
        <div className="flex flex-wrap items-center gap-2.5 px-5 py-4 border-t border-slate-100 bg-slate-50/50">
          <button
            onClick={() => { onMoreLikeThis(item); onClose(); }}
            className="
              flex items-center gap-2 px-4 py-2 text-sm font-semibold
              bg-indigo-600 text-white rounded-xl
              hover:bg-indigo-700 active:scale-95
              transition-all duration-150 shadow-sm
            "
          >
            <RefreshCw size={14} />
            More Like This
          </button>
          <a
            href={item.image_url}
            target="_blank"
            rel="noopener noreferrer"
            className="
              flex items-center gap-2 px-4 py-2 text-sm font-medium
              border border-slate-200 text-slate-600 rounded-xl bg-white
              hover:bg-slate-50 hover:border-slate-300 active:scale-95
              transition-all duration-150 shadow-sm
            "
          >
            <ExternalLink size={14} />
            Open original
          </a>
          <div className="flex-1" />
          <p className="text-[11px] text-slate-400 font-mono truncate max-w-[180px] hidden sm:block">
            {item.point_id}
          </p>
        </div>
      </div>
    </div>
  );
}
