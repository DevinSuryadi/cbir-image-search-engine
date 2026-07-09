"use client";

import { useState } from "react";
import { SearchResultItem } from "@/lib/types";
import { RefreshCw, Maximize2 } from "lucide-react";

interface ResultCardProps {
  item: SearchResultItem;
  rank: number;
  onMoreLikeThis: (item: SearchResultItem) => void;
  onOpenModal: (item: SearchResultItem) => void;
  animationDelay?: number;
}

export default function ResultCard({
  item,
  rank,
  onMoreLikeThis,
  onOpenModal,
  animationDelay = 0,
}: ResultCardProps) {
  const [imgError, setImgError] = useState(false);

  const similarityPercent = (item.score * 100).toFixed(1);

  // Score-based color: high similarity = indigo, mid = slate, low = muted
  const scoreColor =
    item.score >= 0.85
      ? "bg-indigo-500"
      : item.score >= 0.7
      ? "bg-slate-600"
      : "bg-slate-400";

  return (
    <div
      className="card-lift fade-in-up flex flex-col bg-white border border-slate-200/80 rounded-2xl overflow-hidden shadow-sm"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Image */}
      <div
        className="relative aspect-square bg-slate-100 overflow-hidden cursor-pointer"
        onClick={() => onOpenModal(item)}
      >
        {imgError ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-slate-400">Unavailable</span>
          </div>
        ) : (
          <img
            src={item.image_url}
            alt={item.category}
            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
            onError={() => setImgError(true)}
          />
        )}

        {/* Rank badge */}
        <span className="absolute top-2 left-2 px-2 py-0.5 bg-black/50 text-white text-[11px] font-semibold rounded-full backdrop-blur-sm">
          #{rank}
        </span>

        {/* Expand icon — always visible on hover via parent group */}
        <span className="absolute top-2 right-2 p-1.5 bg-black/50 text-white rounded-full backdrop-blur-sm opacity-0 hover:opacity-100 transition-opacity">
          <Maximize2 size={11} />
        </span>
      </div>

      {/* Body */}
      <div className="px-3 pt-2.5 pb-3 flex flex-col gap-2">
        {/* Category + score */}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-700 truncate capitalize leading-tight">
            {item.category.replace(/_/g, " ")}
          </span>
          <span className="shrink-0 text-xs font-bold text-slate-500 tabular-nums">
            {similarityPercent}%
          </span>
        </div>

        {/* Similarity bar */}
        <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${scoreColor}`}
            style={{ width: `${similarityPercent}%` }}
          />
        </div>

        {/* More Like This */}
        <button
          onClick={() => onMoreLikeThis(item)}
          className="
            flex items-center justify-center gap-1.5 w-full
            py-1.5 text-xs font-medium
            text-slate-500 border border-slate-200 rounded-lg
            hover:text-indigo-600 hover:border-indigo-200 hover:bg-indigo-50
            active:scale-95
            transition-all duration-150
          "
        >
          <RefreshCw size={11} />
          More Like This
        </button>
      </div>
    </div>
  );
}
