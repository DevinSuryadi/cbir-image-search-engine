"use client";

import { SearchResultItem } from "@/lib/types";
import ResultCard from "./ResultCard";
import { Loader2 } from "lucide-react";

interface ResultsGridProps {
  results: SearchResultItem[];
  isLoading: boolean;
  querySeconds?: number;
  onMoreLikeThis: (item: SearchResultItem) => void;
  onOpenModal: (item: SearchResultItem) => void;
}

// Skeleton card for loading state
function SkeletonCard() {
  return (
    <div className="flex flex-col bg-white border border-slate-200/80 rounded-2xl overflow-hidden shadow-sm">
      <div className="aspect-square shimmer" />
      <div className="px-3 pt-2.5 pb-3 flex flex-col gap-2">
        <div className="flex justify-between gap-2">
          <div className="h-3 w-24 bg-slate-200 rounded shimmer" />
          <div className="h-3 w-8 bg-slate-200 rounded shimmer" />
        </div>
        <div className="h-1 w-full bg-slate-100 rounded-full" />
        <div className="h-7 w-full bg-slate-100 rounded-lg shimmer" />
      </div>
    </div>
  );
}

export default function ResultsGrid({
  results,
  isLoading,
  querySeconds,
  onMoreLikeThis,
  onOpenModal,
}: ResultsGridProps) {
  if (isLoading) {
    return (
      <div className="w-full">
        <div className="flex items-center justify-between mb-5">
          <div className="h-4 w-32 bg-slate-200 rounded shimmer" />
          <div className="h-4 w-16 bg-slate-200 rounded shimmer" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="w-full">
      {/* Meta bar */}
      <div className="flex items-center justify-between mb-5">
        <p className="text-sm font-semibold text-slate-700">
          {results.length} result{results.length !== 1 ? "s" : ""} found
        </p>
        {querySeconds !== undefined && (
          <span className="text-xs text-slate-400 tabular-nums">
            {querySeconds.toFixed(3)}s
          </span>
        )}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {results.map((item, index) => (
          <ResultCard
            key={item.point_id}
            item={item}
            rank={index + 1}
            onMoreLikeThis={onMoreLikeThis}
            onOpenModal={onOpenModal}
            animationDelay={index * 30}
          />
        ))}
      </div>
    </div>
  );
}
