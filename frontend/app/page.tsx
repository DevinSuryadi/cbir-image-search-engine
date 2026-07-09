"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import SearchArea from "@/components/SearchArea";
import ResultsGrid from "@/components/ResultsGrid";
import ImageModal from "@/components/ImageModal";
import ErrorBanner from "@/components/ErrorBanner";
import { useSearchHistory } from "@/hooks/useSearchHistory";
import { searchByText, searchByImage, searchMoreLikeThis } from "@/lib/api";
import { SearchResultItem, SearchResponse, HistoryEntry } from "@/lib/types";
import { Menu, Layers } from "lucide-react";

export default function HomePage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [querySeconds, setQuerySeconds] = useState<number | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modalItem, setModalItem] = useState<SearchResultItem | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const { history, addEntry, clearHistory, removeEntry } = useSearchHistory();

  // -----------------------------------------------------------------------
  // Shared result handler
  // -----------------------------------------------------------------------

  const handleResponse = useCallback(
    (response: SearchResponse, query: string, mode: "text" | "image") => {
      setResults(response.results);
      setQuerySeconds(response.query_seconds);
      setHasSearched(true);
      addEntry({ mode, query, resultCount: response.result_count });
    },
    [addEntry]
  );

  // -----------------------------------------------------------------------
  // Search handlers
  // -----------------------------------------------------------------------

  const handleTextSearch = useCallback(
    async (query: string, topK: number) => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await searchByText(query, topK);
        handleResponse(response, query, "text");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed. Please try again.");
      } finally {
        setIsLoading(false);
      }
    },
    [handleResponse]
  );

  const handleImageSearch = useCallback(
    async (file: File, topK: number) => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await searchByImage(file, topK);
        handleResponse(response, file.name, "image");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed. Please try again.");
      } finally {
        setIsLoading(false);
      }
    },
    [handleResponse]
  );

  const handleMoreLikeThis = useCallback(
    async (item: SearchResultItem) => {
      setIsLoading(true);
      setError(null);
      setModalItem(null);
      try {
        const response = await searchMoreLikeThis(item.image_url, 20);
        handleResponse(response, `Similar to "${item.category.replace(/_/g, " ")}"`, "image");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed. Please try again.");
      } finally {
        setIsLoading(false);
      }
    },
    [handleResponse]
  );

  // -----------------------------------------------------------------------
  // History re-run (text only — images require re-upload)
  // -----------------------------------------------------------------------

  const handleHistorySelect = useCallback(
    (entry: HistoryEntry) => {
      setSidebarOpen(false);
      if (entry.mode === "text") {
        handleTextSearch(entry.query, 20);
      }
    },
    [handleTextSearch]
  );

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <Sidebar
        history={history}
        onSelect={handleHistorySelect}
        onRemove={removeEntry}
        onClear={clearHistory}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
      />

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">

        {/* ---------------------------------------------------------------- */}
        {/* Header                                                           */}
        {/* ---------------------------------------------------------------- */}
        <header className="sticky top-0 z-10 flex items-center gap-3 px-6 py-3.5 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-sm">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="p-2 rounded-xl text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors lg:hidden"
            aria-label="Toggle sidebar"
          >
            <Menu size={17} />
          </button>

          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-600 text-white">
              <Layers size={14} />
            </div>
            <span className="text-sm font-bold text-slate-800 tracking-tight">
              CBIR Search
            </span>
          </div>

          <div className="flex-1" />

          {/* Tech stack tags */}
          <div className="hidden sm:flex items-center gap-1.5">
            {["Caltech-101", "CLIP", "Qdrant"].map((tag) => (
              <span
                key={tag}
                className="px-2.5 py-1 text-[11px] font-medium text-slate-500 bg-slate-100 rounded-full border border-slate-200/80"
              >
                {tag}
              </span>
            ))}
          </div>
        </header>

        {/* ---------------------------------------------------------------- */}
        {/* Page content                                                     */}
        {/* ---------------------------------------------------------------- */}
        <main className="flex-1 px-6 py-12">

          {/* Hero — only shown before first search */}
          {!hasSearched && !isLoading && (
            <div className="max-w-3xl mx-auto mb-10 text-center">
              <h1 className="text-3xl font-bold text-slate-900 tracking-tight leading-tight">
                Content-Based Image Retrieval
              </h1>
              <p className="mt-3 text-slate-500 text-base leading-relaxed max-w-xl mx-auto">
                Search the Caltech-101 dataset using a natural language description
                or by uploading a reference image.
              </p>
            </div>
          )}

          {/* Search area */}
          <div className="max-w-3xl mx-auto mb-10">
            {error && (
              <div className="mb-4">
                <ErrorBanner message={error} onDismiss={() => setError(null)} />
              </div>
            )}
            <SearchArea
              onTextSearch={handleTextSearch}
              onImageSearch={handleImageSearch}
              isLoading={isLoading}
            />
          </div>

          {/* Results */}
          <ResultsGrid
            results={results}
            isLoading={isLoading}
            querySeconds={querySeconds}
            onMoreLikeThis={handleMoreLikeThis}
            onOpenModal={setModalItem}
          />
        </main>

        {/* Footer */}
        <footer className="px-6 py-4 border-t border-slate-200/80 text-center">
          <p className="text-xs text-slate-400">
            CBIR Image Search &mdash; CLIP Embeddings + Qdrant Cloud
          </p>
        </footer>
      </div>

      {/* Modal */}
      <ImageModal
        item={modalItem}
        onClose={() => setModalItem(null)}
        onMoreLikeThis={handleMoreLikeThis}
      />
    </div>
  );
}
