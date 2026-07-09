"use client";

import { useCallback, useRef, useState } from "react";
import { Search, Upload, X, SlidersHorizontal } from "lucide-react";

interface SearchAreaProps {
  onTextSearch: (query: string, topK: number) => void;
  onImageSearch: (file: File, topK: number) => void;
  isLoading: boolean;
}

export default function SearchArea({
  onTextSearch,
  onImageSearch,
  isLoading,
}: SearchAreaProps) {
  const [textQuery, setTextQuery] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [topK, setTopK] = useState(20);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setTextQuery("");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clearImage = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [imagePreview]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (imageFile) {
        onImageSearch(imageFile, topK);
      } else if (textQuery.trim()) {
        onTextSearch(textQuery.trim(), topK);
      }
    },
    [imageFile, textQuery, topK, onImageSearch, onTextSearch]
  );

  const canSubmit = !isLoading && (!!imageFile || textQuery.trim().length > 0);

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      {/* ------------------------------------------------------------------ */}
      {/* Text search input                                                   */}
      {/* ------------------------------------------------------------------ */}
      <div className="relative flex items-center group">
        <span className="absolute left-4 text-slate-400 pointer-events-none transition-colors group-focus-within:text-slate-600">
          <Search size={17} />
        </span>
        <input
          id="text-search-input"
          type="text"
          placeholder="Describe an image to search for..."
          value={textQuery}
          onChange={(e) => {
            setTextQuery(e.target.value);
            if (imageFile) clearImage();
          }}
          disabled={isLoading || !!imageFile}
          className="
            w-full pl-11 pr-32 py-3.5 rounded-xl
            border border-slate-200 bg-white
            text-sm text-slate-800 placeholder-slate-400
            shadow-sm
            outline-none
            focus:border-indigo-300 focus:ring-3 focus:ring-indigo-100
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-all duration-200
          "
        />
        <div className="absolute right-2 flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowSettings((v) => !v)}
            title="Search settings"
            className={`
              p-2 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100
              transition-colors
              ${showSettings ? "bg-slate-100 text-slate-600" : ""}
            `}
          >
            <SlidersHorizontal size={15} />
          </button>
          <button
            type="submit"
            disabled={!canSubmit}
            className="
              px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg
              hover:bg-indigo-700 active:scale-95
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-150
              shadow-sm
            "
          >
            Search
          </button>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Settings panel                                                      */}
      {/* ------------------------------------------------------------------ */}
      {showSettings && (
        <div className="mt-2 px-4 py-3.5 bg-white border border-slate-200 rounded-xl shadow-sm">
          <label className="flex items-center justify-between text-sm">
            <div>
              <span className="text-slate-700 font-medium">Results to return</span>
              <p className="text-xs text-slate-400 mt-0.5">Maximum number of results shown per query</p>
            </div>
            <div className="flex items-center gap-3 ml-6">
              <input
                type="range"
                min={5}
                max={50}
                step={5}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-28"
              />
              <span className="text-slate-800 font-semibold tabular-nums w-6 text-right">
                {topK}
              </span>
            </div>
          </label>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Divider                                                             */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex items-center gap-3 my-5">
        <div className="flex-1 h-px bg-slate-200" />
        <span className="text-xs text-slate-400 font-medium tracking-wide">
          or upload an image
        </span>
        <div className="flex-1 h-px bg-slate-200" />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Image preview / Drop zone                                           */}
      {/* ------------------------------------------------------------------ */}
      {imagePreview ? (
        <div className="flex items-center gap-4 px-4 py-3.5 bg-white border border-slate-200 rounded-xl shadow-sm">
          <img
            src={imagePreview}
            alt="Query preview"
            className="w-14 h-14 object-cover rounded-lg border border-slate-200 shrink-0"
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-slate-700 truncate">
              {imageFile?.name}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">
              Ready — click Search to find similar images
            </p>
          </div>
          <button
            type="button"
            onClick={clearImage}
            className="shrink-0 p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X size={15} />
          </button>
        </div>
      ) : (
        <div
          id="image-drop-zone"
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative flex flex-col items-center justify-center gap-3 py-9
            border-2 border-dashed rounded-xl cursor-pointer
            transition-all duration-200 overflow-hidden
            ${isDragging
              ? "border-indigo-400 bg-indigo-50/60"
              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60"
            }
          `}
        >
          <div className={`
            flex items-center justify-center w-10 h-10 rounded-xl transition-colors duration-200
            ${isDragging ? "bg-indigo-100 text-indigo-500" : "bg-slate-100 text-slate-400"}
          `}>
            <Upload size={20} />
          </div>
          <div className="text-center">
            <p className={`text-sm font-medium transition-colors duration-200 ${isDragging ? "text-indigo-700" : "text-slate-600"}`}>
              {isDragging ? "Release to upload" : "Drag and drop an image"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              or click to browse &mdash; JPEG, PNG, WebP
            </p>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </form>
  );
}
