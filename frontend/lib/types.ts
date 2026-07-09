// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

export interface SearchResultItem {
  point_id: string;
  image_url: string;
  category: string;
  score: number;
}

export interface SearchResponse {
  results: SearchResultItem[];
  query_seconds: number;
  result_count: number;
}

// ---------------------------------------------------------------------------
// Search history types
// ---------------------------------------------------------------------------

export type SearchMode = "text" | "image";

export interface HistoryEntry {
  id: string;
  mode: SearchMode;
  query: string;           // text query or filename for image uploads
  timestamp: number;
  resultCount: number;
}

// ---------------------------------------------------------------------------
// Component prop types
// ---------------------------------------------------------------------------

export interface SearchState {
  mode: SearchMode;
  textQuery: string;
  imageFile: File | null;
  imagePreviewUrl: string | null;
  topK: number;
}
