import { SearchResultItem, SearchResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Search by text description
// ---------------------------------------------------------------------------

export async function searchByText(
  query: string,
  topK: number = 20
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Search by uploaded image file
// ---------------------------------------------------------------------------

export async function searchByImage(
  file: File,
  topK: number = 20
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(topK));

  const response = await fetch(`${API_BASE_URL}/api/search/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Find images similar to a reference image URL (More Like This)
// ---------------------------------------------------------------------------

export async function searchMoreLikeThis(
  imageUrl: string,
  topK: number = 20
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search/more-like-this`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_url: imageUrl, top_k: topK }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}
