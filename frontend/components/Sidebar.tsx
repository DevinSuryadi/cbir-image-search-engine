"use client";

import { HistoryEntry } from "@/lib/types";
import { Clock, ImageIcon, Type, Trash2, X } from "lucide-react";

interface SidebarProps {
  history: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

function groupByDate(entries: HistoryEntry[]): Record<string, HistoryEntry[]> {
  const groups: Record<string, HistoryEntry[]> = {};
  const now = new Date();

  for (const entry of entries) {
    const date = new Date(entry.timestamp);
    const diffDays = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
    );
    let label: string;
    if (diffDays === 0) label = "Today";
    else if (diffDays === 1) label = "Yesterday";
    else if (diffDays < 7) label = "This Week";
    else label = "Older";

    if (!groups[label]) groups[label] = [];
    groups[label].push(entry);
  }
  return groups;
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Sidebar({
  history,
  onSelect,
  onRemove,
  onClear,
  isOpen,
  onToggle,
}: SidebarProps) {
  const grouped = groupByDate(history);
  const groupOrder = ["Today", "Yesterday", "This Week", "Older"];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-20 lg:hidden backdrop-blur-sm"
          onClick={onToggle}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={`
          fixed top-0 left-0 h-full z-30 flex flex-col
          w-68 bg-slate-50 border-r border-slate-200/80
          transition-transform duration-300 ease-in-out
          lg:static lg:translate-x-0 lg:z-auto
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
        `}
        style={{ width: "272px" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200/80">
          <div className="flex items-center gap-2">
            <Clock size={14} className="text-slate-400" />
            <span className="text-xs font-semibold text-slate-500 tracking-widest uppercase">
              History
            </span>
          </div>
          <div className="flex items-center gap-1">
            {history.length > 0 && (
              <button
                onClick={onClear}
                title="Clear all history"
                className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-200 transition-colors"
              >
                <Trash2 size={13} />
              </button>
            )}
            <button
              onClick={onToggle}
              className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-200 transition-colors lg:hidden"
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* History list */}
        <div className="flex-1 overflow-y-auto px-3 py-3">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center px-4">
              <div className="w-10 h-10 rounded-xl bg-slate-200/80 flex items-center justify-center mb-3">
                <Clock size={18} className="text-slate-400" />
              </div>
              <p className="text-sm font-medium text-slate-500">No history yet</p>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Your searches will appear here after you run a query.
              </p>
            </div>
          ) : (
            groupOrder.map((label) => {
              const entries = grouped[label];
              if (!entries || entries.length === 0) return null;
              return (
                <div key={label} className="mb-5">
                  <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest px-2 mb-1.5">
                    {label}
                  </p>
                  <ul className="space-y-0.5">
                    {entries.map((entry) => (
                      <li key={entry.id} className="group relative">
                        <button
                          onClick={() => onSelect(entry)}
                          className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-left hover:bg-white hover:shadow-sm border border-transparent hover:border-slate-200/80 transition-all duration-150"
                        >
                          <span className="mt-0.5 shrink-0 text-slate-400">
                            {entry.mode === "text" ? (
                              <Type size={12} />
                            ) : (
                              <ImageIcon size={12} />
                            )}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-slate-700 truncate leading-snug font-medium">
                              {entry.query}
                            </p>
                            <p className="text-xs text-slate-400 mt-0.5">
                              {entry.resultCount} results &middot;{" "}
                              {formatTime(entry.timestamp)}
                            </p>
                          </div>
                        </button>
                        <button
                          onClick={() => onRemove(entry.id)}
                          title="Remove"
                          className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded text-slate-300 hover:text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <X size={11} />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })
          )}
        </div>
      </aside>
    </>
  );
}
