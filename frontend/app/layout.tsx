import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CBIR Image Search Engine",
  description:
    "Content-Based Image Retrieval using CLIP embeddings and Qdrant vector search. Search the Caltech-101 dataset by text or image.",
  keywords: ["image search", "CBIR", "CLIP", "Qdrant", "Caltech-101"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
