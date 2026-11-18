import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MemoryPal Agent — Autonomous Cognitive Learning OS",
  description: "Grounded multimodal AI learning platform with spaced repetition, active recall, and adaptive mastery.",
  icons: {
    icon: "/icon.png",
    apple: "/icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body className="min-h-screen bg-[#030712] text-slate-100 antialiased selection:bg-teal-500/30 selection:text-teal-200">
        {children}
      </body>
    </html>
  );
}
