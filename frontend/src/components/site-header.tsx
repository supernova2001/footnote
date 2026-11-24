import Link from "next/link";
import { Suspense } from "react";
import { NavLinks, SiteNav } from "./site-nav";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-paper/95 backdrop-blur supports-[backdrop-filter]:bg-paper/80">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="group flex items-baseline gap-0.5 rounded font-serif text-xl font-semibold tracking-tight">
          Footnote
          <sup className="font-sans text-xs font-semibold text-accent group-hover:underline">1</sup>
        </Link>
        <Suspense fallback={<NavLinks active={null} />}>
          <SiteNav />
        </Suspense>
        <div className="flex items-center">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
