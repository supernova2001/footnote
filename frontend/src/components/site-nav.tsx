"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export const NAV = [
  { href: "/", label: "Ask" },
  { href: "/write", label: "Write" },
  { href: "/chat", label: "Chat" },
  { href: "/scholars", label: "Scholars" },
  { href: "/evals", label: "Evals" },
  { href: "/data", label: "Data" },
] as const;

function isActive(active: string | null, href: string) {
  return active !== null && (active === href || active.startsWith(`${href}/`));
}

/** Desktop links. Rendered without an active item as the Suspense fallback. */
export function NavLinks({ active }: { active: string | null }) {
  return (
    <nav aria-label="Main" className="ml-auto hidden md:block">
      <ul className="flex items-center gap-1 text-sm">
        {NAV.map((item) => {
          const current = isActive(active, item.href);
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={current ? "page" : undefined}
                className={`rounded px-2.5 py-1.5 transition-colors hover:bg-sunk ${
                  current ? "font-medium text-ink underline decoration-accent decoration-2 underline-offset-[6px]" : "text-muted"
                }`}
              >
                {item.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [openedAt, setOpenedAt] = useState<string | null>(null);
  // Close the phone menu whenever the route changes.
  const menuOpen = open && openedAt === pathname;

  return (
    <>
      <NavLinks active={pathname} />
      <div className="ml-auto md:hidden">
        <button
          type="button"
          className="rounded px-2.5 py-1.5 text-sm text-muted hover:bg-sunk"
          aria-expanded={menuOpen}
          aria-controls="mobile-nav"
          onClick={() => {
            setOpen(!menuOpen);
            setOpenedAt(pathname);
          }}
        >
          Menu
        </button>
        {menuOpen && (
          <nav
            id="mobile-nav"
            aria-label="Main"
            className="absolute inset-x-0 top-14 border-b border-rule bg-paper px-4 pb-3 shadow-sm"
          >
            <ul className="flex flex-col text-base">
              {NAV.map((item) => {
                const current = isActive(pathname, item.href);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      aria-current={current ? "page" : undefined}
                      onClick={() => setOpen(false)}
                      className={`block rounded px-2 py-2.5 hover:bg-sunk ${current ? "font-medium text-accent" : "text-ink"}`}
                    >
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        )}
      </div>
    </>
  );
}
