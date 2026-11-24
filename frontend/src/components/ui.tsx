import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";

export function Container({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`mx-auto w-full max-w-6xl px-4 sm:px-6 ${className}`}>{children}</div>;
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <div className="pb-6 pt-8 sm:pt-10">
      {eyebrow && <p className="mb-1 font-mono text-xs uppercase tracking-[0.14em] text-muted">{eyebrow}</p>}
      <h1 className="font-serif text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
      {children && <div className="mt-2 max-w-3xl text-muted">{children}</div>}
    </div>
  );
}

type ButtonProps = ComponentProps<"button"> & { variant?: "primary" | "secondary" | "ghost" };

export function Button({ variant = "primary", className = "", ...props }: ButtonProps) {
  const styles = {
    primary: "bg-ink text-paper hover:opacity-90 disabled:opacity-40",
    secondary: "border border-rule bg-surface text-ink hover:bg-sunk disabled:opacity-50",
    ghost: "text-muted hover:bg-sunk hover:text-ink disabled:opacity-50",
  }[variant];
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex min-h-9 items-center justify-center gap-2 rounded-md px-3.5 text-sm font-medium transition disabled:cursor-not-allowed ${styles} ${className}`}
    />
  );
}

export function ButtonLink({ href, children, variant = "secondary" }: { href: string; children: ReactNode; variant?: "primary" | "secondary" }) {
  const styles =
    variant === "primary" ? "bg-ink text-paper hover:opacity-90" : "border border-rule bg-surface text-ink hover:bg-sunk";
  return (
    <Link
      href={href}
      className={`inline-flex min-h-9 items-center justify-center gap-2 rounded-md px-3.5 text-sm font-medium transition ${styles}`}
    >
      {children}
    </Link>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-lg border border-rule bg-surface ${className}`}>{children}</div>;
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "bad"; children: ReactNode }) {
  const styles = {
    info: "border-rule bg-sunk text-ink",
    warn: "border-warn/40 bg-warn-soft text-ink",
    bad: "border-bad/40 bg-bad-soft text-ink",
  }[tone];
  return (
    <div role={tone === "bad" ? "alert" : "status"} className={`rounded-md border px-3.5 py-2.5 text-sm ${styles}`}>
      {children}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-rule px-5 py-8 text-center">
      <p className="font-serif text-lg font-medium">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-xl text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role="status">
      <span className="pulse inline-block h-2 w-2 rounded-full bg-accent" aria-hidden="true" />
      {label}…
    </span>
  );
}

export function Tag({ children, tone = "neutral", title }: { children: ReactNode; tone?: "neutral" | "accent"; title?: string }) {
  const styles = tone === "accent" ? "border-accent/30 bg-accent-soft text-accent" : "border-rule bg-sunk text-muted";
  return (
    <span title={title} className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[11px] leading-none ${styles}`}>
      {children}
    </span>
  );
}

export function Field({ label, htmlFor, hint, children }: { label: string; htmlFor: string; hint?: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  );
}

export const inputClass =
  "w-full rounded-md border border-rule bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-focus focus:outline-none focus-visible:outline-2 focus-visible:outline-focus";
