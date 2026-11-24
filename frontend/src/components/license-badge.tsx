import { LICENSE_NAME, LICENSE_SHORT } from "@/lib/format";
import type { License, LicenseId } from "@/lib/types";

export function LicenseBadge({ id, license }: { id?: LicenseId; license?: License }) {
  const lid = license?.id ?? id;
  if (!lid) return null;
  const title = license?.name ?? LICENSE_NAME[lid];
  const pd = lid === "PD-US" || lid === "CC0-1.0";
  const body = (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[11px] leading-none ${
        pd ? "border-ok/30 bg-ok-soft text-ok" : "border-rule bg-sunk text-muted"
      }`}
    >
      <span aria-hidden="true">{pd ? "◯" : "©"}</span>
      <span className="sr-only">Licence: </span>
      {LICENSE_SHORT[lid]}
    </span>
  );
  if (license?.url) {
    return (
      <a href={license.url} target="_blank" rel="noopener noreferrer" className="rounded hover:opacity-80">
        {body}
      </a>
    );
  }
  return body;
}

/** The attribution line a reader sees next to text derived from a passage. */
export function attributionFor(licenseId: LicenseId, title: string): string {
  switch (licenseId) {
    case "PD-US":
      return `“${title}”: public domain in the United States.`;
    case "CC0-1.0":
      return `“${title}”: dedicated to the public domain (CC0 1.0).`;
    case "CC-BY-4.0":
      return `From “${title}”, licensed CC BY 4.0; credit the authors when reusing it.`;
    case "CC-BY-SA-4.0":
      return `From “${title}”, licensed CC BY-SA 4.0; derived text is shared under the same licence.`;
  }
}
