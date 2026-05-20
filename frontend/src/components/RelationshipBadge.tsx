import type { RelationshipType } from "../types/analysis";

const relationshipMap: Record<
  RelationshipType,
  { label: string; className: string }
> = {
  Reinforcing: {
    label: "Reinforcing",
    className: "bg-rose-100 text-rose-700 border-rose-200",
  },
  Contrastive: {
    label: "Contrastive",
    className: "bg-sky-100 text-sky-700 border-sky-200",
  },
  Supplementary: {
    label: "Supplementary",
    className: "bg-slate-100 text-slate-700 border-slate-200",
  },
  "": {
    label: "Pending",
    className: "bg-slate-100 text-slate-500 border-slate-200",
  },
};

export function RelationshipBadge({ value }: { value: RelationshipType }) {
  const meta = relationshipMap[value];

  return (
    <span
      className={`inline-flex max-w-full items-center rounded-full border px-3 py-1 text-sm font-medium ${meta.className}`}
    >
      {meta.label}
    </span>
  );
}
