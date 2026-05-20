import type { CueCode } from "../types/analysis";

export const CUE_ORDER: CueCode[] = [
  "V1_Salience",
  "V2_Color_Polarity",
  "V3_Power_Angle",
  "V4_Visual_Selectivity",
  "T1_Agent_Label",
  "T2_Causal_Attribution",
  "M1_Affect_Mismatch",
  "M2_Binary_Roles",
  "M3_Symbol_Decontex",
];

export const CUE_META: Record<
  CueCode,
  { label: string; group: "Visual" | "Textual" | "Multimodal" }
> = {
  V1_Salience: { label: "Subject salience", group: "Visual" },
  V2_Color_Polarity: { label: "Color polarity", group: "Visual" },
  V3_Power_Angle: { label: "Power angle", group: "Visual" },
  V4_Visual_Selectivity: { label: "Visual selectivity", group: "Visual" },
  T1_Agent_Label: { label: "Agent labeling", group: "Textual" },
  T2_Causal_Attribution: { label: "Causal attribution", group: "Textual" },
  M1_Affect_Mismatch: { label: "Affect-fact mismatch", group: "Multimodal" },
  M2_Binary_Roles: { label: "Binary role framing", group: "Multimodal" },
  M3_Symbol_Decontex: { label: "Symbol decontextualization", group: "Multimodal" },
};

export const CUE_GROUPS = [
  {
    key: "Visual" as const,
    label: "Visual Cues",
    description: "How the image introduces bias through subject choice, angle, color, and framing.",
  },
  {
    key: "Textual" as const,
    label: "Textual Cues",
    description: "How the text shapes stance through labels and causal framing.",
  },
  {
    key: "Multimodal" as const,
    label: "Multimodal Cues",
    description: "Whether the image-text combination introduces mismatch, binary roles, or symbolic decontextualization.",
  },
];

export const CUE_GROUP_COLORS = {
  Visual: "from-visual/15 to-visual/5 text-visual",
  Textual: "from-textual/15 to-textual/5 text-textual",
  Multimodal: "from-multimodal/15 to-multimodal/5 text-multimodal",
};
