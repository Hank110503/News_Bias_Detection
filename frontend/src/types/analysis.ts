export type CueCode =
  | "V1_Salience"
  | "V2_Color_Polarity"
  | "V3_Power_Angle"
  | "V4_Visual_Selectivity"
  | "T1_Agent_Label"
  | "T2_Causal_Attribution"
  | "M1_Affect_Mismatch"
  | "M2_Binary_Roles"
  | "M3_Symbol_Decontex";

export type RelationshipType =
  | "Reinforcing"
  | "Contrastive"
  | "Supplementary"
  | "";

export type CueItem = {
  reason: string;
  present: boolean;
  score: number;
};

export type CueMap = Partial<Record<CueCode, CueItem>>;

export type Stage1Result = {
  visual_observation: string;
  textual_observation: string;
  joint_mechanism: string;
};

export type Stage2Result = {
  cues: CueMap;
  consistency_dimension: {
    reason: string;
    D1_Relationship_Type: RelationshipType;
  };
};

export type FullAnalysisResult = {
  observations: Stage1Result;
  final_result: Stage2Result;
};

export type AnalysisInput = {
  articleText: string;
  imagePath?: string;
  imageName?: string;
  imageDataBase64?: string;
  sampleCase?: CaseRecord | null;
};

export type AnalysisStreamEvent =
  | { type: "status"; stage: string; message: string }
  | { type: "partial"; stage: "stage1"; data: Stage1Result; sourceLabel?: string }
  | { type: "final"; data: FullAnalysisResult; sourceLabel?: string }
  | { type: "error"; message: string };

export type LocalInferenceRuntime = {
  mode: string;
  repoRoot: string;
  pythonPath: string;
  adapterPath: string;
  modelName: string;
};

export type LocalInferenceResponse = {
  result: FullAnalysisResult;
  runtime: LocalInferenceRuntime;
};

export type CaseRecord = {
  newsId: string;
  title: string;
  imageHint: string;
  imagePath?: string;
  articleText: string;
  sourceUrl?: string;
  sourceIndex?: number;
  observationSource?: string;
  observationDisplaySource?: string;
  goldObservations?: Stage1Result;
  result: FullAnalysisResult;
};

export type EvaluationMetrics = {
  jsonParseRate: number;
  matchedRows: number;
  cuePresentMacroF1: number;
  cueScoreAccuracy: number;
  cueScoreMae: number;
  relationMacroF1: number;
  cueReasonNonEmptyRate: number;
  consistencyReasonNonEmptyRate: number;
};

export type EvaluationDataset = {
  generatedAt: string;
  totalCases: number;
  cases: CaseRecord[];
};

export type RecentAnalysisRecord = {
  id: string;
  createdAt: string;
  title: string;
  imageName?: string;
  relationship: RelationshipType;
  activeCueCount: number;
  result: FullAnalysisResult;
};
