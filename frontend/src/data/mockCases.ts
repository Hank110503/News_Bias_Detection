import type { CaseRecord } from "../types/analysis";

export const MOCK_CASES: CaseRecord[] = [
  {
    newsId: "20575",
    title:
      "Russia - Ukraine crisis : Bethenny Frankel says her BStrong Foundation is hoping to help 1M refugees",
    imageHint: "bethenny-frankel-getty.jpg",
    articleText:
      "Bethenny Frankel said her BStrong Foundation is hoping to help one million refugees affected by the Ukraine crisis, expanding funding and logistics for relief support.",
    result: {
      observations: {
        visual_observation:
          "A close-up of Bethenny Frankel speaking in a studio-like setting. The frame emphasizes her facial expression and isolates her from broader context.",
        textual_observation:
          "The text focuses on Frankel's humanitarian effort for Ukrainian refugees. Terms like 'refugees' carry sympathy framing, and the conflict is tied to the Russian invasion.",
        joint_mechanism:
          "The image personalizes the charitable effort by centering Frankel, while the text explains the aid narrative. The combination supports a human-centered reading of the report.",
      },
      final_result: {
        cues: {
          V1_Salience: {
            reason:
              "The image centers one speaker in a tight shot but does not depict extreme suffering or triumph.",
            present: false,
            score: 0,
          },
          V2_Color_Polarity: {
            reason:
              "Color appears natural for an indoor interview setting without deliberate emotional processing.",
            present: false,
            score: 0,
          },
          V3_Power_Angle: {
            reason:
              "The shot is close and slightly authoritative but not strongly manipulative in angle.",
            present: false,
            score: 0,
          },
          V4_Visual_Selectivity: {
            reason:
              "The frame excludes refugees and wider context, keeping attention on the spokesperson figure.",
            present: false,
            score: 0,
          },
          T1_Agent_Label: {
            reason:
              "The term 'refugees' introduces a sympathy-oriented label for the affected group.",
            present: true,
            score: 1,
          },
          T2_Causal_Attribution: {
            reason:
              "The text links humanitarian need to the Russian invasion without extended sourcing inside the short report.",
            present: true,
            score: 1,
          },
          M1_Affect_Mismatch: {
            reason:
              "The article is mainly factual and the image remains restrained, so strong affect mismatch is not activated.",
            present: false,
            score: 0,
          },
          M2_Binary_Roles: {
            reason:
              "The text shows sympathetic framing, but the image does not visually reinforce a victim-aggressor binary.",
            present: false,
            score: 0,
          },
          M3_Symbol_Decontex: {
            reason:
              "No obvious symbolic object in the image requires extra context from the text.",
            present: false,
            score: 0,
          },
        },
        consistency_dimension: {
          reason:
            "The text is sympathetic toward refugees and positive toward relief efforts. The image is a neutral portrait of the spokesperson, so the relationship is supplementary.",
          D1_Relationship_Type: "Supplementary",
        },
      },
    },
  },
  {
    newsId: "3587",
    title: "US Warns Russia Over Potential War Crimes in Ukraine",
    imageHint: "093a0000-0a00-0242-70dc-08da02f7d3cc_w1200_r1.jpg",
    articleText:
      "The report discusses sanctions and policy pressure on Russia after the invasion of Ukraine, while showing destruction at the conflict site.",
    result: {
      observations: {
        visual_observation:
          "A uniformed person examines a missile amid destroyed urban ruins. The frame emphasizes debris and destruction.",
        textual_observation:
          "The text presents Russia's actions with strongly negative framing and discusses sanctions and possible war crimes.",
        joint_mechanism:
          "The image provides emotionally charged destruction imagery that amplifies the article's critical framing of Russia.",
      },
      final_result: {
        cues: {
          V1_Salience: {
            reason:
              "Destruction and debris are visually central, making the image emotionally salient.",
            present: true,
            score: 2,
          },
          V2_Color_Polarity: {
            reason:
              "Muted grays and browns follow the natural look of a ruined urban scene.",
            present: false,
            score: 0,
          },
          V3_Power_Angle: {
            reason:
              "The low framing around the uniformed person adds a mild authority effect.",
            present: true,
            score: 2,
          },
          V4_Visual_Selectivity: {
            reason:
              "The composition foregrounds the missile and immediate wreckage instead of broader civilian context.",
            present: true,
            score: 2,
          },
          T1_Agent_Label: {
            reason:
              "The report uses explicitly negative descriptors for Russia's actions.",
            present: true,
            score: 1,
          },
          T2_Causal_Attribution: {
            reason:
              "The article assigns responsibility for invasion and aggression to Russia and Putin in direct terms.",
            present: true,
            score: 2,
          },
          M1_Affect_Mismatch: {
            reason:
              "Policy-heavy reporting is amplified by an emotionally stronger war-destruction image.",
            present: true,
            score: 1,
          },
          M2_Binary_Roles: {
            reason:
              "Text and image together support an aggressor-victim narrative structure.",
            present: true,
            score: 1,
          },
          M3_Symbol_Decontex: {
            reason:
              "The image does not rely on a symbol whose meaning is left unexplained by the article.",
            present: false,
            score: 0,
          },
        },
        consistency_dimension: {
          reason:
            "Both modalities move in the same negative direction toward the conflict event and its perpetrators.",
          D1_Relationship_Type: "Reinforcing",
        },
      },
    },
  },
];

